"""
Continually fine-tune a causal language model.

The model is fine-tuned to auto-regressively predict the next token in the input sequence (as a GPT-like decoder).
    - We do not fine-tune the model specifically for API call or API usage prediction. We found out
        that these strategies provide too few samples for the model to converge without overfitting.

We use early stopping for each experience with a patience argument.
    - For validation, we compute the validation loss by obfuscating all tokens that are not OOD API calls
        (see `generate_validation_samples` function).
    - The model only has access to validation samples of the current experience
        (see `EarlyStoppingPlugin` with metric_name='Loss_Exp' and criteria='exp')
    - We store the best model checkpoint for each experience.
"""
import logging
import multiprocessing
import os
from itertools import chain
from typing import List

import avalanche
import omegaconf
import pandas as pd
import torch
import transformers
from avalanche.benchmarks import CLScenario, CLStream, CLExperience
from avalanche.benchmarks.utils import DataAttribute, ConstantSequence, AvalancheDataset
from avalanche.evaluation.metrics import loss_metrics
from avalanche.training import Naive, JointTraining, Cumulative
from avalanche.training.plugins import SynapticIntelligencePlugin, EWCPlugin, LwFPlugin
from datasets import Dataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    default_data_collator
)

from tasks.plugins.early_stopping import EarlyStoppingPlugin

logger = logging.getLogger(__name__)


def generate_validation_samples(example):
    samples = []
    try:
        # split api usages
        api_usages = list(filter(lambda e: e != '', example['api_seq'].split('|')))
        api_usages = [u.split('.') for u in api_usages]
        # remove java wildcards (e.g., ArrayList<String> -> ArrayList)
        #   and get a list of api usage in the form of [api_class, api_call, index_in_source_code]
        api_usages = list(map(lambda e: [e[0].strip().split(' ')[0],
                                         e[1].strip().split(' ')[0],
                                         e[1].strip().split(' ')[1]], api_usages))
    except IndexError:
        return samples

    for usage in api_usages:
        try:
            # ignore API initialization samples
            if usage[1] == '<init>':
                continue
            # OOD data -> ignore APIs that are not OOD
            if usage[0] != example['api']:
                continue

            api_call = usage[1]
            start_token_idx = int(usage[2])
            end_token_idx = start_token_idx + len(api_call)
            api_call_in_code = example['source_code'][start_token_idx:end_token_idx]

            # ensure the usage and the source code match
            if api_call != api_call_in_code:
                continue

            new_sample = {
                'context': example['source_code'][:start_token_idx],
                'ground_truth': api_call
            }
            samples.append(new_sample)
        except:
            pass
    return samples


def preprocess_dataset(cfg, dataset, tokenizer):
    def tokenize_train_ds(examples):
        return tokenizer(examples['source_code'])

    # 90% training / 10% validation
    n_valid_samples = int(len(dataset) * 0.10)
    valid_ds = dataset.select(list(range(n_valid_samples)))
    train_ds = dataset.select(list(range(n_valid_samples, len(dataset))))

    train_dataset = train_ds.map(
        tokenize_train_ds,
        batched=True,
        num_proc=cfg.run.preprocessing_num_workers,
        load_from_cache_file=True,
        remove_columns=[cname for cname in train_ds.column_names if cname not in ['input_ids', 'labels']],
        desc="Running tokenizer on training dataset",
    )

    logger.info("Generating validation samples.")
    with multiprocessing.Pool(cfg.run.preprocessing_num_workers) as pool:
        results = list(tqdm(pool.imap(generate_validation_samples, iter(valid_ds)), total=len(valid_ds)))
    results = [item for sublist in results for item in sublist]
    valid_dataset = Dataset.from_pandas(pd.DataFrame(results))
    valid_dataset = valid_dataset.filter(
        lambda e: len(tokenizer(e['ground_truth']).input_ids) <= cfg.run.max_new_tokens)

    def convert_sample_to_features(example):
        context_tokenized = tokenizer(example['context'], add_special_tokens=False).input_ids
        ground_truth_tokenized = tokenizer(example['ground_truth'], add_special_tokens=False).input_ids
        ground_truth_len = len(ground_truth_tokenized)

        max_context_len = tokenizer.model_max_length - ground_truth_len
        context_tokenized = context_tokenized[-max_context_len:]
        context_len = len(context_tokenized)
        padding_len = tokenizer.model_max_length - context_len - ground_truth_len

        labels_padding = [-100] * padding_len
        labels_context = [-100] * context_len
        labels = labels_context + ground_truth_tokenized + labels_padding
        input_ids = context_tokenized + ground_truth_tokenized + [tokenizer.pad_token_id] * padding_len

        assert len(input_ids) == tokenizer.model_max_length == len(labels)
        return {'input_ids': input_ids, 'labels': labels}

    valid_dataset = valid_dataset.map(
        convert_sample_to_features,
        batched=False,
        num_proc=cfg.run.preprocessing_num_workers,
        load_from_cache_file=True,
        desc="Creating validation samples features.",
    )

    block_size = tokenizer.model_max_length

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result['labels'] = result['input_ids'].copy()
        return result

    train_dataset = train_dataset.map(
        group_texts,
        batched=True,
        num_proc=cfg.run.preprocessing_num_workers,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = train_dataset.remove_columns(
        [cname for cname in train_dataset.column_names if cname not in ['input_ids', 'labels']])
    valid_dataset = valid_dataset.remove_columns(
        [cname for cname in valid_dataset.column_names if cname not in ['input_ids', 'labels']])

    return train_dataset, valid_dataset


class BaseHGStrategy:
    @property
    def mb_x(self):
        """Current mini-batch input."""
        return self.mbatch['input_ids']

    @property
    def mb_y(self):
        """Current mini-batch target."""
        return self.mbatch['labels']

    def _unpack_minibatch(self):
        """HuggingFace minibatches are dictionaries of tensors.
        Move tensors to the current device."""
        for k in self.mbatch.keys():
            self.mbatch[k] = self.mbatch[k].to(self.device)

    def forward(self):
        out = self.model(
            input_ids=self.mb_x,
            labels=self.mb_y
        )
        return out

    def criterion(self):
        mb_output = self.mb_output
        return mb_output.loss


class HGNaive(BaseHGStrategy, Naive):
    pass


class HGJoint(BaseHGStrategy, JointTraining):
    pass


class HGCumulative(BaseHGStrategy, Cumulative):
    pass


def finetune(cfg: omegaconf.DictConfig,
             model: transformers.AutoModelForCausalLM,
             tokenizer: transformers.AutoTokenizer,
             datasets: List[torch.utils.data.Dataset]):
    train_datasets, valid_datasets = [], []
    for i, dataset in enumerate(datasets):
        logger.info(f"Preprocessing OOD dataset #{i}")
        train_data, valid_data = preprocess_dataset(cfg, dataset, tokenizer)
        train_datasets.append(train_data)
        valid_datasets.append(valid_data)

    logger.info("Creating continual learning experiences.")
    train_exps, valid_exps = [], []
    for i, (train_data, valid_data) in enumerate(zip(train_datasets, valid_datasets)):
        tl_train = DataAttribute(
            ConstantSequence(i, len(train_data)), 'targets_task_labels'
        )
        exp_train = CLExperience()
        exp_train.dataset = AvalancheDataset(
            [train_data], data_attributes=[tl_train], collate_fn=default_data_collator
        )

        tl_valid = DataAttribute(
            ConstantSequence(i, len(valid_data)), 'targets_task_labels'
        )
        exp_valid = CLExperience()
        exp_valid.dataset = AvalancheDataset(
            [valid_data], data_attributes=[tl_valid], collate_fn=default_data_collator
        )

        train_exps.append(exp_train), valid_exps.append(exp_valid)

    benchmark = CLScenario(
        [
            CLStream('train', train_exps),
            CLStream('valid', valid_exps)
        ]
    )

    loggers = []
    loggers.append(avalanche.logging.InteractiveLogger())
    if cfg.use_wandb:
        wandb_logger = avalanche.logging.WandBLogger(
            project_name='cl-code',
            run_name=f'{cfg.model.model_name}_ft_{cfg.run.strategy}',
            config=vars(cfg)
        )
        loggers.append(wandb_logger)

    eval_plugin = avalanche.training.plugins.EvaluationPlugin(
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=loggers,
        strict_checks=False,
    )

    if cfg.run.strategy == 'joint':
        # @todo: debug joint fine-tuning
        strategy_cls = HGJoint
    elif cfg.run.strategy == 'cumulative':
        strategy_cls = HGCumulative
    else:
        # for all other strategies, we add CL plugins to the strategy
        strategy_cls = HGNaive

    optimizer = AdamW(model.parameters(), lr=cfg.run.learning_rate)
    early_stopping = EarlyStoppingPlugin(patience=cfg.run.patience, val_stream_name='valid_stream',
                                         metric_name='Loss_Exp', mode='min', criteria='exp', verbose=True)
    plugins = [early_stopping]
    if cfg.run.strategy == 'si':
        logger.info("Fine-tuning with Synaptic intelligence (si).")
        cl_plugin = SynapticIntelligencePlugin(si_lambda=cfg.run.si_lambda, eps=cfg.run.si_eps)
        plugins.append(cl_plugin)
    elif cfg.run.strategy == 'ewc':
        logger.info("Fine-tuning with EWC.")
        cl_plugin = EWCPlugin(ewc_lambda=cfg.run.ewc_lambda)
        plugins.append(cl_plugin)
    elif cfg.run.strategy == 'lwf':
        logger.info("Fine-tuning with EWC.")
        cl_plugin = LwFPlugin(alpha=cfg.run.lwf_alpha, temperature=cfg.run.lwf_temperature)
        plugins.append(cl_plugin)

    strategy = strategy_cls(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        evaluator=eval_plugin,
        train_epochs=cfg.run.num_epochs_per_experience,
        train_mb_size=cfg.run.train_batch_size,
        eval_mb_size=cfg.run.valid_batch_size,
        eval_every=1,
        plugins=plugins,
        device=cfg.device
    )

    for exp_id, train_exp in enumerate(benchmark.train_stream):
        strategy.train(train_exp, eval_streams=[benchmark.valid_stream])
        logger.info("Training completed!")

        logger.info(f"Saving best model after exp {exp_id}.")
        checkpoint_prefix = f'exp_{exp_id}'
        # EarlyStoppingPlugin loads the best model after training in the strategy
        strategy.model.save_pretrained(checkpoint_prefix)
        tokenizer.save_pretrained(checkpoint_prefix)
        logger.info(f"New model checkpoint saved {os.path.join(os.getcwd(), checkpoint_prefix)}")
