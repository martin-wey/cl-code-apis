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
from avalanche.training import Naive, JointTraining, Cumulative
from datasets import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    default_data_collator
)

from utils.jdk_apis import JDK_APIS

logger = logging.getLogger(__name__)


def get_api_usage_completion_samples(example):
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
            ground_truth = f'{usage[0]}.{usage[1]}'
            # ignore API initialization samples
            if usage[1] == '<init>':
                continue
            # ignore API calls not part of JDK when fine-tuning on ID dataset
            if example['api'] == 'NaN' and ground_truth not in JDK_APIS:
                continue
            # for OOD data, ignore APIs that are not OOD
            if example['api'] != 'NaN' and usage[0] != example['api']:
                continue

            api_usage_idx = int(usage[2])
            api_interface = usage[0]
            api_call = usage[1]
            api_call_in_code = example['source_code'][api_usage_idx:api_usage_idx + len(api_call)]
            # ensure the usage and the source code match
            if api_call != api_call_in_code:
                continue

            api_interface_start_idx = api_usage_idx - (len(api_interface) + 1)  # +1 for the dot between API and call
            api_interface_in_code = example['source_code'][
                                    api_interface_start_idx:api_interface_start_idx + len(api_interface)]
            # case <API>.<call>(<params>) -> we keep <API>. in the ground-truth.
            if api_interface == api_interface_in_code:
                start_token_idx = api_interface_start_idx
                end_token_idx = start_token_idx + len(api_interface) + 1 + len(api_call)
            # case <var>.<call>(<params>) -> we ignore <var> in the ground-truth.
            else:
                start_token_idx = api_usage_idx
                end_token_idx = start_token_idx + len(api_call)

            open_parenthesis = 0
            n_iter = 0
            while True:
                next_token = example['source_code'][end_token_idx:end_token_idx + 1]
                if next_token == '(':
                    open_parenthesis += 1
                if next_token == ')':
                    open_parenthesis -= 1
                end_token_idx += 1
                n_iter += 1
                if open_parenthesis == 0 or n_iter > 10000:
                    # ill-formed case...
                    break

            if open_parenthesis == 0:
                new_sample = {
                    'source_code': example['source_code'],
                    'context': example['source_code'][:start_token_idx],
                    'ground_truth': example['source_code'][start_token_idx:end_token_idx],
                    'domain': example['domain'],
                    'api': example['api']
                }
                samples.append(new_sample)
        except:
            pass

    return samples


def get_api_call_completion_samples(example):
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
            ground_truth = f'{usage[0]}.{usage[1]}'
            # ignore API initialization samples
            if usage[1] == '<init>':
                continue
            # ignore API calls not part of JDK when fine-tuning on ID dataset
            if example['api'] == 'NaN' and ground_truth not in JDK_APIS:
                continue
            # for OOD data, ignore APIs that are not OOD
            if example['api'] != 'NaN' and usage[0] != example['api']:
                continue

            api_call = usage[1]
            start_token_idx = int(usage[2])
            end_token_idx = start_token_idx + len(api_call)
            api_call_in_code = example['source_code'][start_token_idx:end_token_idx]

            # ensure the usage and the source code match
            if api_call != api_call_in_code:
                continue

            new_sample = {
                'source_code': example['source_code'],
                'context': example['source_code'][:start_token_idx],
                'ground_truth': api_call,
                'domain': example['domain'],
                'api': example['api'],
            }
            samples.append(new_sample)
        except:
            pass
    return samples


def preprocess_dataset(cfg, dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples['source_code'], return_attention_mask=False)

    logger.info("Generating fine-tuning samples.")
    with multiprocessing.Pool(cfg.run.preprocessing_num_workers) as pool:
        if cfg.run.task == 'usage':
            results = list(tqdm(pool.imap(get_api_usage_completion_samples, iter(dataset)), total=len(dataset)))
        elif cfg.run.task == 'call':
            results = list(tqdm(pool.imap(get_api_call_completion_samples, iter(dataset)), total=len(dataset)))
    results = [item for sublist in results for item in sublist]
    dataset = Dataset.from_pandas(pd.DataFrame(results))

    dataset_tokenized = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=cfg.run.preprocessing_num_workers,
        load_from_cache_file=True,
        remove_columns=[cname for cname in dataset.column_names if cname not in
                        ['input_ids', 'attention_mask', 'context', 'ground_truth']],
        desc="Running tokenizer on dataset",
    )

    # filter samples that are too long for the model to generate
    dataset_tokenized = dataset_tokenized.filter(
        lambda e: len(tokenizer(e['ground_truth']).input_ids) <= cfg.run.max_new_tokens)

    def create_labels(example):
        """Create labels by hand to ignore all `ids` that are not part of the API usage in the loss."""
        left_context_len = len(tokenizer(example['context']).input_ids)
        ground_truth_len = len(tokenizer(example['ground_truth']).input_ids)
        right_context_len = len(example['input_ids'][left_context_len + ground_truth_len:])

        labels_context = [-100] * left_context_len
        labels_ground_truth = example['input_ids'][left_context_len:left_context_len + ground_truth_len]
        labels_right = [-100] * right_context_len

        labels = labels_context + labels_ground_truth + labels_right
        return {'labels': labels}

    dataset_tokenized = dataset_tokenized.map(
        create_labels,
        batched=False,
        num_proc=cfg.run.preprocessing_num_workers,
        load_from_cache_file=True,
        remove_columns=[cname for cname in dataset_tokenized.column_names if cname not in
                        ['input_ids', 'attention_mask', 'labels']],
        desc="Creating sample labels.",
    )

    # 90% training / 10% validation
    n_valid_samples = int(len(dataset_tokenized) * 0.10)
    valid_dataset = dataset_tokenized.select(list(range(n_valid_samples)))
    train_dataset = dataset_tokenized.select(list(range(n_valid_samples, len(dataset_tokenized))))

    block_size = tokenizer.model_max_length if tokenizer.model_max_length <= 1024 else 1024

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
        return result

    train_dataset = train_dataset.map(
        group_texts,
        batched=True,
        num_proc=cfg.run.preprocessing_num_workers,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )
    valid_dataset = valid_dataset.map(
        group_texts,
        batched=True,
        num_proc=cfg.run.preprocessing_num_workers,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

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
            labels=self.mb_y,
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


def finetune_decoder(cfg: omegaconf.DictConfig,
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
            run_name=f'{cfg.model.model_name}_ft_{cfg.run.task}_{cfg.run.strategy}',
            config=vars(cfg)
        )
        loggers.append(wandb_logger)

    eval_plugin = avalanche.training.plugins.EvaluationPlugin(
        avalanche.evaluation.metrics.loss_metrics(
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True
        ),
        loggers=loggers,
        strict_checks=False,
    )

    optimizer = AdamW(model.parameters(), lr=cfg.run.learning_rate)

    if cfg.run.strategy == 'joint':
        strategy_cls = HGJoint
    elif cfg.run.strategy == 'cumulative':
        strategy_cls = HGCumulative
    else:
        strategy_cls = HGNaive

    strategy = strategy_cls(
        model=model,
        optimizer=optimizer,
        criterion=torch.nn.CrossEntropyLoss(),
        evaluator=eval_plugin,
        train_epochs=cfg.run.num_epochs_per_experience,
        train_mb_size=cfg.run.train_batch_size,
        eval_mb_size=cfg.run.valid_batch_size,
        eval_every=1,
        device=cfg.device
    )

    for exp_id, train_exp in enumerate(benchmark.train_stream):
        strategy.train(train_exp, eval_streams=[benchmark.valid_stream])
        logger.info("Training completed!")

        logger.info(f"Saving model after exp {exp_id}.")
        checkpoint_prefix = f'exp_{exp_id}'
        model.save_pretrained(checkpoint_prefix)
        tokenizer.save_pretrained(checkpoint_prefix)
        logger.info(f"New model checkpoint saved {os.path.join(os.getcwd(), checkpoint_prefix)}")
