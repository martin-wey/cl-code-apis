import logging
import multiprocessing
import os
from itertools import chain

import numpy as np
import omegaconf
import pandas as pd
import torch
import transformers
import wandb
from datasets import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    default_data_collator,
    get_linear_schedule_with_warmup
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
            # ignore API calls not part of JDK when testing on ID dataset
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
                    'source_code': example['source_code'][:end_token_idx],
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
            # ignore API calls not part of JDK when testing on ID dataset
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
                'source_code': example['source_code'][:end_token_idx],
                'context': example['source_code'][:start_token_idx],
                'ground_truth': api_call,
                'domain': example['domain'],
                'api': example['api'],
            }
            samples.append(new_sample)
        except:
            pass
    return samples


def finetune_decoder(cfg: omegaconf.DictConfig,
                     model: transformers.AutoModelForCausalLM,
                     tokenizer: transformers.AutoTokenizer,
                     dataset: torch.utils.data.Dataset):
    def tokenize_function(examples):
        return tokenizer(examples['source_code'], return_attention_mask=False)

    logger.info("Generating fine-tuning samples.")
    with multiprocessing.Pool(cfg.run.preprocessing_num_workers) as pool:
        if cfg.run.task == 'api-usage-completion':
            results = list(tqdm(pool.imap(get_api_usage_completion_samples, iter(dataset)), total=len(dataset)))
        elif cfg.run.task == 'api-call-completion':
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

    def create_labels(example):
        """Create labels by hand to ignore all `ids` that are not part of the API usage in the loss."""
        context_len = len(tokenizer(example['context']).input_ids)
        ground_truth_len = len(tokenizer(example['ground_truth']).input_ids)

        labels_context = [-100] * context_len
        labels_ground_truth = example['input_ids'][context_len:context_len + ground_truth_len]

        labels = labels_context + labels_ground_truth
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

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=cfg.run.train_batch_size
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        shuffle=False,
        collate_fn=default_data_collator,
        batch_size=cfg.run.valid_batch_size
    )

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': cfg.run.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.run.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=cfg.run.max_train_steps)

    logger.info("***** Running fine-tuning *****")
    logger.info("  Task = %s", cfg.run.task)
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Max num Epochs = %d", cfg.run.num_train_epochs)
    logger.info("  Total train batch size  = %d", cfg.run.train_batch_size)
    logger.info("  Total optimization steps = %d", cfg.run.max_train_steps)

    model.zero_grad()
    model.train()

    progress_bar = tqdm(range(cfg.run.max_train_steps), desc='Training')
    completed_steps = 0
    best_eval_loss = 10e6
    tr_num, tr_loss = 0, 0
    for epoch in range(1, cfg.run.num_train_epochs + 1):
        logger.info(f"Starting epoch {epoch}")
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(cfg.device)
            labels = batch['labels'].to(cfg.device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            completed_steps += 1
            progress_bar.update(1)
            tr_loss += loss.item()
            tr_num += 1
            if completed_steps % cfg.run.logging_steps == 0:
                avg_loss = round(tr_loss / tr_num, 5)
                if cfg.use_wandb:
                    wandb.log(
                        {
                            'train/loss': avg_loss,
                            'epoch': epoch,
                            'step': completed_steps
                        }, step=completed_steps)
                logger.info(f"epoch {epoch} | step {completed_steps} | loss {avg_loss}")

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.run.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if completed_steps % cfg.run.eval_steps == 0:
                logger.info(f"Running validation after {completed_steps} steps.")
                model.eval()
                eval_loss = 0
                for eval_step, batch in enumerate(tqdm(valid_dataloader, desc='Validation')):
                    with torch.no_grad():
                        input_ids = batch['input_ids'].to(cfg.device)
                        labels = batch['labels'].to(cfg.device)
                        outputs = model(input_ids=input_ids, labels=labels)
                        loss = outputs.loss
                        eval_loss += loss.item()
                model.train()
                eval_loss /= eval_step
                eval_ppl = round(np.exp(eval_loss), 5)
                if cfg.use_wandb:
                    wandb.log(
                        {
                            'eval/loss': round(eval_loss, 5),
                            'eval/perplexity': eval_ppl,
                            'epoch': epoch,
                            'step': completed_steps
                        }, step=completed_steps)
                logger.info(f"epoch {epoch} | eval loss {round(eval_loss, 5)} | eval ppl {eval_ppl}")

                checkpoint_prefix = f'step_{completed_steps}'
                if not os.path.exists(checkpoint_prefix):
                    os.makedirs(checkpoint_prefix)
                model.save_pretrained(checkpoint_prefix)
                logger.info(f"New model checkpoint saved {os.path.join(os.getcwd(), checkpoint_prefix)}")

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    logger.info("  " + "*" * 20)
                    logger.info("  Best eval loss:%s", round(best_eval_loss, 4))
                    logger.info("  " + "*" * 20)
                    if cfg.use_wandb:
                        wandb.run.summary['best_eval_loss'] = best_eval_loss
                        wandb.run.summary['best_checkpoint_step'] = completed_steps
