import logging
import multiprocessing
import os

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


def finetune_decoder(cfg: omegaconf.DictConfig,
                     model: transformers.AutoModelForCausalLM,
                     tokenizer: transformers.AutoTokenizer,
                     dataset: torch.utils.data.Dataset):
    def tokenize_function(examples):
        return tokenizer(examples['source_code'], padding='max_length', truncation=False)

    logger.info("Generating fine-tuning samples.")
    with multiprocessing.Pool(cfg.run.preprocessing_num_workers) as pool:
        results = list(tqdm(pool.imap(get_api_usage_completion_samples, iter(dataset)), total=len(dataset)))
    results = [item for sublist in results for item in sublist]
    dataset = Dataset.from_pandas(pd.DataFrame(results))

    # filter too long samples
    dataset = dataset.filter(lambda e: len(tokenizer(e['source_code']).input_ids) <= tokenizer.model_max_length)

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
        padding_len = len(example['input_ids']) - context_len - ground_truth_len

        labels_context = [-100] * context_len
        labels_ground_truth = example['input_ids'][context_len:context_len + ground_truth_len]
        labels_padding = [-100] * padding_len

        labels = labels_context + labels_ground_truth + labels_padding
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

    optimizer = AdamW(model.parameters(), lr=cfg.run.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_dataloader) * cfg.run.num_train_epochs)

    logger.info("***** Running fine-tuning *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", cfg.run.num_train_epochs)
    logger.info("  Total train batch size  = %d", cfg.run.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader) * cfg.run.num_train_epochs)

    model.zero_grad()
    model.train()

    best_eval_loss = 10e6
    tr_num, tr_loss = 0, 0
    for idx in range(cfg.run.num_train_epochs):
        logger.info(f"Starting epoch {idx}")
        for step, batch in enumerate(tqdm(train_dataloader, desc='Training')):
            input_ids = batch['input_ids'].to(cfg.device)
            attention_mask = batch['attention_mask'].to(cfg.device)
            labels = batch['labels'].to(cfg.device)

            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss

            tr_loss += loss.item()
            tr_num += 1
            if (step + 1) % cfg.run.logging_steps == 0:
                avg_loss = round(tr_loss / tr_num, 5)
                if cfg.use_wandb:
                    wandb.log({'train/loss': avg_loss, 'epoch': idx, 'step': step + 1}, step=step + 1)
                logger.info(f"epoch {idx} | step {step + 1} | loss {avg_loss}")

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.run.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        model.eval()
        eval_loss = 0
        for eval_step, batch in enumerate(tqdm(valid_dataloader, desc='Validation')):
            with torch.no_grad():
                input_ids = batch['input_ids'].to(cfg.device)
                attention_mask = batch['attention_mask'].to(cfg.device)
                labels = batch['labels'].to(cfg.device)

                outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                loss = outputs.loss
                eval_loss += loss.item()
        model.train()
        eval_loss /= eval_step
        eval_ppl = round(np.exp(eval_loss), 5)
        if cfg.use_wandb:
            wandb.log({'eval/loss': round(eval_loss, 5), 'eval/perplexity': eval_ppl, 'epoch': idx, 'step': step + 1},
                      step=step + 1)
        logger.info(f"epoch {idx} | eval loss {round(eval_loss, 5)} | eval ppl {eval_ppl}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            logger.info("  " + "*" * 20)
            logger.info("  Best eval loss:%s", round(best_eval_loss, 4))
            logger.info("  " + "*" * 20)

            checkpoint_prefix = 'checkpoint-best-loss'
            if not os.path.exists(checkpoint_prefix):
                os.makedirs(checkpoint_prefix)
            model.save_pretrained(checkpoint_prefix)
            logger.info(f"New best model checkpoint saved {os.path.join(os.getcwd(), checkpoint_prefix)}")
