from itertools import chain

import datasets
import omegaconf
import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import default_data_collator


def evaluate_perplexity(cfg: omegaconf.DictConfig,
                        model: transformers.AutoModelForCausalLM,
                        tokenizer: transformers.AutoTokenizer,
                        dataset: datasets.Dataset):
    def tokenize_function(examples):
        if cfg.run.task == 'mlm':
            return tokenizer(examples['source_code'], return_special_tokens_mask=True)
        return tokenizer(examples['source_code'])

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=cfg.run.preprocessing_num_workers,
        load_from_cache_file=True,
        remove_columns=['source_code', 'api_seq', 'domain', 'api'],
        desc="Running tokenizer on train dataset.",
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
        if cfg.run.task == 'clm':
            result['labels'] = result['input_ids'].copy()
        return result

    dataset_blocks = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=cfg.run.preprocessing_num_workers,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )
    eval_dataloader = DataLoader(
        dataset_blocks,
        collate_fn=default_data_collator,
        batch_size=cfg.run.batch_size
    )

    model.eval()
    losses = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            losses.append(loss)
    loss = torch.cat(losses)
    try:
        loss = torch.mean(loss)
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()
