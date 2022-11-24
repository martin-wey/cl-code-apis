import os
from itertools import chain

import datasets
import hydra
import omegaconf
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    default_data_collator,
    DataCollatorForLanguageModeling
)

from tasks import pretraining

logger = get_logger(__name__)


@hydra.main(config_path='configuration', config_name='defaults', version_base='1.1')
def main(cfg: omegaconf.DictConfig):
    if cfg.run.seed is not None:
        set_seed(cfg.run.seed)
    # hydra changes the current working dir, so we have to keep in memory the base path of the project
    cfg.run.base_path = hydra.utils.get_original_cwd()

    accelerator_log_kwargs = {}
    if cfg.use_wandb:
        accelerator_log_kwargs['log_with'] = 'wandb'
        accelerator_log_kwargs['logging_dir'] = os.getcwd()

    accelerator = Accelerator(**accelerator_log_kwargs)
    if cfg.use_wandb:
        accelerator.init_trackers(project_name='cl-code')
    logger.info(accelerator.state, main_process_only=True)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if cfg.model.tokenizer_name_or_path is None:
        raise ValueError(
            "You need to train a tokenizer and provide its name or path"
            "before initiating/continuing model pretraining."
        )

    if cfg.model.model_name_or_path is None:
        # instantiating a new model from scratch
        if cfg.model.model_config_path is None:
            raise ValueError(
                "You need to provide a path for a .json model config when instantiating"
                "a new model from scratch."
            )
        config_path = os.path.join(cfg.run.base_path, cfg.model.model_config_path)
        config = AutoConfig.from_pretrained(config_path)

        tokenizer_path = os.path.join(cfg.run.base_path, cfg.model.tokenizer_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        if cfg.run.task == 'clm':
            model = AutoModelForCausalLM.from_config(config)
        elif cfg.run.task == 'mlm':
            model = AutoModelForMaskedLM.from_config(config)
    else:
        # loading model from checkpoint or HF hub
        logger.info(f"Loading pre-trained model from checkpoint ({cfg.model.model_name_or_path}).")
        if cfg.run.task == 'clm':
            model = AutoModelForCausalLM.from_pretrained(cfg.model.model_name_or_path)
        elif cfg.run.task == 'mlm':
            model = AutoModelForMaskedLM.from_pretrained(cfg.model.model_name_or_path)
        tokenizer_path = os.path.join(cfg.run.base_path, cfg.model.tokenizer_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Training a new model from scratch (# parameters: {model_params})")
    model.resize_token_embeddings(len(tokenizer))

    # limit memory footprint
    if cfg.run.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_dataset = load_dataset(cfg.run.train_dataset_name, split='train', use_auth_token=True)
    train_dataset = train_dataset.remove_columns(
        [cname for cname in train_dataset.column_names if cname != 'source_code'])
    valid_dataset = load_dataset(cfg.run.valid_dataset_name, split='train', use_auth_token=True)
    valid_dataset = valid_dataset.remove_columns(
        [cname for cname in valid_dataset.column_names if cname != 'source_code'])

    def tokenize_function(examples):
        if cfg.run.task == 'mlm':
            return tokenizer(examples['source_code'], return_special_tokens_mask=True)
        return tokenizer(examples['source_code'])

    with accelerator.main_process_first():
        train_dataset_tokenized = train_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=cfg.run.preprocessing_num_workers,
            load_from_cache_file=True,
            remove_columns=['source_code'],
            desc="Running tokenizer on train dataset.",
        )
        valid_dataset_tokenized = valid_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=cfg.run.preprocessing_num_workers,
            load_from_cache_file=True,
            remove_columns=['source_code'],
            desc="Running tokenizer on valid dataset.",
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

    with accelerator.main_process_first():
        train_dataset_blocks = train_dataset_tokenized.map(
            group_texts,
            batched=True,
            num_proc=cfg.run.preprocessing_num_workers,
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {block_size}",
        )
        valid_dataset_blocks = valid_dataset_tokenized.map(
            group_texts,
            batched=True,
            num_proc=cfg.run.preprocessing_num_workers,
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    if cfg.run.task == 'clm':
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=cfg.run.mlm_probability)

    train_dataloader = DataLoader(
        train_dataset_blocks,
        collate_fn=data_collator,
        batch_size=cfg.run.train_batch_size,
        shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset_blocks,
        collate_fn=data_collator,
        batch_size=cfg.run.valid_batch_size
    )

    pretraining.train(cfg, accelerator, model, train_dataloader, valid_dataloader, valid_dataset)


if __name__ == '__main__':
    main()
