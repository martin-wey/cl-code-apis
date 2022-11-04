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
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer
)

from tasks import pretraining

logger = get_logger(__name__)
MAX_GPU_BATCH_SIZE = 12


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

    # if per_device_train_batch_size is too large, we accumulate gradients
    if cfg.run.per_device_train_batch_size > MAX_GPU_BATCH_SIZE:
        cfg.run.gradient_accumulation_steps = cfg.run.per_device_train_batch_size // MAX_GPU_BATCH_SIZE
        cfg.run.per_device_train_batch_size = MAX_GPU_BATCH_SIZE

    accelerator = Accelerator(gradient_accumulation_steps=cfg.run.gradient_accumulation_steps,
                              mixed_precision=cfg.run.mixed_precision,
                              **accelerator_log_kwargs)
    logger.info(accelerator.state, main_process_only=False)

    if cfg.use_wandb:
        accelerator.init_trackers(project_name='cl-code')

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if cfg.model.model_name_or_path is None:
        # instantiating a new model from scratch
        if cfg.model.model_config_path is None:
            raise ValueError(
                "You need to provide a path for a .json model config when instantiating"
                "a new model from scratch."
            )
        config_path = os.path.join(cfg.run.base_path, cfg.model.model_config_path)
        config = AutoConfig.from_pretrained(config_path)

        if cfg.model.tokenizer_name_or_path is None:
            raise ValueError(
                "You need to train a tokenizer and provide its name or path"
                "before initiating a new model pretraining."
            )
        tokenizer_path = os.path.join(cfg.run.base_path, cfg.model.tokenizer_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        if cfg.run.task == 'clm':
            model = AutoModelForCausalLM.from_config(config)
        elif cfg.run.task == 'mlm':
            model = AutoModelForMaskedLM.from_config(config)
        else:
            raise ValueError(
                "This scripts only support CLM and MLM pretraining."
            )
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Training a new model from scratch (# parameters: {model_params})")
    else:
        # loading model from checkpoint or HF hub
        logger.info(f"Loading pre-trained model from checkpoint ({cfg.model.model_name_or_path}).")
        if cfg.run.task == 'clm':
            model = AutoModelForCausalLM.from_pretrained(cfg.model.model_name_or_path)
        elif cfg.run.task == 'mlm':
            model = AutoModelForMaskedLM.from_pretrained(cfg.model.model_name_or_path)

    model.resize_token_embeddings(len(tokenizer))

    if cfg.run.dataset_name is not None:
        # use streaming for large datasets
        raw_datasets = load_dataset(cfg.run.dataset_name, streaming=True)
    elif cfg.run.dataset_dir is not None:
        dataset_dir = os.path.join(cfg.run.base_path, cfg.run.dataset_dir)
        raw_datasets = load_dataset(dataset_dir)
        if 'validation' not in raw_datasets.keys():
            raw_datasets['validation'] = load_dataset(dataset_dir, split=f'train[:5%]')
            raw_datasets['train'] = load_dataset(dataset_dir, split=f'train[5%:]')
        for ds_key in ['train', 'validation']:
            raw_datasets[ds_key] = raw_datasets[ds_key].remove_columns(
                [cname for cname in raw_datasets[ds_key].column_names if cname != 'source_code'])
    else:
        raise ValueError(
            "You must either specify a dataset name using `run.dataset_name` config key "
            "or a dataset directory using `run.dataset_dir` config key."
        )

    def tokenize_function(examples):
        if cfg.run.task == 'mlm':
            return tokenizer(examples['source_code'], return_special_tokens_mask=True)
        return tokenizer(examples['source_code'])

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=cfg.run.preprocessing_num_workers,
            load_from_cache_file=True,
            remove_columns=['source_code'],
            desc="Running tokenizer on dataset",
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
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=cfg.run.preprocessing_num_workers,
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    train_dataset = lm_datasets['train']
    valid_dataset = lm_datasets['validation']

    pretraining.train(cfg, accelerator, model, tokenizer, train_dataset, valid_dataset)


if __name__ == '__main__':
    main()
