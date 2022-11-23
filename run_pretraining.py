import os

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
from utils.dataset import ConstantLengthDataset

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

    accelerator = Accelerator(gradient_accumulation_steps=cfg.run.gradient_accumulation_steps,
                              mixed_precision=cfg.run.mixed_precision,
                              **accelerator_log_kwargs)
    logger.info(accelerator.state, main_process_only=True)

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

        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Training a new model from scratch (# parameters: {model_params})")
    else:
        # loading model from checkpoint or HF hub
        logger.info(f"Loading pre-trained model from checkpoint ({cfg.model.model_name_or_path}).")
        if cfg.run.task == 'clm':
            model = AutoModelForCausalLM.from_pretrained(cfg.model.model_name_or_path)
        elif cfg.run.task == 'mlm':
            model = AutoModelForMaskedLM.from_pretrained(cfg.model.model_name_or_path)
        tokenizer_path = os.path.join(cfg.run.base_path, cfg.model.tokenizer_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    model.resize_token_embeddings(len(tokenizer))
    if cfg.run.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_data = load_dataset(cfg.run.train_dataset_name, split='train', streaming=True, use_auth_token=True)
    valid_data = load_dataset(cfg.run.valid_dataset_name, split='train', streaming=True, use_auth_token=True)
    train_dataset = ConstantLengthDataset(tokenizer, train_data, infinite=True, seq_length=cfg.run.seq_length)
    valid_dataset = ConstantLengthDataset(tokenizer, valid_data, infinite=False, seq_length=cfg.run.seq_length)
    if cfg.run.task == 'clm':
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=cfg.run.mlm_probability)
    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, batch_size=cfg.run.train_batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, collate_fn=data_collator, batch_size=cfg.run.valid_batch_size)

    pretraining.train(cfg, accelerator, model, train_dataloader, valid_dataloader)


if __name__ == '__main__':
    main()
