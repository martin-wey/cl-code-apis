import logging
import os

import hydra
import omegaconf
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    AutoConfig,
    set_seed
)

from tasks.finetuning import finetune_decoder

logger = logging.getLogger(__name__)

MODEL_CLS = {
    'encoder': (AutoConfig, AutoModelForMaskedLM, AutoTokenizer),
    'decoder': (AutoConfig, AutoModelForCausalLM, AutoTokenizer)
}


@hydra.main(config_path='configuration', config_name='defaults', version_base='1.1')
def main(cfg: omegaconf.DictConfig):
    if cfg.run.seed is not None:
        set_seed(cfg.run.seed)
    # hydra changes the current working dir, so we have to keep in memory the base path of the project
    cfg.run.base_path = hydra.utils.get_original_cwd()

    if cfg.use_wandb:
        wandb.init(project='cl-code')

    model_path = os.path.join(cfg.run.base_path, cfg.model.model_name_or_path)
    config_cls, model_cls, tokenizer_cls = MODEL_CLS[cfg.model.model_type]
    try:
        logger.info(f"Attempting to load pre-trained model from local checkpoint ({cfg.model.model_name_or_path}).")
        config = config_cls.from_pretrained(model_path)
        model = model_cls.from_pretrained(model_path, config=config)
        tokenizer = tokenizer_cls.from_pretrained(model_path, use_fast=True)
    except:
        logger.info(f"Loading pre-trained model from hub ({cfg.model.model_name_or_path}).")
        model = model_cls.from_pretrained(cfg.model.model_name_or_path)
        tokenizer = tokenizer_cls.from_pretrained(cfg.model.model_name_or_path)
    model.to(cfg.device)
    # for inference only
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    logger.info(f"Loading fine-tuning dataset: ({cfg.run.dataset_name}).")
    dataset_url = os.path.join(cfg.run.hf_user, cfg.run.dataset_name)
    dataset = load_dataset(dataset_url, split='train', use_auth_token=True)
    dataset = dataset.remove_columns(['repo_name', 'method_path', 'method_name', 'docstring'])

    # when using ID dataset, these columns are not present
    if 'domain' not in dataset.column_names and 'api' not in dataset.column_names:
        domain = ['NaN'] * len(dataset)
        api = ['NaN'] * len(dataset)
        dataset = dataset.add_column('domain', domain)
        dataset = dataset.add_column('api', api)

    # remove useless columns
    dataset = dataset.remove_columns([name for name in dataset.column_names if name not in
                                      ['source_code', 'domain', 'api', 'api_seq']])

    if cfg.model.model_type == 'decoder':
        finetune_decoder(cfg, model, tokenizer, dataset)
    elif cfg.model.model_type == 'encoder':
        pass


if __name__ == '__main__':
    main()
