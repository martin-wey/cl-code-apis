import os

import hydra
import omegaconf
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer
)

from tasks import code_generation

logger = get_logger(__name__)


@hydra.main(config_path='configuration', config_name='defaults', version_base='1.1')
def main(cfg: omegaconf.DictConfig):
    if cfg.run.seed is not None:
        set_seed(cfg.run.seed)
    # hydra changes the current working dir, so we have to keep in memory the base path of the project
    cfg.run.base_path = hydra.utils.get_original_cwd()

    accelerator = Accelerator()
    # loading model from checkpoint or HF hub
    logger.info(f"Loading pre-trained model from checkpoint ({cfg.model.model_name_or_path}).")
    model_path = os.path.join(cfg.run.base_path, cfg.model.model_name_or_path)
    tokenizer_path = os.path.join(cfg.run.base_path, cfg.model.tokenizer_name_or_path)
    if cfg.run.task == 'clm':
        model = AutoModelForCausalLM.from_pretrained(model_path)
    elif cfg.run.task == 'mlm':
        model = AutoModelForMaskedLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    # for inference only
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    if cfg.run.dataset_dir is None:
        raise ValueError(
            "You must provide a value for `dataset_dir` for inference."
        )
    dataset_dir = os.path.join(cfg.run.base_path, cfg.run.dataset_dir)
    raw_dataset = load_dataset(dataset_dir)
    raw_dataset['train'] = load_dataset(dataset_dir, split='train[:5%]')
    raw_dataset['train'] = raw_dataset['train'].remove_columns(['method_name', 'docstring', 'hash', 'lines'])

    if cfg.run.task == 'clm':
        if 'perplexity' in cfg.run.evaluate:
            logger.info("Evaluating loss and perplexity on input dataset.")
            loss, perplexity = code_generation.evaluate_perplexity(cfg, accelerator, model, tokenizer, raw_dataset)
            logger.info(f'Loss: {round(loss, 4)} | perplexity: {round(perplexity, 4)}')
        elif 'token_generation' in cfg.run.evaluate:
            logger.info("Evaluating model on token generation.")
            code_generation.evaluate_token_generation(cfg, accelerator, model, tokenizer, raw_dataset)
        elif 'api_generation' in cfg.run.evaluate:
            logger.info("Evaluating model on api generation.")
            code_generation.evaluate_api_generation(cfg, accelerator, model, tokenizer, raw_dataset)
    elif cfg.run.task == 'mlm':
        pass


if __name__ == '__main__':
    main()
