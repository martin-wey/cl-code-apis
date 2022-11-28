import logging
import os

import hydra
import omegaconf
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    AutoConfig,
    set_seed
)

from tasks.code_completion import evaluate_token_completion
from tasks.perplexity import evaluate_perplexity

logger = logging.getLogger(__name__)


@hydra.main(config_path='configuration', config_name='defaults', version_base='1.1')
def main(cfg: omegaconf.DictConfig):
    if cfg.run.seed is not None:
        set_seed(cfg.run.seed)
    # hydra changes the current working dir, so we have to keep in memory the base path of the project
    cfg.run.base_path = hydra.utils.get_original_cwd()

    logger.info(f"Loading pre-trained model from checkpoint ({cfg.model.model_name_or_path}).")
    config_path = os.path.join(cfg.run.base_path, cfg.model.model_config_path)
    model_path = os.path.join(cfg.run.base_path, cfg.model.model_name_or_path)
    tokenizer_path = os.path.join(cfg.run.base_path, cfg.model.tokenizer_name_or_path)

    config = AutoConfig.from_pretrained(config_path)
    if cfg.run.task == 'clm':
        model = AutoModelForCausalLM.from_pretrained(model_path, config=config)
    elif cfg.run.task == 'mlm':
        model = AutoModelForMaskedLM.from_pretrained(model_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    model.to(cfg.device)
    # for inference only
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    logger.info(f"Loading test dataset: ({cfg.run.dataset_name}).")
    dataset = load_dataset(cfg.run.dataset_name, split='train[:5%]')
    dataset = dataset.remove_columns(['repo_name', 'method_path', 'method_name', 'docstring'])

    if 'perplexity' in cfg.run.evaluate:
        logger.info("Evaluating loss and perplexity on test dataset.")
        loss, perplexity = evaluate_perplexity(cfg, model, tokenizer, dataset)
        logger.info(f'Loss: {round(loss, 4)} | perplexity: {round(perplexity, 4)}')

    if cfg.run.task == 'clm':
        if 'token_completion' in cfg.run.evaluate:
            evaluate_token_completion(cfg, model, tokenizer, dataset)
        if 'api_completion' in cfg.run.evaluate:
            pass
    elif cfg.run.task == 'mlm':
        pass


if __name__ == '__main__':
    main()
