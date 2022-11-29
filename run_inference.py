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

from tasks.code_completion import evaluate_token_completion, evaluate_api_completion
from tasks.perplexity import evaluate_perplexity

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

    config_path = os.path.join(cfg.run.base_path, cfg.model.model_config_path)
    model_path = os.path.join(cfg.run.base_path, cfg.model.model_name_or_path)
    tokenizer_path = os.path.join(cfg.run.base_path, cfg.model.tokenizer_name_or_path)

    config_cls, model_cls, tokenizer_cls = MODEL_CLS[cfg.model.model_type]
    try:
        logger.info(f"Attempting to load pre-trained model from local checkpoint ({cfg.model.model_name_or_path}).")
        config = config_cls.from_pretrained(config_path)
        model = model_cls.from_pretrained(model_path, config=config)
        tokenizer = tokenizer_cls.from_pretrained(tokenizer_path, use_fast=True)
    except:
        logger.info(f"Loading pre-trained model from hub ({cfg.model.model_name_or_path}).")
        model = model_cls.from_pretrained(cfg.model.model_name_or_path)
        tokenizer = tokenizer_cls.from_pretrained(cfg.model.model_name_or_path)
    model.to(cfg.device)
    # for inference only
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    logger.info(f"Loading test dataset: ({cfg.run.dataset_name}).")
    # for test purpose: select 5%, @todo: remove this
    dataset = load_dataset(cfg.run.dataset_name, split='train[:5%]', use_auth_token=True)
    dataset = dataset.remove_columns(['repo_name', 'method_path', 'method_name', 'docstring'])

    if cfg.run.evaluate == 'perplexity':
        logger.info("Evaluating loss and perplexity on test dataset.")
        loss, perplexity = evaluate_perplexity(cfg, model, tokenizer, dataset)
        logger.info(f'Loss: {round(loss, 4)} | perplexity: {round(perplexity, 4)}')
    elif cfg.run.evaluate == 'token_completion':
        evaluate_token_completion(cfg, model, tokenizer, dataset)
    elif cfg.run.evaluate == 'api_completion':
        cfg.run.batch_size = 1
        evaluate_api_completion(cfg, model, tokenizer, dataset)
    elif cfg.run.evaluate == 'api_prediction':
        pass
    else:
        raise ValueError("Please select an evaluation task "
                         "(perplexity | token_completion | api_completion | api_prediction")


if __name__ == '__main__':
    main()
