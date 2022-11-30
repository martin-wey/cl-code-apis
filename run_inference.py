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

from tasks.code_completion import evaluate_token_completion, evaluate_api_completion, evaluate_api_usage_completion
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

    logger.info(f"Loading test dataset: ({cfg.run.dataset_name}).")
    dataset_url = os.path.join(cfg.run.hf_user, cfg.run.dataset_name)
    dataset = load_dataset(dataset_url, split='train', use_auth_token=True)
    dataset = dataset.remove_columns(['repo_name', 'method_path', 'method_name', 'docstring'])

    if cfg.run.domain != 'all':
        logger.info(f"Filtering dataset to keep sample of domain: `{cfg.run.domain}`")
        dataset = dataset.filter(lambda e: e['domain'] == cfg.run.domain, num_proc=cfg.run.preprocessing_num_workers)

    # loss and perplexity using CLM or MLM model
    if cfg.run.evaluate == 'perplexity':
        logger.info("***** Evaluating loss and perplexity on input dataset ******")
        logger.info(f"  Num test samples: {len(dataset)}")
        loss, perplexity = evaluate_perplexity(cfg, model, tokenizer, dataset)
        logger.info(f"Loss: {round(loss, 4)} | perplexity: {round(perplexity, 4)}")
    # next-token prediction using CLM model
    elif cfg.run.evaluate == 'token_completion':
        logger.info("***** Evaluating token completion on input dataset *****")
        logger.info(f"  Num test samples: {len(dataset)}")
        n_test, correct = evaluate_token_completion(cfg, model, tokenizer, dataset)
        logger.info(f"Accuracy: {round(correct / n_test, 3)} (num tests: {n_test})")
    # next-API prediction using CLM model
    elif cfg.run.evaluate == 'api_completion':
        logger.info("***** Evaluating API completion on input dataset *****")
        cfg.run.batch_size = 1
        n_test, pass_1, pass_5, pass_10 = evaluate_api_completion(cfg, model, tokenizer, dataset)
        logger.info(f"Number of test calls: {n_test}")
        logger.info(f"Pass@1: {round(pass_1 / n_test, 3)}")
        logger.info(f"Pass@5: {round(pass_5 / n_test, 3)}")
        logger.info(f"Pass@10: {round(pass_10 / n_test, 3)}")
    # API usage statement completion using CLM model
    elif cfg.run.evaluate == 'api_usage_completion':
        logger.info("***** Evaluating API usage completion on input dataset *****")
        cfg.run.batch_size = 1
        evaluate_api_usage_completion(cfg, model, tokenizer, dataset)
    # API prediction using MLM model
    elif cfg.run.evaluate == 'api_prediction':
        pass
    else:
        raise ValueError("Please select an evaluation task "
                         "(perplexity | token_completion | api_completion | api_prediction")


if __name__ == '__main__':
    main()
