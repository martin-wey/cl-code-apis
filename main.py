import logging
import os
import random

import hydra
import numpy as np
import omegaconf
import torch
import wandb
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from data.utils import remove_comments_and_docstrings
from tasks import codesearch

logger = logging.getLogger(__name__)


def preprocess_code(code, lang):
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        return None
    return code


@hydra.main(config_path='configuration', config_name='defaults', version_base='1.1')
def main(cfg: omegaconf.DictConfig):
    if cfg.run.seed > 0:
        random.seed(cfg.run.seed)
        np.random.seed(cfg.run.seed)
        torch.manual_seed(cfg.run.seed)
        torch.cuda.manual_seed_all(cfg.run.seed)

    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.parallel = torch.cuda.device_count() > 1
    # hydra changes the current working dir, so we have to keep in memory the base path of the project
    cfg.run.base_path = hydra.utils.get_original_cwd()

    if cfg.use_wandb:
        wandb_cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(**cfg.wandb.setup, config=wandb_cfg)

    dataset_files = {
        'train': os.path.join(cfg.run.base_path, cfg.run.dataset_dir, cfg.run.dataset_lang, f'train.jsonl'),
        'valid': os.path.join(cfg.run.base_path, cfg.run.dataset_dir, cfg.run.dataset_lang, f'valid.jsonl'),
        'test': os.path.join(cfg.run.base_path, cfg.run.dataset_dir, cfg.run.dataset_lang, f'test.jsonl'),
    }

    splits = []
    if cfg.run.do_train:
        splits += ['train', 'valid']
    if cfg.run.do_test:
        splits += ['test']

    dataset = {}
    for split in splits:
        dataset[split] = load_dataset('json', data_files=dataset_files, split=split)
        dataset[split] = dataset[split].map(
            lambda e: {'original_string': preprocess_code(e['original_string'], cfg.run.dataset_lang)}, num_proc=8)
        dataset[split] = dataset[split].filter(lambda e: e['original_string'] is not None, num_proc=8)

    # @todo: load model checkpoint if specified
    model = AutoModel.from_pretrained(cfg.model.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name_or_path)

    if cfg.parallel:
        model = torch.nn.DataParallel(model)
    model.to(cfg.device)

    if cfg.run.do_test:
        task_func = getattr(globals().get(cfg.run.task), 'test')
        task_func(cfg, model, tokenizer, dataset['test'])


if __name__ == '__main__':
    main()
