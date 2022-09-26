import logging
import os
import random

import hydra
import numpy as np
import omegaconf
import torch
import wandb
from datasets import load_dataset, load_from_disk
from transformers import AutoModel, AutoTokenizer, T5EncoderModel

from data.utils import remove_comments_and_docstrings
from tasks import codesearch

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'codebert_codesearch': (AutoModel, AutoTokenizer),
    'codet5_codesearch': (T5EncoderModel, AutoTokenizer)
}


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

    base_dataset_dir = os.path.join(cfg.run.base_path, cfg.run.dataset_dir, cfg.run.dataset_lang)
    dataset_files = {
        'train': os.path.join(base_dataset_dir, 'train.jsonl'),
        'valid': os.path.join(base_dataset_dir, 'valid.jsonl'),
        'test': os.path.join(base_dataset_dir, 'test.jsonl'),
    }

    splits = []
    if cfg.run.do_train:
        splits += ['train', 'valid']
    if cfg.run.do_test:
        splits += ['test']

    dataset = {}
    for split in splits:
        if cfg.run.load_data_from_disk:
            logger.info(f'Loading preprocessed {split} set from disk.')
            dataset[split] = load_from_disk(os.path.join(base_dataset_dir, f'{split}_preprocessed'))
        else:
            logger.info(f'Preprocessing {split} set.')
            dataset[split] = load_dataset('json', data_files=dataset_files, split=split)
            dataset[split] = dataset[split].map(
                lambda e: {'original_string': preprocess_code(e['original_string'], cfg.run.dataset_lang)}, num_proc=8)
            dataset[split] = dataset[split].filter(lambda e: e['original_string'] is not None, num_proc=8)
            logger.info(f'Saving {split} dataset to disk.')
            dataset[split].save_to_disk(os.path.join(base_dataset_dir, f'{split}_preprocessed'))

    model_cls = MODEL_CLASSES[f'{cfg.model.model_name}_{cfg.run.task}']
    if cfg.model.checkpoint_dir is not None:
        logger.info('Loading model from local checkpoint.')
        model = model_cls[0].from_pretrained(os.path.join(cfg.run.base_path, cfg.model.checkpoint_dir))
    else:
        logger.info('Loading model from HuggingFace hub.')
        model = model_cls[0].from_pretrained(cfg.model.hf_model_name)
    tokenizer = model_cls[1].from_pretrained(cfg.model.hf_tokenizer_name)

    if cfg.parallel:
        model = torch.nn.DataParallel(model)
    model.to(cfg.device)

    if cfg.run.do_train:
        logger.info(f'***** Fine-tuning {cfg.model.model_name} model on {cfg.run.task} (lang: {cfg.run.dataset_lang}) *****')
        task_func = getattr(globals().get(cfg.run.task), 'train')
        task_func(cfg, model, tokenizer, dataset['train'], dataset['valid'])

    if cfg.run.do_test:
        logger.info(f'***** Testing {cfg.model.model_name} model on {cfg.run.task} (lang: {cfg.run.dataset_lang}) *****')
        task_func = getattr(globals().get(cfg.run.task), 'test')
        task_func(cfg, model, tokenizer, dataset['test'])


if __name__ == '__main__':
    main()
