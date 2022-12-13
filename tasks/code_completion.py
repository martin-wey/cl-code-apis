import logging
from itertools import chain

import datasets
import omegaconf
import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import default_data_collator

logger = logging.getLogger(__name__)


def evaluate_code_completion(cfg: omegaconf.DictConfig,
                             model: transformers.AutoModelForCausalLM,
                             tokenizer: transformers.AutoTokenizer,
                             dataset: datasets.Dataset):
    def tokenize_function(examples):
        return tokenizer(examples['source_code'], return_attention_mask=False)

    dataset_tokenized = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=cfg.run.preprocessing_num_workers,
        load_from_cache_file=True,
        remove_columns=[name for name in dataset.column_names if name != 'input_ids'],
        desc="Running tokenizer on test dataset.",
    )

    block_size = tokenizer.model_max_length if tokenizer.model_max_length <= 1024 else 1024

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
        if cfg.model.model_type == 'decoder':
            result['labels'] = result['input_ids'].copy()
        return result

    dataset_blocks = dataset_tokenized.map(
        group_texts,
        batched=True,
        num_proc=cfg.run.preprocessing_num_workers,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )
    dataloader = DataLoader(
        dataset_blocks,
        collate_fn=default_data_collator,
        batch_size=cfg.run.batch_size
    )

    def decode_ids(idxs):
        codes = ''
        for idx in idxs:
            to_add = tokenizer.convert_ids_to_tokens(idx)
            if tokenizer.convert_ids_to_tokens(idx)[0] == '\u0120':
                if not codes.endswith(' '):
                    codes += ' ' + to_add[1:]
                else:
                    codes += to_add[1:]
            elif idx in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                         tokenizer.pad_token_id]:
                codes += ' ' + to_add + ' '
            else:
                codes += to_add
        return codes.strip(' ')

    model.eval()
    correct = 0.0
    total = 0
    progress_bar = tqdm(range(len(dataloader)))

    for step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(cfg.device)  # (bsz, 1024)
        with torch.no_grad():
            outputs = model(input_ids)
            pred_ids = outputs.logits.argmax(-1)  # (bsz, 1024)

        all_pred = []
        all_gt = []
        prev_pred = None
        for pred, gt in zip(pred_ids, input_ids):
            pred = pred.cpu().tolist()
            gt = gt.cpu().tolist()

            for i, y in enumerate(gt):
                if i == 0:
                    if y in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                             tokenizer.pad_token_id]:
                        now_gt = [y]
                        now_pred = [0] if prev_pred is None else [prev_pred]
                        all_pred.append(decode_ids(now_pred).strip().split()[0])
                        all_gt.append(decode_ids(now_gt).strip())
                        now_gt = []
                        now_pred = []
                    else:
                        now_gt = [y]
                        now_pred = [0] if prev_pred is None else [prev_pred]
                else:
                    if tokenizer.convert_ids_to_tokens(y)[0] == '\u0120':
                        if len(now_gt) > 0:
                            try:
                                all_pred.append(decode_ids(now_pred).strip().split()[0])
                            except IndexError:
                                all_pred.append('<SPACE>')
                            all_gt.append(decode_ids(now_gt).strip())
                            now_gt = []
                            now_pred = []
                    if y in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                             tokenizer.pad_token_id]:
                        if len(now_gt) > 0:
                            try:
                                all_pred.append(decode_ids(now_pred).strip().split()[0])
                            except IndexError:
                                all_pred.append('<SPACE>')
                            all_gt.append(decode_ids(now_gt).strip())
                        now_gt = [y]
                        now_pred = [pred[i - 1]]
                        try:
                            all_pred.append(decode_ids(now_pred).strip().split()[0])
                        except IndexError:
                            all_pred.append('<SPACE>')
                        all_gt.append(decode_ids(now_gt).strip())
                        now_gt = []
                        now_pred = []
                        continue
                    now_gt.append(y)
                    now_pred.append(pred[i - 1])

            assert len(all_pred) == len(all_gt)

            for x, y in zip(all_pred, all_gt):
                if y != '<|endoftext|>' and x != '<SPACE>':
                    total += 1
                    if x == y:
                        correct += 1
                        progress_bar.set_description(f'acc: {round(correct / total, 3)}')
        progress_bar.update(1)
    return total, correct
