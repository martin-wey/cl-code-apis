from itertools import chain
import logging

import datasets
import omegaconf
import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import default_data_collator

logger = logging.getLogger(__name__)


def evaluate_token_completion(cfg: omegaconf.DictConfig,
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
    logger.info(f'Accuracy on {total} samples: {round(correct / total, 3)}')


def evaluate_api_completion(cfg: omegaconf.DictConfig,
                            model: transformers.AutoModelForCausalLM,
                            tokenizer: transformers.AutoTokenizer,
                            dataset: datasets.Dataset):
    def tokenize_function(tokenizer, example):
        return tokenizer(example['source_code'], return_attention_mask=False)

    def get_test_samples(example):
        samples = []
        # split api usages
        api_usages = list(filter(lambda e: e != '', example['api_seq'].split('|')))
        api_usages = [u.split('.') for u in api_usages]
        # remove java wildcards (e.g., ArrayList<String> -> ArrayList)
        #   and get a list of api usage in the form of [api_class, api_call, index_in_source_code]
        api_usages = list(map(lambda e: [e[0].strip().split(' ')[0],
                                         e[1].strip().split(' ')[0],
                                         e[1].strip().split(' ')[1]], api_usages))

        for usage in api_usages:
            end_idx = int(usage[2])  # each API usage gives the index of the call site in the `source_code` field

            sample_input_ids = tokenizer(example['source_code'][:end_idx])['input_ids']
            if len(sample_input_ids) < tokenizer.model_max_length \
                    and usage[1] != '<init>':
                new_sample = {
                    'source_code': example['source_code'][:end_idx],
                    # 'domain': example['domain'],
                    # 'api': example['api'],
                    'api_seq': '',
                    'is_test_sample': True,
                    'ground_truth': f'{usage[0]}.{usage[1]}'
                }
                samples.append(new_sample)
        return samples

    dataset = dataset.remove_columns([name for name in dataset.column_names if name not in
                                      ['source_code', 'domain', 'api', 'api_seq']])

    # add a column to determine whether a sample should be included in the test set
    #   we create the new test samples according to the API location in the `source_code` field.
    is_test_sample = [False] * len(dataset)
    ground_truth = [None] * len(dataset)
    dataset = dataset.add_column('is_test_sample', is_test_sample)
    dataset = dataset.add_column('ground_truth', ground_truth)
    dataset = dataset.select(list(range(100)))

    logger.info('Generating test samples.')
    for sample in tqdm(dataset):
        for new_sample in get_test_samples(sample):
            dataset = dataset.add_item(new_sample)
    # remove non-test samples and too long samples for the model
    dataset = dataset.filter(lambda e: e['is_test_sample'])

    dataset_tokenized = dataset.map(
        lambda e: tokenize_function(tokenizer, e),
        batched=True,
        num_proc=cfg.run.preprocessing_num_workers,
        load_from_cache_file=True,
        remove_columns=['source_code'],
        desc="Running tokenizer on test samples",
    )
    dataloader = DataLoader(
        dataset_tokenized,
        collate_fn=default_data_collator,
        batch_size=cfg.run.batch_size
    )

    results = {}
    ground_truths = []
    predictions = []

    model.eval()
    for step, sample in enumerate(tqdm(dataloader, desc='Validation')):
        with torch.no_grad():
            try:
                generated_tokens = model.generate(
                    sample['input_ids'].to(cfg.device),
                    max_new_tokens=1,
                    do_sample=True,
                    num_return_sequences=10
                )

                input_ids_len = len(sample['input_ids'].squeeze())
                predictions_topk = tokenizer.batch_decode(generated_tokens[:, input_ids_len:], skip_special_tokens=True)

                ground_truth_splitted = dataset_tokenized[step]['ground_truth'].split('.')
                ground_truth_api = ground_truth_splitted[0]
                ground_truth_call = ground_truth_splitted[1]

                if ground_truth_api not in results:
                    results[ground_truth_api] = {}

                if ground_truth_call not in results[ground_truth_api]:
                    results[ground_truth_api][ground_truth_call] = {
                        'pass@1': 0,
                        'pass@5': 0,
                        'pass@10': 0,
                        'n_test': 0
                    }

                if ground_truth_call in predictions_topk[:1]:
                    # add correctly predicted test sample
                    results[ground_truth_api][ground_truth_call]['pass@1'] += 1
                    results[ground_truth_api][ground_truth_call]['pass@5'] += 1
                    results[ground_truth_api][ground_truth_call]['pass@10'] += 1
                elif ground_truth_call in predictions_topk[1:5]:
                    results[ground_truth_api][ground_truth_call]['pass@5'] += 1
                    results[ground_truth_api][ground_truth_call]['pass@10'] += 1
                elif ground_truth_call in predictions_topk[5:]:
                    results[ground_truth_api][ground_truth_call]['pass@10'] += 1

                # add test sample
                results[ground_truth_api][ground_truth_call]['n_test'] += 1
                # store ground-truth and prediction for further analysis
                ground_truths.append(dataset_tokenized[step]['ground_truth'])
                predictions.append(predictions_topk)
            except:
                print('error')

    pass_1 = 0
    pass_5 = 0
    pass_10 = 0
    all_tests = 0

    for api, items in results.items():
        for pred, metrics in items.items():
            pass_1 += (metrics['pass@1'] / metrics['n_test'])
            pass_5 += (metrics['pass@5'] / metrics['n_test'])
            pass_10 += (metrics['pass@10'] / metrics['n_test'])
            all_tests += metrics['n_test']

    logger.info(f'Number of test calls: {all_tests}')
    logger.info(f'Pass@1: {round(pass_1, 3)}')
    logger.info(f'Pass@5: {round(pass_5, 3)}')
    logger.info(f'Pass@10: {round(pass_10, 3)}')
