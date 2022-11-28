import datasets
import numpy as np
import omegaconf
import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import default_data_collator


def evaluate_token_completion(cfg: omegaconf.DictConfig,
                              model: transformers.AutoModelForCausalLM,
                              tokenizer: transformers.AutoTokenizer,
                              dataset: datasets.Dataset):
    def tokenize_function(example):
        return tokenizer(
            example['source_code'],
            padding=False,
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_attention_mask=False
        )

    dataset_tokenized = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=cfg.run.preprocessing_num_workers,
        load_from_cache_file=True,
        remove_columns=['source_code', 'api_seq', 'domain', 'api'],
        desc="Running tokenizer on test dataset",
    )

    pass_1 = 0
    pass_5 = 0
    pass_10 = 0
    n_test = 1

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
    progress_bar = tqdm(range(len(dataset_tokenized)))
    for step, ex in enumerate(iter(dataset_tokenized)):
        input_ids = torch.tensor(ex['input_ids']).to(cfg.device)

        with torch.no_grad():
            outputs = model(input_ids)
            pred_ids = outputs.logits.argmax(-1)

        all_pred = []
        all_gt = []

        print(pred_ids.shape, pred_ids)
        print(input_ids.shape, input_ids)

        for pred, gt in zip(pred_ids, input_ids):
            pass

        """
        with torch.no_grad():
            outputs = model(input_ids)
            pred_top_k = outputs.logits.topk(10).indices

        all_pred = []
        all_gt = []

        pred = pred_top_k.cpu().numpy()
        gt = input_ids.cpu().tolist()

        now_gt = None
        now_pred = None
        for i, y in enumerate(gt):
            if i == 0:
                # predict bos token
                now_gt = [y]
                now_pred = [[0] for _ in range(10)]
                if y in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                         tokenizer.pad_token_id]:
                    current_decoded = []
                    for p in now_pred:
                        current_decoded.append(tokenizer.decode(p).strip().split()[0])
                    all_pred.append(current_decoded)
                    all_gt.append(decode_ids(now_gt).strip())
                    now_gt = []
                    now_pred = []
            else:
                # \u0120 == space = beginning/end of a token
                if tokenizer.convert_ids_to_tokens(y)[0] == '\u0120':
                    if len(now_gt) > 0:
                        id_loop = 0
                        try:
                            for current_pred in now_pred:
                                current_decoded = []
                                for p in current_pred:
                                    current_decoded.append(tokenizer.decode(p).strip().split()[0])
                                all_pred.append(current_decoded)
                                id_loop += 1
                        except IndexError:
                            for _ in now_pred[id_loop:]:
                                current_decoded = ['<SPACE>' for _ in range(10)]
                                all_pred.append(current_decoded)
                                id_loop += 1
                        all_gt.append(decode_ids(now_gt).strip())
                        now_gt = []
                        now_pred = []

                        print(all_pred)
                        print('----------------------------------')
                        print(all_gt)
                        print(f'First IF: {len(all_pred)} - {len(all_gt)}')

                if y in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                         tokenizer.pad_token_id]:
                    if len(now_gt) > 0:
                        id_loop = 0
                        try:
                            for current_pred in now_pred:
                                current_decoded = []
                                for p in current_pred:
                                    current_decoded.append(tokenizer.decode(p).strip().split()[0])
                                all_pred.append(current_decoded)
                                id_loop += 1
                        except IndexError:
                            for _ in now_pred[id_loop:]:
                                current_decoded = ['<SPACE>' for _ in range(10)]
                                all_pred.append(current_decoded)
                                id_loop += 1
                        all_gt.append(decode_ids(now_gt).strip())
                    now_gt = [y]
                    now_pred = [pred[i - 1]]
                    try:
                        for current_pred in now_pred:
                            current_decoded = []
                            for p in current_pred:
                                current_decoded.append(tokenizer.decode(p).strip().split()[0])
                            all_pred.append(current_decoded)
                    except IndexError:
                        current_decoded = ['<SPACE>' for _ in range(10)]
                        all_pred.append(current_decoded)
                    all_gt.append(decode_ids(now_gt).strip())
                    now_gt = []
                    now_pred = []
                    continue
                now_gt.append(y)
                now_pred.append(pred[i - 1])
        """

        print(len(all_gt), len(all_pred))
        assert len(all_pred) == len(all_gt)

        for x, y in zip(all_pred, all_gt):
            if y in x[:1]:
                pass_1 += 1
                pass_5 += 1
                pass_10 += 1
            elif y in x[1:5]:
                pass_5 += 1
                pass_10 += 1
            elif y in x[5:]:
                pass_10 += 1
            n_test += 1

        progress_bar.update(1)
        progress_bar.set_description(f'Number of test: {n_test}')
        break

    print(f'Pass@1: {round(pass_1 / n_test, 3)}')
    print(f'Pass@5: {round(pass_5 / n_test, 3)}')
    print(f'Pass@10: {round(pass_10 / n_test, 3)}')


"""

def tokenize_function(cfg, tokenizer, example):
    return tokenizer(
        example['source_code'],
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_attention_mask=False,
        return_special_tokens_mask=False
    )

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
    # to only create OOD samples
    # api_usages = list(filter(lambda e: e[0] in OOD_APIS, api_usages))
    for usage in api_usages:
        # API class initialization case (e.g., new InputStreamReader ...)
        if usage[1] == '<init>':
            usage_pred = usage[0]
        # API call case (e.g., Collections.sort ...)
        else:
            usage_pred = usage[1]
        end_idx = int(usage[2])
        new_sample = {
            'api_seq': '',
            'source_code': example['source_code'][:end_idx],
            'is_test_sample': True,
            'ground_truth': f'{usage[0]}.{usage[1]}'
        }
        samples.append(new_sample)
    return samples


def evaluate_api_completion(cfg: omegaconf.DictConfig,
                            accelerator: accelerate.Accelerator,
                            model: transformers.AutoModelForCausalLM,
                            tokenizer: transformers.AutoTokenizer,
                            dataset: datasets.Dataset):
    # @todo: externalize the test samples creation

    # add a column to determine whether a sample should be included in the test set
    #   we create the new test samples according to the API location in the `source_code` field.
    is_test_sample = [False] * len(dataset['train'])
    ground_truth = [None] * len(dataset['train'])
    dataset['train'] = dataset['train'].add_column('is_test_sample', is_test_sample)
    dataset['train'] = dataset['train'].add_column('ground_truth', ground_truth)
    eval_dataset = dataset['train'].select(list(range(200)))

    logger.info('Generating test samples.')
    for sample in tqdm(eval_dataset):
        for new_sample in get_test_samples(sample):
            eval_dataset = eval_dataset.add_item(new_sample)
    # remove non-test samples and too long samples for the model
    eval_dataset = eval_dataset.filter(lambda e: e['is_test_sample'])

    eval_dataset_tokenized = eval_dataset.map(
        lambda e: tokenize_function(cfg, tokenizer, e),
        batched=True,
        num_proc=cfg.run.preprocessing_num_workers,
        load_from_cache_file=True,
        remove_columns=['source_code'],
        desc="Running tokenizer on test samples",
    )
    eval_dataloader = DataLoader(eval_dataset_tokenized, collate_fn=default_data_collator,
                                 batch_size=cfg.run.per_device_eval_batch_size)
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    results = {}
    ground_truths = []
    predictions = []

    model.eval()
    for step, sample in enumerate(tqdm(eval_dataloader, desc='Validation')):
        with torch.no_grad():
            try:
                generated_tokens = model.generate(sample['input_ids'], max_new_tokens=1)

                # retrieve the last generated token
                input_ids_len = len(sample['input_ids'].squeeze())
                last_predicted_token = tokenizer.decode(generated_tokens.squeeze()[input_ids_len:], skip_special_tokens=True)

                ground_truth_splitted = eval_dataset_tokenized[step]['ground_truth'].split('.')
                if ground_truth_splitted[0] not in results:
                    results[ground_truth_splitted[0]] = {}
                # API class initialization case
                if ground_truth_splitted[1] == '<init>':
                    if '<init>' not in results[ground_truth_splitted[0]]:
                        results[ground_truth_splitted[0]]['<init>'] = [0, 0]
                    ground_truth = ground_truth_splitted[0]
                # API call case
                else:
                    if ground_truth_splitted[1] not in results[ground_truth_splitted[0]]:
                        results[ground_truth_splitted[0]][ground_truth_splitted[1]] = [0, 0]
                    ground_truth = ground_truth_splitted[1]

                if ground_truth == last_predicted_token:
                    # add correctly predicted test sample
                    results[ground_truth_splitted[0]][ground_truth_splitted[1]][0] += 1

                # add test sample
                results[ground_truth_splitted[0]][ground_truth_splitted[1]][1] += 1
                # store ground-truth and prediction for further analysis
                ground_truths.append(eval_dataset_tokenized[step]['ground_truth'])
                predictions.append(last_predicted_token)
            except RuntimeError:
                print(sample['input_ids'])

    api_inits_accuracy = 0
    api_inits_n_samples = 0
    api_calls_accuracy = 0
    api_calls_n_samples = 0

    for api, items in results.items():
        for pred, metrics in items.items():
            if pred == '<init>':
                api_inits_accuracy += (metrics[0] / metrics[1])
                api_inits_n_samples += metrics[1]
            else:
                api_calls_accuracy += (metrics[0] / metrics[1])
                api_calls_n_samples += metrics[1]

    logger.info(f'API init accuracy: {round(api_inits_accuracy / api_inits_n_samples, 4)} - # test samples: {api_inits_n_samples}')
    logger.info(f'API calls accuracy: {round(api_calls_accuracy / api_calls_n_samples, 4)} - # test samples: {api_calls_n_samples}')
"""
