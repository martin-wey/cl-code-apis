import difflib

import accelerate
import datasets
import omegaconf
import torch
import transformers
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from transformers import default_data_collator
from tqdm import tqdm

OOD_APIS = ('InputStreamReader', 'Collections')
logger = get_logger(__name__)


def tokenize_function(cfg, tokenizer, examples):
    return_special_tokens = False
    if cfg.run.task == 'mlm':
        return_special_tokens = True
    return tokenizer(
        examples['source_code'],
        padding=False,
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_attention_mask=False,
        return_special_tokens_mask=return_special_tokens
    )


def evaluate_perplexity(cfg: omegaconf.DictConfig,
                        accelerator: accelerate.Accelerator,
                        model: transformers.AutoModelForCausalLM,
                        tokenizer: transformers.AutoTokenizer,
                        dataset: datasets.Dataset):
    with accelerator.main_process_first():
        tokenized_dataset = dataset.map(
            lambda e: tokenize_function(cfg, tokenizer, e),
            batched=True,
            num_proc=cfg.run.preprocessing_num_workers,
            load_from_cache_file=True,
            remove_columns=['source_code'],
            desc="Running tokenizer on dataset",
        )
    eval_dataloader = DataLoader(tokenized_dataset['train'], collate_fn=default_data_collator,
                                 batch_size=cfg.run.per_device_eval_batch_size)
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    model.eval()
    losses = []
    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader, desc='Validation')):
        with torch.no_grad():
            outputs = model(**batch)

        """The following code allows to compute the loss and cross-entropy for specific API classes and calls.

            labels = torch.zeros(batch['input_ids'].size(), dtype=torch.long).to(accelerator.device)
            unmasked_labels_indices = find_patterns_in_batch_ids(batch['input_ids'], api_calls_ood_tokenized)
            for i in range(batch['input_ids'].shape[0]):
                unmasked_idx = torch.as_tensor(unmasked_labels_indices[i]).to(accelerator.device)
                src = batch['input_ids'][i, :]
                x = torch.zeros(src.size(), dtype=torch.int64).to(accelerator.device)
                x.scatter_(0, unmasked_idx, src)
                labels[i, :] = x
        """

        loss = outputs.loss.repeat(cfg.run.per_device_eval_batch_size)
        losses.append(accelerator.gather(loss))
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()


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


def evaluate_token_generation(cfg: omegaconf.DictConfig,
                              accelerator: accelerate.Accelerator,
                              model: transformers.AutoModelForCausalLM,
                              tokenizer: transformers.AutoTokenizer,
                              dataset: datasets.Dataset):
    eval_dataset_tokenized = dataset['train'].map(
        lambda e: tokenize_function(cfg, tokenizer, e),
        batched=True,
        num_proc=cfg.run.preprocessing_num_workers,
        load_from_cache_file=True,
        remove_columns=['source_code'],
        desc="Running tokenizer on test dataset",
    )
    eval_dataloader = DataLoader(eval_dataset_tokenized, collate_fn=default_data_collator,
                                 batch_size=cfg.run.per_device_eval_batch_size)
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    model.eval()

    acc = 0.0
    ratio = 0.0
    n_test = 0

    total_pred = []
    total_gt = []
    for step, sample in enumerate(tqdm(eval_dataloader, desc='Validation')):
        with torch.no_grad():
            outputs = model(**sample)
            pred_ids = outputs.logits.argmax(-1)
        all_pred = []
        all_gt = []
        for pred, gt in zip(pred_ids, sample['input_ids']):
            pred = pred.cpu().tolist()
            gt = gt.cpu().tolist()

            now_gt = None
            now_pred = None
            for i, y in enumerate(gt):
                if i == 0:
                    # predict bos token
                    current_gt = [y]
                    current_pred = [0]
                    now_gt = [y]
                    now_pred = [0]
                    if y in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                             tokenizer.pad_token_id]:
                        all_pred.append(tokenizer.decode(current_pred).strip().split()[0])
                        all_gt.append(tokenizer.decode(current_gt).strip())
                        now_gt.clear()
                        now_pred.clear()
                else:
                    # \u0120 == space = beginning/end of a token
                    if tokenizer.convert_ids_to_tokens(y)[0] == '\u0120':
                        if len(now_gt) > 0:
                            try:
                                all_pred.append(tokenizer.decode(now_pred).strip().split()[0])
                            except IndexError:
                                all_pred.append('<SPACE>')
                            all_gt.append(tokenizer.decode(now_gt).strip())
                            now_gt.clear()
                            now_pred.clear()
                    if y in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                             tokenizer.pad_token_id]:
                        if len(now_gt) > 0:
                            try:
                                all_pred.append(tokenizer.decode(now_pred).strip().split()[0])
                            except IndexError:
                                all_pred.append('<SPACE>')
                            all_gt.append(tokenizer.decode(now_gt).strip())
                        now_gt = [y]
                        now_pred = [pred[i - 1]]
                        try:
                            all_pred.append(tokenizer.decode(now_pred).strip().split()[0])
                        except IndexError:
                            all_pred.append('<SPACE>')
                        all_gt.append(tokenizer.decode(now_gt).strip())
                        now_gt.clear()
                        now_pred.clear()
                        continue
                now_gt.append(y)
                now_pred.append(pred[i - 1])
        assert len(all_pred) == len(all_gt)

        total_pred.extend(all_pred)
        total_gt.extend(all_gt)

        for x, y in zip(all_pred, all_gt):
            if x == y:
                acc += 1
            ratio += difflib.SequenceMatcher(None, x, y).ratio() * 100
            n_test += 1

    if n_test != 0:
        acc = round(acc / n_test, 4)
        ratio = round(ratio / n_test, 2)
    print(acc, ratio, n_test)


def evaluate_api_generation(cfg: omegaconf.DictConfig,
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
    eval_dataset = dataset['train'].select(list(range(25)))

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

    model.eval()
    for step, sample in enumerate(tqdm(eval_dataloader, desc='Validation')):
        with torch.no_grad():
            greedy_output = model.generate(sample['input_ids'], max_new_tokens=1)
            print(f'Ground-truth: {eval_dataset_tokenized[step]["ground_truth"]}')
            print(greedy_output)
            print(tokenizer.decode(sample['input_ids'].squeeze(), skip_special_tokens=True))
            print('-' * 100)
            print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
            print('=' * 100)

            if step == 10: break
