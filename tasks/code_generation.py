import difflib

import accelerate
import omegaconf
import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.utils import get_apis_tokenized


def evaluate_perplexity(cfg: omegaconf.DictConfig,
                        accelerator: accelerate.Accelerator,
                        model: transformers.AutoModelForCausalLM,
                        eval_dataloader: DataLoader):
    model.eval()
    losses = []
    for step, batch in enumerate(
            tqdm(eval_dataloader, desc='Validation', disable=not accelerator.is_local_main_process)):
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


def evaluate_generation(cfg: omegaconf.DictConfig,
                        accelerator: accelerate.Accelerator,
                        model: transformers.AutoModelForCausalLM,
                        tokenizer: transformers.AutoTokenizer,
                        eval_dataloader: DataLoader):
    model.eval()

    api_cls_ood_tokenized, api_calls_ood_tokenized = get_apis_tokenized(tokenizer, ('HashMap.', 'Set.'))
    print(api_cls_ood_tokenized)
    print('---------------')
    print(api_calls_ood_tokenized)

    for step, sample in enumerate(
            tqdm(eval_dataloader, desc='Validation', disable=not accelerator.is_local_main_process)):
        pass

    """
    acc = 0.0
    ratio = 0.0
    n_test = 0

    total_pred = []
    total_gt = []
    for step, sample in enumerate(
            tqdm(eval_dataloader, desc='Validation', disable=not accelerator.is_local_main_process)):
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
    """
