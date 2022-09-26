import logging
import os
from typing import Tuple

import datasets
import numpy as np
import omegaconf
import torch
import transformers
import wandb
from more_itertools import chunked
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)


def collator_fn(batch, tokenizer, max_code_length, max_query_length):
    code_batch = [b['original_string'] for b in batch]
    query_batch = [b['docstring'] for b in batch]

    code_tokenized = tokenizer(
        code_batch,
        padding='max_length',
        max_length=max_code_length,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt')
    all_code_ids = code_tokenized.input_ids
    all_code_masks = code_tokenized.attention_mask

    query_tokenized = tokenizer(
        query_batch,
        padding='max_length',
        max_length=max_query_length,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt')
    all_query_ids = query_tokenized.input_ids
    all_query_masks = query_tokenized.attention_mask

    return all_code_ids, all_code_masks, all_query_ids, all_query_masks


def compute_ranks(src_representations: np.ndarray,
                  tgt_representations: np.ndarray,
                  distance_metric: str = 'cosine') -> Tuple[np.array, np.array]:
    distances = cdist(src_representations, tgt_representations,
                      metric=distance_metric)
    # By construction the diagonal contains the correct elements
    correct_elements = np.expand_dims(np.diag(distances), axis=-1)
    return np.sum(distances <= correct_elements, axis=-1), distances


def train(cfg: omegaconf.DictConfig,
          model: transformers.RobertaModel,
          tokenizer: transformers.PreTrainedTokenizerFast,
          train_dataset: datasets.Dataset,
          valid_dataset: datasets.Dataset) -> None:
    logger.info('Preparing train and validation sets.')
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=cfg.run.train_batch_size,
                                  collate_fn=lambda batch: collator_fn(batch, tokenizer, cfg.run.max_code_length,
                                                                       cfg.run.max_query_length),
                                  shuffle=True,
                                  num_workers=8)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=cfg.run.valid_batch_size,
                                  collate_fn=lambda batch: collator_fn(batch, tokenizer, cfg.run.max_code_length,
                                                                       cfg.run.max_query_length),
                                  shuffle=False,
                                  num_workers=8)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.run.learning_rate, eps=1e-8)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()

    best_eval_loss = float('inf')
    patience_count = 0
    with logging_redirect_tqdm():
        for epoch in range(1, cfg.run.epochs + 1):
            training_loss = 0.0
            step_loss, step_num = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):
                code_representations = model(input_ids=batch[0], attention_mask=batch[1])[1]
                query_representations = model(input_ids=batch[2], attention_mask=batch[3])[1]

                # compute similarity matrix and softmax loss
                scores = torch.einsum('ab,cb->ac', query_representations, code_representations)
                loss = criterion(scores, torch.arange(batch[0].size(0)).cuda())

                # report loss
                training_loss += loss.item()
                step_loss += loss.item()
                step_num += 1
                if step % 200 == 0 and step > 0:
                    avg_loss = round(step_loss / step_num, 4)
                    logger.info(f'epoch #{epoch} | step {step} | loss {round(avg_loss, 3)}')
                    if cfg.use_wandb:
                        wandb.log({'train_loss': avg_loss, 'epoch': epoch, 'step': epoch * step})
                    step_loss, step_num = 0, 0

                # backpropagation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            logger.info('Running validation.')
            eval_loss, eval_mrr = evaluate(model, valid_dataloader, criterion)
            model.train()

            training_loss /= len(train_dataloader)
            logger.info(f'end of epoch #{epoch} | '
                        f'train loss {round(training_loss, 3)} | '
                        f'eval loss {round(eval_loss, 3)} | '
                        f'eval mrr {round(eval_mrr, 3)}')
            if cfg.use_wandb:
                wandb.log({'eval_loss': eval_loss, 'eval_mrr': eval_mrr, 'epoch': epoch})

            if eval_loss < best_eval_loss:
                logger.info('Saving new best model checkpoint.')
                model_to_save = model.module if hasattr(model, 'module') else model
                output_path = os.path.join(f'pytorch_model.bin')
                torch.save(model_to_save.state_dict(), output_path)
                patience_count = 0
                best_eval_loss = eval_loss
            else:
                patience_count += 1
            if patience_count == cfg.run.patience:
                logger.info(f'Stopping training (out of patience, patience={cfg.run.patience})')
                break


def evaluate(model: transformers.RobertaModel,
             valid_dataloader: DataLoader,
             criterion: torch.nn.CrossEntropyLoss) -> (float, float):
    model.eval()
    eval_loss = 0.0
    eval_mrr = 0.0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(valid_dataloader, desc='Iteration')):
            code_representations = model(input_ids=batch[0], attention_mask=batch[1])[1]
            query_representations = model(input_ids=batch[2], attention_mask=batch[3])[1]

            # compute similarity matrix and softmax loss
            scores = torch.einsum('ab,cb->ac', query_representations, code_representations)
            loss = criterion(scores, torch.arange(batch[0].size(0)).cuda())

            # diagonal part of the score matrix is the ground-truth (correspondance between code and query)
            correct_scores = torch.diagonal(scores)
            # compute how many queries have a bigger score than the ground-truth (= incorrect)
            compared_scores = scores >= torch.unsqueeze(correct_scores, dim=-1)

            eval_mrr += torch.sum((1 / torch.sum(compared_scores, dtype=torch.int, dim=0))).item() / batch[0].size(0)
            eval_loss += loss.item()
    return eval_loss / len(valid_dataloader), eval_mrr / len(valid_dataloader)


def test(cfg: omegaconf.DictConfig,
         model: transformers.RobertaModel,
         tokenizer: transformers.PreTrainedTokenizerFast,
         test_dataset: datasets.Dataset) -> None:
    def tokenize(batch, max_length):
        return tokenizer(
            batch,
            padding='max_length',
            max_length=max_length,
            truncation=True,
            return_tensors='np',
        ).input_ids

    logger.info('Preparing test set.')
    test_dataset = test_dataset.map(
        lambda batch: {'code_tokenized': tokenize(batch['original_string'], cfg.run.max_code_length),
                       'docstring_tokenized': tokenize(batch['docstring'], cfg.run.max_query_length)},
        batched=True,
        num_proc=4)
    test_dataset = test_dataset.remove_columns(
        [col for col in test_dataset.column_names if col not in ['code_tokenized', 'docstring_tokenized']])
    test_dataset = test_dataset.shuffle(seed=cfg.run.seed)

    # Because we have huge batch size at test (e.g., 1000), we transform tensors into np arrays
    #   otherwise it wouldn't fit in memory
    data = np.array(list(zip(test_dataset['code_tokenized'], test_dataset['docstring_tokenized'])), dtype=np.object)

    sum_mrr = 0.0
    num_batches = 0
    batched_data = chunked(data, cfg.run.test_batch_size)
    for batch_data in tqdm(batched_data, desc='Iteration', total=len(data) // cfg.run.test_batch_size):
        if len(batch_data) < cfg.run.test_batch_size:
            break  # the last batch is smaller than the others, exclude.
        num_batches += 1

        batch_data_list = list(zip(*batch_data))
        code_query_dataset = TensorDataset(torch.tensor(batch_data_list[0]), torch.tensor(batch_data_list[1]))
        minibatch_loader = DataLoader(code_query_dataset, batch_size=cfg.run.train_batch_size, pin_memory=True)

        code_representations = []
        query_representations = []
        for mini_batch in minibatch_loader:
            code_inputs = mini_batch[0].to(cfg.device)
            query_inputs = mini_batch[1].to(cfg.device)

            code_reps = model(code_inputs)[1]
            query_reps = model(query_inputs)[1]

            code_representations.append(code_reps.cpu().detach().numpy())
            query_representations.append(query_reps.cpu().detach().numpy())

        code_representations = np.concatenate(code_representations, axis=0)
        query_representations = np.concatenate(query_representations, axis=0)
        assert len(code_representations) == len(query_representations) == cfg.run.test_batch_size

        ranks, distances = compute_ranks(code_representations, query_representations)
        sum_mrr += np.mean(1.0 / ranks)

    mrr = sum_mrr / num_batches
    logger.info(f'Test MRR: {round(mrr, 4)}')
