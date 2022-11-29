"""
Pre-training a GPT-like model for causal language modeling or
             a RoBERTa-like model for masked language modeling.
"""
import math

import accelerate
import datasets
import omegaconf
import torch
import transformers
from accelerate.logging import get_logger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

transformers.utils.logging.set_verbosity_error()
logger = get_logger(__name__)


def evaluate(cfg: omegaconf.DictConfig,
             accelerator: accelerate.Accelerator,
             model: transformers.AutoModelForCausalLM,
             valid_dataloader: torch.utils.data.DataLoader,
             valid_dataset: datasets.Dataset) -> (float, float):
    model.eval()
    losses = []
    for step, batch in enumerate(
            tqdm(valid_dataloader, desc='Validation', disable=not accelerator.is_local_main_process)):
        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'].to(accelerator.device),
                            labels=batch['labels'].to(accelerator.device))
        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(cfg.run.valid_batch_size)))
    losses = torch.cat(losses)
    losses = losses[:len(valid_dataset)]
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float('inf')
    return perplexity, eval_loss


def train(cfg: omegaconf.DictConfig,
          accelerator: accelerate.Accelerator,
          model: transformers.AutoModelForCausalLM,
          train_dataloader: torch.utils.data.DataLoader,
          valid_dataloader: torch.utils.data.DataLoader,
          valid_dataset: torch.utils.data.Dataset) -> None:
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': cfg.run.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.run.learning_rate)

    lr_scheduler = get_scheduler(
        name=cfg.run.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.run.num_warmup_steps,
        num_training_steps=cfg.run.max_train_steps,
    )
    accelerator.register_for_checkpointing(lr_scheduler)

    def get_lr():
        return optimizer.param_groups[0]['lr']

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, lr_scheduler
    )

    progress_bar = tqdm(range(cfg.run.max_train_steps), disable=not accelerator.is_local_main_process, desc='Training')
    completed_steps = 0
    starting_epoch = 0
    best_metric = None
    best_metric_checkpoint = None

    model.train()
    for epoch in range(cfg.run.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            track_loss = loss.detach().float().item()
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                completed_steps += 1
                progress_bar.update(1)
                progress_bar.set_description(f'step {completed_steps} - loss {track_loss}')

            if completed_steps % cfg.run.logging_steps == 0:
                accelerator.log({'train/loss': track_loss, 'lr': get_lr(), 'steps': completed_steps}, step=completed_steps)

            if completed_steps > 0 and completed_steps % cfg.run.save_checkpoint_steps == 0:
                logger.info("Running validation and saving model checkpoint.")
                perplexity, eval_loss = evaluate(cfg, accelerator, model, valid_dataloader, valid_dataset)
                accelerator.log({'eval/loss': eval_loss, 'eval/perplexity': perplexity}, step=completed_steps)

                # save accelerator and pre-trained model
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(f'step_{completed_steps}', save_function=accelerator.save)
                accelerator.save_state(output_dir=f'step_{completed_steps}')

                if best_metric is None or best_metric > perplexity:
                    best_metric = perplexity
                    best_metric_checkpoint = f'step_{completed_steps}'
                    accelerator.print(f"New best metric: {best_metric} at epoch {epoch} and step {completed_steps}")
                    accelerator.print(f"best_metric_checkpoint: {best_metric_checkpoint}")
                    accelerator.log({'best_perplexity': best_metric, 'best_checkpoint_step': best_metric_checkpoint})

                model.train()
            if completed_steps >= cfg.run.max_train_steps:
                break
