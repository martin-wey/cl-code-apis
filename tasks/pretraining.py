"""
Pre-training a GPT-like model for causal language modeling or
             a BERT-like model for masked language modeling.
"""
import math
import os

import accelerate
import datasets
import omegaconf
import torch
import transformers
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler, default_data_collator, DataCollatorForLanguageModeling

logger = get_logger(__name__)


def evaluate(cfg: omegaconf.DictConfig,
             accelerator: accelerate.Accelerator,
             model: transformers.AutoModelForCausalLM,
             eval_dataloader: DataLoader,
             eval_dataset: datasets.Dataset) -> (float, float):
    model.eval()
    losses = []
    for step, batch in enumerate(
            tqdm(eval_dataloader, desc='Validation', disable=not accelerator.is_local_main_process)):
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(cfg.run.per_device_eval_batch_size)))
    losses = torch.cat(losses)
    losses = losses[:len(eval_dataset)]
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float('inf')
    return perplexity, eval_loss


def train(cfg: omegaconf.DictConfig,
          accelerator: accelerate.Accelerator,
          model: transformers.AutoModelForCausalLM,
          tokenizer: transformers.AutoTokenizer,
          train_dataset: datasets.Dataset,
          valid_dataset: datasets.Dataset) -> None:
    if cfg.run.task == 'clm':
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=cfg.run.mlm_probability)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=cfg.run.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        valid_dataset, collate_fn=data_collator, batch_size=cfg.run.per_device_eval_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.run.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg.run.learning_rate)

    def get_lr():
        return optimizer.param_groups[0]['lr']

    # Scheduler and math around the number of training steps.
    lr_scheduler = get_scheduler(
        name=cfg.run.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.run.num_warmup_steps * cfg.run.gradient_accumulation_steps,
        num_training_steps=cfg.run.max_train_steps * cfg.run.gradient_accumulation_steps)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.run.gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    cfg.run.num_train_epochs = math.ceil(cfg.run.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = cfg.run.per_device_train_batch_size * accelerator.num_processes * cfg.run.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {cfg.run.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.run.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.run.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.run.max_train_steps}")

    progress_bar = tqdm(range(cfg.run.max_train_steps), disable=not accelerator.is_local_main_process, desc='Training')
    completed_steps = 0
    starting_epoch = 0
    best_metric = None
    best_checkpoint_step = None

    if cfg.run.resume_from_checkpoint:
        if cfg.run.resume_from_checkpoint is not None or cfg.run.resume_from_checkpoint != "":
            checkpoint_path = os.path.join(cfg.run.base_path, cfg.run.resume_from_checkpoint)
            accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
            accelerator.load_state(checkpoint_path)
            path = os.path.basename(checkpoint_path)
        # Extract the step of the checkpoint to continue from there
        training_difference = os.path.splitext(path)[0]
        resume_step = int(training_difference.replace('step_', ''))

    for epoch in range(starting_epoch, cfg.run.num_train_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if cfg.run.resume_from_checkpoint and step < resume_step:
                continue  # we need to skip steps until we reach the resumed step
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps % cfg.run.logging_steps == 0 and cfg.use_wandb:
                accelerator.log({'train/loss': loss, 'lr': get_lr()}, step=completed_steps)

            if completed_steps % cfg.run.save_checkpoint_steps == 0:
                logger.info("Running validation and saving model checkpoint.")
                perplexity, eval_loss = evaluate(cfg, accelerator, model, eval_dataloader, valid_dataset)
                accelerator.log({'eval/loss': eval_loss, 'eval/perplexity': perplexity}, step=completed_steps)

                # save accelerator and pr-etrained model
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(f'step_{completed_steps}/model', save_function=accelerator.save)
                accelerator.save_state(output_dir=f'step_{completed_steps}')

                # track the best metric and checkpoint step
                if best_metric is None or best_metric > perplexity:
                    best_metric = perplexity
                    best_checkpoint_step = completed_steps
                    accelerator.log({'best_perplexity': best_metric, 'best_checkpoint_step': best_checkpoint_step})

                model.train()
            if completed_steps >= cfg.run.max_train_steps:
                break

        logger.info(f"Evaluate model and saving after epoch #{epoch}.")
        perplexity, eval_loss = evaluate(cfg, accelerator, model, eval_dataloader, valid_dataset)

        accelerator.log(
            {
                'eval/loss': eval_loss,
                'eval/perplexity': perplexity,
                'step': completed_steps,
                'epoch': epoch
            }, step=completed_steps
        )

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(f'step_{completed_steps}/model', save_function=accelerator.save)
        accelerator.save_state(output_dir=f'step_{completed_steps}')

        # track the best metric and checkpoint step
        if best_metric is None or best_metric > perplexity:
            best_metric = perplexity
            best_checkpoint_step = completed_steps
            accelerator.log({'best_perplexity': best_metric, 'best_checkpoint_step': best_checkpoint_step})

    logger.info('Evaluating model after training completed.')
    perplexity, eval_loss = evaluate(cfg, accelerator, model, eval_dataloader, valid_dataset)
    accelerator.log({'eval/loss': eval_loss, 'eval/perplexity': perplexity}, step=completed_steps)
