"""
Pre-training a GPT-like model for causal language modeling or
             a BERT-like model for masked language modeling.
"""
import os

import accelerate
import omegaconf
import torch
import transformers
from accelerate.logging import get_logger
from huggingface_hub import Repository
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

logger = get_logger(__name__)


def evaluate(cfg: omegaconf.DictConfig,
             accelerator: accelerate.Accelerator,
             model: transformers.AutoModelForCausalLM,
             valid_dataloader: torch.utils.data.DataLoader) -> (float, float):
    model.eval()
    losses = []
    for step, batch in enumerate(valid_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss.repeat(cfg.run.valid_batch_size)
        losses.append(accelerator.gather(loss))
    losses = torch.cat(losses)
    loss = losses[:valid_dataloader.dataset.current_size].mean()
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float('inf')
    return loss.item(), perplexity.item()


def get_grouped_params(model, cfg, no_decay=["bias", "ln_1.weight", "ln_2.weight", "ln_f.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": cfg.run.weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


def train(cfg: omegaconf.DictConfig,
          accelerator: accelerate.Accelerator,
          model: transformers.AutoModelForCausalLM,
          train_dataloader: torch.utils.data.DataLoader,
          valid_dataloader: torch.utils.data.DataLoader) -> None:
    if accelerator.is_main_process and cfg.run.push_to_hub:
        repo = Repository(os.getcwd(), clone_from=cfg.run.repo_name)

    optimizer = AdamW(get_grouped_params(model, cfg), lr=cfg.run.learning_rate)
    lr_scheduler = get_scheduler(
        name=cfg.run.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.run.num_warmup_steps * cfg.run.gradient_accumulation_steps,
        num_training_steps=cfg.run.max_train_steps * cfg.run.gradient_accumulation_steps,
    )
    accelerator.register_for_checkpointing(lr_scheduler)

    def get_lr():
        return optimizer.param_groups[0]['lr']

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, lr_scheduler
    )

    model.train()
    completed_steps = 0
    loss_tracking = 0
    progress_bar = tqdm(range(cfg.run.max_train_steps), disable=not accelerator.is_local_main_process)

    for step, batch in enumerate(train_dataloader, start=1):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            loss_tracking += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            completed_steps += 1

        if completed_steps % cfg.run.logging_steps == 0:
            if accelerator.is_main_process:
                accelerator.log({'train/loss': loss, 'lr': get_lr(), 'steps': completed_steps}, step=completed_steps)

        if completed_steps % cfg.run.save_checkpoint_steps == 0:
            logger.info("Running validation and saving model checkpoint.")
            eval_loss, perplexity = evaluate(cfg, accelerator, model, valid_dataloader)
            if accelerator.is_main_process:
                accelerator.log({'eval/loss': eval_loss, 'eval/perplexity': perplexity}, step=completed_steps)

            # save accelerator and pre-trained model
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(f'step_{completed_steps}', save_function=accelerator.save)
            accelerator.save_state(output_dir=f'step_{completed_steps}')

            if accelerator.is_main_process and cfg.run.push_to_hub:
                repo.push_to_hub(commit_message=f"step {step}")

            model.train()
        if completed_steps >= cfg.run.max_train_steps:
            break

    logger.info("Evaluating and saving model after training")
    loss, perplexity = evaluate(cfg, accelerator, model, valid_dataloader)
    accelerator.log({'eval/loss': loss, 'eval/perplexity': perplexity})
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(os.getcwd(), save_function=accelerator.save)
    accelerator.save_state(f'step_{step}')
    if accelerator.is_main_process and cfg.run.push_to_hub:
        repo.push_to_hub(commit_message="final model")
