"""Training helpers for sequential language-model fine-tuning.

This module contains the per-task training loop and the optional LoRA wrapping
used by the continual-learning experiment. The training loop expects batches in
the format returned by ``SFTCollator``: ``input_ids``, ``labels``, and
``attention_mask`` tensors.
"""

import gc
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from tqdm.auto import tqdm

from constants import LORA_ALPHA, LORA_DROPOUT, LORA_RANK, LORA_TARGET_MODULES
from utils import is_config_lora


def build_lora_model(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> torch.nn.Module:
    """Wrap a causal language model with a LoRA adapter.

    Args:
        model: Base model to adapt.
        config: Experiment configuration with a ``peft`` section. Missing LoRA
            hyperparameters fall back to project defaults from ``constants``.

    Returns:
        Model returned by ``peft.get_peft_model``.
    """
    lora_cfg = config["peft"]

    peft_config = LoraConfig(
        r=lora_cfg.get("r", LORA_RANK),
        lora_alpha=lora_cfg.get("lora_alpha", LORA_ALPHA),
        lora_dropout=lora_cfg.get("lora_dropout", LORA_DROPOUT),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg.get("target_modules", LORA_TARGET_MODULES),
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


def train_one_task(
    model: torch.nn.Module,
    train_data_loader: torch.utils.data.DataLoader,
    config: dict[str, Any],
    device: torch.device | str,
) -> None:
    """Fine-tune a model on one task's SFT batches.

    The loop supports gradient accumulation, optional gradient checkpointing,
    optional batch limiting for smoke tests, and gradient clipping. It disables
    ``model.config.use_cache`` during training to avoid incompatibilities with
    checkpointing.

    Args:
        model: Model whose ``forward`` call returns an object with a ``loss``
            tensor.
        train_data_loader: Iterable dataloader yielding tensor dictionaries.
        config: Experiment configuration containing a ``training`` section.
        device: Device name or object where batch tensors should be moved.
    """
    num_epochs = config["training"]["num_epochs"]
    learning_rate = float(config["training"].get("learning_rate", 5e-5))
    weight_decay = float(config["training"].get("weight_decay", 0.0))
    grad_accum_steps = config["training"].get("gradient_accumulation_steps", 1)
    max_grad_norm = config["training"].get("max_grad_norm", 1.0)
    max_train_batches = config["training"].get("max_train_batches")

    model.train()
    model.config.use_cache = False

    if config["training"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

        if is_config_lora(config):
            model.enable_input_require_grads()

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    total_batches = len(train_data_loader)
    if max_train_batches is not None:
        total_batches = min(total_batches, max_train_batches)

    total_steps = num_epochs * total_batches
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        optimizer_step_count = 0

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_data_loader):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            # Scale loss for gradient accumulation.
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step_count += 1

            epoch_loss += loss.item()
            batch_count += 1
            progress_bar.update(1)

        # Handle leftover gradients if number of batches is not divisible by grad_accum_steps.
        if batch_count % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step_count += 1

        avg_loss = epoch_loss / max(batch_count, 1)

        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"loss: {avg_loss:.4f} - "
            f"optimizer steps: {optimizer_step_count}"
        )

    # Drop optimizer state after the task.
    del optimizer

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
