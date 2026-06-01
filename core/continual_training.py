import math

import torch
from peft import PeftModel
from torch.utils.data import DataLoader

from dynamic_lora.core.adapters import (
    l2_penalty,
    orthogonal_penalty,
    set_active_adapters,
    set_only_adapter_trainable,
)
from dynamic_lora.core.lora_app.config import TrainingConfig


def train_one_task(
    model: PeftModel,
    dataloader: DataLoader,
    config: TrainingConfig,
    train_adapter_name: str,
    active_adapters: list[str],
    old_adapter_name: str | None,
    orthogonal_penalty_enabled: bool,
    orthogonal_penalty_weight: float,
    l2_penalty_weight: float,
) -> list[dict[str, float | int]]:
    set_active_adapters(model, active_adapters)
    set_only_adapter_trainable(model, train_adapter_name)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    device = next(model.parameters()).device
    accumulation_steps = config.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(dataloader) / accumulation_steps)
    warmup_steps = max(0, int(config.warmup_epochs * steps_per_epoch))
    optimizer_step_count = 0
    epoch_losses: list[dict[str, float | int]] = []

    def lr_lambda(current_step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    model.train()

    for epoch in range(config.epochs):
        total_loss = 0.0
        total_ce_loss = 0.0
        total_orthogonal_penalty = 0.0
        total_l2_penalty = 0.0
        optimizer.zero_grad()
        for step_index, batch in enumerate(dataloader, start=1):
            outputs = model(
                input_ids=batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
                labels=batch.labels.to(device),
            )
            ce_loss = outputs.loss
            orth_penalty = (
                orthogonal_penalty(model, old_adapter_name, train_adapter_name)
                if orthogonal_penalty_enabled
                else torch.zeros((), dtype=torch.float32, device=device)
            )
            l2_reg = (
                l2_penalty(model, train_adapter_name)
                if l2_penalty_weight > 0
                else torch.zeros((), dtype=torch.float32, device=device)
            )
            loss = ce_loss + orthogonal_penalty_weight * orth_penalty + l2_penalty_weight * l2_reg
            (loss / accumulation_steps).backward()

            if step_index % accumulation_steps == 0 or step_index == len(dataloader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step_count += 1

            total_loss += float(loss.detach().cpu())
            total_ce_loss += float(ce_loss.detach().cpu())
            total_orthogonal_penalty += float(orth_penalty.detach().cpu())
            total_l2_penalty += float(l2_reg.detach().cpu())

        row = {
            "epoch": epoch + 1,
            "loss": total_loss / len(dataloader),
            "ce_loss": total_ce_loss / len(dataloader),
            "orthogonal_penalty": total_orthogonal_penalty / len(dataloader),
            "l2_penalty": total_l2_penalty / len(dataloader),
            "optimizer_steps": optimizer_step_count,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        epoch_losses.append(row)
        print(
            f"adapter={train_adapter_name} epoch={row['epoch']} "
            f"loss={row['loss']:.4f} ce_loss={row['ce_loss']:.4f} "
            f"orthogonal_penalty={row['orthogonal_penalty']:.6f} "
            f"l2_penalty={row['l2_penalty']:.6f} "
            f"lr={row['learning_rate']:.6g} optimizer_steps={optimizer_step_count}"
        )
    return epoch_losses
