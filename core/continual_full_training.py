import math

import torch
from torch import nn
from torch.utils.data import DataLoader

from dynamic_lora.core.lora_app.config import TrainingConfig


def train_full_one_task(
    model: nn.Module,
    dataloader: DataLoader,
    config: TrainingConfig,
    task_name: str,
    log_every_steps: int,
) -> list[dict[str, float | int]]:
    for parameter in model.parameters():
        parameter.requires_grad_(True)

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
    print(
        f"[train:start] mode=full task={task_name} epochs={config.epochs} "
        f"batches_per_epoch={len(dataloader)} accumulation_steps={accumulation_steps} "
        f"optimizer_steps_per_epoch={steps_per_epoch} warmup_steps={warmup_steps}",
        flush=True,
    )

    for epoch in range(config.epochs):
        total_loss = 0.0
        optimizer.zero_grad()
        print(
            f"[train:epoch:start] mode=full task={task_name} epoch={epoch + 1}/{config.epochs}",
            flush=True,
        )
        for step_index, batch in enumerate(dataloader, start=1):
            outputs = model(
                input_ids=batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
                labels=batch.labels.to(device),
            )
            loss = outputs.loss
            (loss / accumulation_steps).backward()

            if step_index % accumulation_steps == 0 or step_index == len(dataloader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step_count += 1

            total_loss += float(loss.detach().cpu())
            should_log_step = log_every_steps > 0 and (
                step_index % log_every_steps == 0 or step_index == len(dataloader)
            )
            if should_log_step:
                print(
                    f"[train:step] mode=full task={task_name} epoch={epoch + 1}/{config.epochs} "
                    f"batch={step_index}/{len(dataloader)} avg_loss={total_loss / step_index:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.6g} optimizer_steps={optimizer_step_count}",
                    flush=True,
                )

        row = {
            "epoch": epoch + 1,
            "loss": total_loss / len(dataloader),
            "optimizer_steps": optimizer_step_count,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        epoch_losses.append(row)
        print(
            f"[train:epoch:done] mode=full task={task_name} epoch={row['epoch']}/{config.epochs} "
            f"loss={row['loss']:.4f} lr={row['learning_rate']:.6g} "
            f"optimizer_steps={optimizer_step_count}",
            flush=True,
        )

    print(f"[train:done] mode=full task={task_name}", flush=True)
    return epoch_losses
