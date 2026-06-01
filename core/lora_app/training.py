import gc
import json
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Literal, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from dynamic_lora.core.lora_app.config import (
    FULL_EPOCHS,
    FULL_LEARNING_RATE,
    LORA_EPOCHS,
    LORA_LEARNING_RATE,
    TrainingConfig,
)
from dynamic_lora.core.lora_app.data import collate_batch, create_dataset
from dynamic_lora.core.lora_app.modeling import add_lora_adapter, load_base_model

try:
    from peft import PeftModel
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "Missing dependency: peft. Install it with `pip install peft` before running this script."
    ) from exc

TrainMode = Literal["lora", "full"]


def _build_dataloader(tokenizer: AutoTokenizer, config: TrainingConfig) -> DataLoader:
    dataset = create_dataset(tokenizer, config)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )


def _run_training_loop(model: nn.Module, dataloader: DataLoader, config: TrainingConfig) -> list[dict[str, float | int]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    device = next(model.parameters()).device
    epoch_losses: list[dict[str, float | int]] = []
    accumulation_steps = config.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(dataloader) / accumulation_steps)
    warmup_steps = max(0, int(config.warmup_epochs * steps_per_epoch))
    optimizer_step_count = 0

    def lr_lambda(current_step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    for epoch in range(config.epochs):
        total_loss = 0.0
        optimizer.zero_grad()
        for step_index, batch in enumerate(dataloader, start=1):

            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            labels = batch.labels.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            (loss / accumulation_steps).backward()
            if step_index % accumulation_steps == 0 or step_index == len(dataloader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step_count += 1

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append({"epoch": epoch + 1, "loss": avg_loss})
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"epoch={epoch + 1} loss={avg_loss:.4f} "
            f"lr={current_lr:.6g} optimizer_steps={optimizer_step_count}"
        )

    return epoch_losses


def _save_epoch_losses(output_dir: str, epoch_losses: list[dict[str, float | int]]) -> None:
    path = os.path.join(output_dir, "epoch_losses.json")
    with open(path, "w", encoding="utf-8") as loss_file:
        json.dump(epoch_losses, loss_file, indent=2)
    print(f"[save] epoch losses -> {path}")


def _save_lora_ab(model: PeftModel, output_dir: str) -> None:
    lora_ab = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }
    path = os.path.join(output_dir, "lora_ab.pt")
    torch.save(lora_ab, path)
    print(f"[save] lora A/B -> {path}")


def _custom_initialize_lora_a(model: PeftModel, std: float) -> None:
    initialized = 0
    for name, parameter in model.named_parameters():
        if ".lora_A." in name:
            torch.nn.init.normal_(parameter, mean=0.0, std=std)
            initialized += 1
        elif ".lora_B." in name:
            torch.nn.init.zeros_(parameter)
    print(f"[init] custom LoRA init applied: A ~ N(0, {std}), B = 0 across {initialized} A matrices")


def _save_experiment_metadata(output_dir: str, metadata: dict[str, str | float | int]) -> None:
    path = Path(output_dir) / "experiment_metadata.json"
    with open(path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, ensure_ascii=False)
    print(f"[save] experiment metadata -> {path}")


def _release_training_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_model(
    config: TrainingConfig,
    mode: TrainMode = "lora",
    output_mode: str | None = None,
    lora_a_init_std: float | None = None,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_target_modules: Sequence[str] | None = None,
    epochs: int | None = None,
    learning_rate: float | None = None,
) -> tuple[AutoTokenizer, AutoModelForCausalLM | PeftModel]:
    default_epochs = FULL_EPOCHS if mode == "full" else LORA_EPOCHS
    default_learning_rate = FULL_LEARNING_RATE if mode == "full" else LORA_LEARNING_RATE
    run_config = replace(
        config,
        epochs=epochs if epochs is not None else default_epochs,
        learning_rate=learning_rate if learning_rate is not None else default_learning_rate,
    )
    print(
        f"[start] mode={mode} dataset={run_config.dataset_id} "
        f"subset={run_config.dataset_subset} split={run_config.dataset_split}"
    )
    token = os.environ.get("HF_TOKEN")
    print(f"[load] base model -> {run_config.model_id}")
    tokenizer, base_model = load_base_model(run_config, token)
    resolved_output_mode = output_mode or mode
    output_dir = run_config.run_output_dir(resolved_output_mode)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[path] output_dir -> {output_dir}")

    dataloader = _build_dataloader(tokenizer, run_config)
    effective_batch_size = run_config.batch_size * run_config.gradient_accumulation_steps
    print(
        f"[train] epochs={run_config.epochs} batch_size={run_config.batch_size} "
        f"grad_accumulation={run_config.gradient_accumulation_steps} "
        f"effective_batch_size={effective_batch_size} "
        f"learning_rate={run_config.learning_rate} "
        f"weight_decay={run_config.weight_decay} warmup_epochs={run_config.warmup_epochs}"
    )

    if mode == "lora":
        print("[mode] attaching LoRA adapter")
        model = add_lora_adapter(
            base_model,
            rank=lora_rank,
            alpha=lora_alpha,
            target_modules=lora_target_modules,
        )
        print(f"[mode] LoRA hyperparameters rank={lora_rank} alpha={lora_alpha}")
        if lora_target_modules is not None:
            print(f"[mode] LoRA target modules={list(lora_target_modules)}")
        if lora_a_init_std is not None:
            _custom_initialize_lora_a(model, lora_a_init_std)
            _save_experiment_metadata(
                output_dir,
                {
                    "mode": mode,
                    "output_mode": resolved_output_mode,
                    "lora_rank": lora_rank,
                    "lora_alpha": lora_alpha,
                    "lora_target_modules": list(lora_target_modules) if lora_target_modules is not None else None,
                    "lora_a_init_distribution": "normal",
                    "lora_a_init_mean": 0.0,
                    "lora_a_init_std": lora_a_init_std,
                },
            )
        elif resolved_output_mode != mode or lora_rank != 8 or lora_alpha != 16 or lora_target_modules is not None:
            _save_experiment_metadata(
                output_dir,
                {
                    "mode": mode,
                    "output_mode": resolved_output_mode,
                    "lora_rank": lora_rank,
                    "lora_alpha": lora_alpha,
                    "lora_target_modules": list(lora_target_modules) if lora_target_modules is not None else None,
                },
            )
        model.train()
        epoch_losses = _run_training_loop(model, dataloader, run_config)
        _save_epoch_losses(output_dir, epoch_losses)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"[save] lora adapter/tokenizer -> {output_dir}")
        _save_lora_ab(model, output_dir)
        print(f"[done] mode={mode}")
        return tokenizer, model

    print("[mode] full finetuning (no LoRA)")
    model = base_model
    model.train()
    epoch_losses = _run_training_loop(model, dataloader, run_config)
    _save_epoch_losses(output_dir, epoch_losses)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[save] full model/tokenizer -> {output_dir}")
    print(f"[done] mode={mode}")
    return tokenizer, model


def train_both_modes(
    config: TrainingConfig,
) -> dict[str, str]:
    print("[pipeline] running both modes: full -> lora")
    full_output_dir = config.run_output_dir("full")
    lora_output_dir = config.run_output_dir("lora")

    full_result = train_model(config, mode="full")
    del full_result
    _release_training_memory()

    lora_result = train_model(config, mode="lora")
    del lora_result
    _release_training_memory()

    print("[pipeline] completed both modes")
    return {
        "full": full_output_dir,
        "lora": lora_output_dir,
    }
