"""Full-parameter causal-LM fine-tuning on a deterministic Dolma subset."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterator

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--dataset-id", default="allenai/dolma")
    p.add_argument("--dataset-config", default="v1_6-sample")
    p.add_argument("--num-documents", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle-buffer-size", type=int, default=10_000)
    p.add_argument("--max-length", type=int, default=1_024)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/dolma_full_20k"))
    p.add_argument("--save-steps", type=int, default=0)
    p.add_argument("--log-steps", type=int, default=25)
    p.add_argument("--token", default=os.getenv("HF_TOKEN"))
    a = p.parse_args()
    positive = ("num_documents", "shuffle_buffer_size", "max_length", "batch_size",
                "gradient_accumulation_steps", "epochs", "learning_rate", "max_grad_norm")
    if any(getattr(a, name) <= 0 for name in positive):
        p.error("document, buffer, length, batch, epoch, LR, and gradient values must be positive")
    if not 0 <= a.warmup_ratio < 1:
        p.error("--warmup-ratio must be in [0, 1)")
    if a.save_steps < 0 or a.log_steps < 0:
        p.error("--save-steps and --log-steps cannot be negative")
    return a


def resolve_dtype(name: str) -> torch.dtype:
    if name != "auto":
        return getattr(torch, name)
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def dolma_stream(a: argparse.Namespace):
    manifest = hf_hub_download(
        repo_id=a.dataset_id, filename=f"urls/{a.dataset_config}.txt",
        repo_type="dataset", token=a.token,
    )
    urls = [line.strip() for line in Path(manifest).read_text().splitlines() if line.strip()]
    if not urls:
        raise RuntimeError(f"Dolma URL manifest is empty: {manifest}")
    return load_dataset(
        "json", data_files={"train": urls}, split="train",
        streaming=True, token=a.token,
    ).shuffle(seed=a.seed, buffer_size=a.shuffle_buffer_size)


def batches(a: argparse.Namespace) -> Iterator[list[dict[str, Any]]]:
    rows = (row for row in dolma_stream(a) if str(row.get("text", "")).strip())
    seen = 0
    while seen < a.num_documents:
        batch = []
        try:
            for _ in range(min(a.batch_size, a.num_documents - seen)):
                batch.append(next(rows))
        except StopIteration as error:
            raise RuntimeError(f"Dolma ended after {seen + len(batch):,} usable documents") from error
        seen += len(batch)
        yield batch


def save_model(model, tokenizer, path: Path, metadata: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    (path / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    a = parse_args()
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    dtype = resolve_dtype(a.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(a.model_id, token=a.token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(a.model_id, token=a.token, dtype=dtype)
    model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    batches_per_epoch = math.ceil(a.num_documents / a.batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / a.gradient_accumulation_steps)
    total_updates = updates_per_epoch * a.epochs
    warmup_steps = int(total_updates * a.warmup_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    metadata: dict[str, Any] = {
        "model_id": a.model_id, "dataset_id": a.dataset_id,
        "dataset_config": a.dataset_config, "num_documents": a.num_documents,
        "seed": a.seed, "shuffle_buffer_size": a.shuffle_buffer_size,
        "max_length": a.max_length, "batch_size": a.batch_size,
        "gradient_accumulation_steps": a.gradient_accumulation_steps,
        "epochs": a.epochs, "learning_rate": a.learning_rate,
        "weight_decay": a.weight_decay, "warmup_ratio": a.warmup_ratio,
        "dtype": str(dtype).removeprefix("torch."), "training_mode": "full_finetuning",
        "total_optimizer_steps": total_updates, "epoch_losses": [],
    }
    a.output_dir.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    started = time.time()

    for epoch in range(1, a.epochs + 1):
        epoch_nll = 0.0
        epoch_tokens = 0
        for batch_step, batch in enumerate(batches(a), 1):
            encoded = tokenizer(
                [row["text"] for row in batch], padding=True, truncation=True,
                max_length=a.max_length, return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            labels = input_ids.masked_fill(attention_mask == 0, -100)
            predicted_tokens = int((labels[:, 1:] != -100).sum().item())
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            (loss / a.gradient_accumulation_steps).backward()
            epoch_nll += float(loss.detach()) * predicted_tokens
            epoch_tokens += predicted_tokens

            update = batch_step % a.gradient_accumulation_steps == 0 or batch_step == batches_per_epoch
            if update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if a.log_steps and global_step % a.log_steps == 0:
                    mean_loss = epoch_nll / epoch_tokens
                    print(
                        f"epoch={epoch}/{a.epochs} step={global_step}/{total_updates} "
                        f"documents={min(batch_step*a.batch_size, a.num_documents):,} "
                        f"loss={mean_loss:.6f} ppl={math.exp(mean_loss):.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.3g}",
                        flush=True,
                    )
                if a.save_steps and global_step % a.save_steps == 0:
                    metadata["completed_optimizer_steps"] = global_step
                    save_model(model, tokenizer, a.output_dir / f"checkpoint-{global_step}", metadata)

        mean_loss = epoch_nll / epoch_tokens
        metadata["epoch_losses"].append({
            "epoch": epoch, "mean_nll": mean_loss, "perplexity": math.exp(mean_loss),
            "predicted_tokens": epoch_tokens,
        })
        print(f"epoch={epoch} complete loss={mean_loss:.6f} ppl={math.exp(mean_loss):.4f}", flush=True)

    metadata["completed_optimizer_steps"] = global_step
    metadata["elapsed_seconds"] = time.time() - started
    model.config.use_cache = True
    save_model(model, tokenizer, a.output_dir / "final", metadata)
    (a.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved full model and tokenizer to {a.output_dir / 'final'}")


if __name__ == "__main__":
    main()
