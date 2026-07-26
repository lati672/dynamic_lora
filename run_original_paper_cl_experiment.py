#!/usr/bin/env python3
"""Train full-FT and/or additive stacked-LoRA on the original paper task sequence."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

REPO_PARENT = Path(__file__).resolve().parent.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from dynamic_lora.original_paper_experiment.data import TASK_SPECS, load_and_sample_tasks
from dynamic_lora.original_paper_experiment.modeling import ContinualClassifier


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", "--model-name", default="roberta-base")
    parser.add_argument("--output_dir", "--output-dir", type=Path, default=Path("outputs/original_paper_tasks"))
    parser.add_argument("--train_samples_per_task", "--train-samples-per-task", type=int, default=1000)
    parser.add_argument("--eval_samples_per_task", "--eval-samples-per-task", type=int, default=500)
    parser.add_argument("--task_sequence", "--task-sequence", nargs="+", default=list(TASK_SPECS))
    parser.add_argument("--methods", nargs="+", choices=("full", "stacked_lora"), default=["full", "stacked_lora"])
    parser.add_argument("--adapter_eval_mode", "--adapter-eval-mode", choices=("all", "task_specific"), default="all")
    parser.add_argument("--lora_rank", "--lora-rank", type=int, default=8)
    parser.add_argument("--lora_alpha", "--lora-alpha", type=float, default=32)
    parser.add_argument("--lora_dropout", "--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", "--target-modules", nargs="+",
                        default=["attention.self.query", "attention.self.value", "intermediate.dense", "output.dense"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=8)
    parser.add_argument("--learning_rate", "--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", "--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", "--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max_length", "--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=0)
    args = parser.parse_args()
    if min(args.train_samples_per_task, args.eval_samples_per_task, args.epochs, args.batch_size) <= 0:
        parser.error("sample counts, epochs, and batch size must be positive")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, tokenizer, args, shuffle: bool, seed: int) -> DataLoader:
    def collate(rows):
        tokens = tokenizer([row["text"] for row in rows], padding=True, truncation=True,
                           max_length=args.max_length, return_tensors="pt")
        tokens["labels"] = torch.tensor([row["label"] for row in rows], dtype=torch.long)
        return tokens

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, args.batch_size, shuffle=shuffle, collate_fn=collate,
                      num_workers=args.num_workers, generator=generator)


def train_stage(model, task, dataset, tokenizer, args, device, seed) -> float:
    loader = make_loader(dataset, tokenizer, args, True, seed)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * args.warmup_ratio), steps)
    model.train()
    total, seen = 0.0, 0
    for epoch in range(args.epochs):
        for batch in loader:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(task, **inputs), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * labels.numel()
            seen += labels.numel()
        print(f"[train] task={task} epoch={epoch + 1}/{args.epochs} loss={total / seen:.5f}", flush=True)
    return total / seen


@torch.no_grad()
def evaluate(model, task, dataset, tokenizer, args, device) -> tuple[float, float]:
    model.eval()
    correct = count = 0
    loss_sum = 0.0
    for batch in make_loader(dataset, tokenizer, args, False, args.seed):
        labels = batch.pop("labels").to(device)
        inputs = {key: value.to(device) for key, value in batch.items()}
        logits = model(task, **inputs)
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(-1) == labels).sum().item()
        count += labels.numel()
    return correct / count, loss_sum / count


def write_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "stage", "eval_task", "accuracy", "loss"])
        writer.writeheader()
        writer.writerows(rows)


def run_method(method: str, data, tokenizer, args, device) -> None:
    task_labels = {task: TASK_SPECS[task].num_labels for task in args.task_sequence}
    model = ContinualClassifier(args.model_name, task_labels)
    if method == "stacked_lora":
        model.install_lora(args.target_modules, args.lora_rank, args.lora_alpha, args.lora_dropout)
    model.to(device)
    adapters: list[str] = []
    result_rows: list[dict] = []
    method_dir = args.output_dir / ("full_finetune" if method == "full" else "stacked_lora")

    for stage_index, task in enumerate(args.task_sequence):
        adapter = task if method == "stacked_lora" else None
        if adapter:
            model.add_adapter(adapter)
            adapters.append(adapter)
            model.set_active_adapters(adapters)
        model.set_trainable(task, adapter)
        print(f"[stage:start] method={method} stage={task} active_adapters={model.active_adapters}", flush=True)
        train_stage(model, task, data[task][0], tokenizer, args, device, args.seed + stage_index)

        metadata = {
            "method": method, "stage": task, "stage_index": stage_index, "model_name": args.model_name,
            "task_labels": task_labels, "task_sequence": args.task_sequence, "adapters": adapters,
            "target_modules": args.target_modules, "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha, "adapter_eval_mode": args.adapter_eval_mode,
        }
        checkpoint = method_dir / f"{'full' if method == 'full' else 'stacked_lora'}_after_{task}"
        model.save_checkpoint(checkpoint, metadata)
        tokenizer.save_pretrained(checkpoint)

        for eval_task in args.task_sequence[: stage_index + 1]:
            if method == "stacked_lora":
                active = adapters if args.adapter_eval_mode == "all" else [eval_task]
                model.set_active_adapters(active)
            accuracy, loss = evaluate(model, eval_task, data[eval_task][1], tokenizer, args, device)
            result_rows.append({"method": method, "stage": task, "eval_task": eval_task,
                                "accuracy": accuracy, "loss": loss})
            print(f"[eval] method={method} stage={task} task={eval_task} accuracy={accuracy:.4f}", flush=True)
        if method == "stacked_lora":
            model.set_active_adapters(adapters)
        write_results(method_dir / "results.csv", result_rows)


def main() -> None:
    args = arguments()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args) | {"output_dir": str(args.output_dir)}, indent=2) + "\n")
    data = load_and_sample_tasks(args.task_sequence, args.train_samples_per_task, args.eval_samples_per_task,
                                 args.seed, args.output_dir / "sampled_data")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] device={device} tasks={args.task_sequence} methods={args.methods}", flush=True)
    for method in args.methods:
        seed_everything(args.seed)
        run_method(method, data, tokenizer, args, device)


if __name__ == "__main__":
    main()
