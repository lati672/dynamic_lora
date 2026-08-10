#!/usr/bin/env python3
"""Train full-FT and/or additive stacked-LoRA for the intruder experiment."""

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

from dynamic_lora.intruder_experiment.data import TASK_SPECS, load_and_sample_tasks
from dynamic_lora.intruder_experiment.modeling import ContinualClassifier


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", "--model-name", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--output_dir", "--output-dir", type=Path, default=Path("outputs/intruder_experiment"))
    parser.add_argument("--train_samples_per_task", "--train-samples-per-task", type=int, default=8000)
    parser.add_argument("--eval_samples_per_task", "--eval-samples-per-task", type=int, default=1000)
    parser.add_argument("--task_sequence", "--task-sequence", nargs="+", default=list(TASK_SPECS))
    parser.add_argument(
        "--methods", nargs="+", choices=("full", "single_lora", "stacked_lora"),
        default=["full", "single_lora", "stacked_lora"],
    )
    parser.add_argument("--adapter_eval_mode", "--adapter-eval-mode", choices=("all", "task_specific", "learned_gates"), default="learned_gates")
    parser.add_argument("--rank", type=int, default=16, help="Shared rank for single and stacked LoRA.")
    parser.add_argument("--lora_alpha", "--lora-alpha", type=float, default=32)
    parser.add_argument("--lora_dropout", "--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--orthogonal_penalty_weight", "--orthogonal-penalty-weight", type=float, default=0.1,
        help="Weight of the normalized ||A_previous @ A_new.T||_1 penalty for stacked_lora.",
    )
    parser.add_argument(
        "--orthogonal_penalty_type", "--orthogonal-penalty-type",
        choices=("a_subspace", "effective_update"), default="effective_update",
    )
    parser.add_argument(
        "--orthogonal_schedule", "--orthogonal-schedule",
        choices=("immediate", "delayed", "linear"), default="linear",
        help="Activation schedule for stacked-LoRA orthogonal regularization.",
    )
    parser.add_argument(
        "--orthogonal_delay_epochs", "--orthogonal-delay-epochs", type=int, default=1,
        help="CE-only epochs before the delayed orthogonal schedule activates.",
    )
    parser.add_argument("--target_modules", "--target-modules", nargs="+",
                        default=["q_proj", "v_proj", "up_proj", "down_proj"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=8)
    parser.add_argument("--learning_rate", "--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", "--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", "--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max_length", "--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=0)
    parser.add_argument(
        "--save_checkpoints", "--save-checkpoints", action=argparse.BooleanOptionalAction, default=True,
        help="Save model/tokenizer checkpoints after each task stage.",
    )
    args = parser.parse_args()
    if min(args.train_samples_per_task, args.eval_samples_per_task, args.epochs, args.batch_size) <= 0:
        parser.error("sample counts, epochs, and batch size must be positive")
    if args.rank <= 0:
        parser.error("LoRA rank must be positive")
    if args.orthogonal_penalty_weight < 0:
        parser.error("orthogonal penalty weight must be non-negative")
    if args.orthogonal_delay_epochs < 0:
        parser.error("orthogonal delay epochs must be non-negative")
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


def scheduled_orthogonal_weight(args, epoch: int, batch_index: int, batches_per_epoch: int) -> float:
    """Return the current regularization weight for immediate, delayed, or linear schedules."""
    target = args.orthogonal_penalty_weight
    schedule = getattr(args, "orthogonal_schedule", "immediate")
    if schedule == "immediate":
        return target
    if schedule == "delayed":
        return 0.0 if epoch < getattr(args, "orthogonal_delay_epochs", 1) else target
    if schedule == "linear":
        total_steps = max(1, args.epochs * batches_per_epoch)
        current_step = epoch * batches_per_epoch + batch_index + 1
        return target * current_step / total_steps
    raise ValueError(f"Unknown orthogonal schedule: {schedule}")


def train_stage(model, task, dataset, tokenizer, args, device, seed, adapter=None, previous_a=None, previous_adapters=None) -> float:
    loader = make_loader(dataset, tokenizer, args, True, seed)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * args.warmup_ratio), steps)
    model.train()
    total, seen = 0.0, 0
    for epoch in range(args.epochs):
        for batch_index, batch in enumerate(loader):
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            ce_loss = F.cross_entropy(model(task, **inputs), labels)
            if adapter and args.orthogonal_penalty_type == "effective_update":
                orthogonal_penalty = model.orthogonal_penalty_effective(adapter, previous_adapters or [])
            elif adapter:
                orthogonal_penalty = model.orthogonal_penalty(adapter, previous_a, normalize=True)
            else:
                orthogonal_penalty = ce_loss.new_zeros(())
            orthogonal_weight = scheduled_orthogonal_weight(args, epoch, batch_index, len(loader)) if adapter else 0.0
            loss = ce_loss + orthogonal_weight * orthogonal_penalty
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
    method_lora_rank = getattr(args, "rank", getattr(args, "lora_rank", 16))
    if method in {"single_lora", "stacked_lora"}:
        model.install_lora(args.target_modules, method_lora_rank, args.lora_alpha, args.lora_dropout)
    model.to(device)
    adapters: list[str] = []
    result_rows: list[dict] = []
    method_dir = args.output_dir / ("full_finetune" if method == "full" else method)

    if method == "single_lora":
        model.add_adapter("shared")
        adapters.append("shared")

    for stage_index, task in enumerate(args.task_sequence):
        adapter = task if method == "stacked_lora" else "shared" if method == "single_lora" else None
        prior_adapters = list(adapters)
        previous_a = (
            model.snapshot_adapters_a(prior_adapters)
            if method == "stacked_lora" and prior_adapters and args.orthogonal_penalty_type == "a_subspace"
            else None
        )
        if method == "stacked_lora":
            model.add_adapter(adapter)
            adapters.append(adapter)
            if args.adapter_eval_mode == "learned_gates":
                model.configure_task_gate(task, adapters)
                model.set_task_gate(task)
            else:
                model.set_active_adapters(adapters)
        elif method == "single_lora":
            model.set_active_adapters([adapter])
        model.set_trainable(task, adapter)
        print(f"[stage:start] method={method} stage={task} active_adapters={model.active_adapters}", flush=True)
        train_stage(model, task, data[task][0], tokenizer, args, device, args.seed + stage_index,
                    adapter=adapter if method == "stacked_lora" else None, previous_a=previous_a,
                    previous_adapters=prior_adapters)

        metadata = {
            "method": method, "stage": task, "stage_index": stage_index, "model_name": args.model_name,
            "task_labels": task_labels, "task_sequence": args.task_sequence, "adapters": adapters,
            "target_modules": args.target_modules, "lora_rank": method_lora_rank,
            "lora_alpha": args.lora_alpha, "adapter_eval_mode": args.adapter_eval_mode,
            "orthogonal_penalty": args.orthogonal_penalty_type if method == "stacked_lora" else None,
            "orthogonal_penalty_type": args.orthogonal_penalty_type if method == "stacked_lora" else None,
            "orthogonal_penalty_weight": args.orthogonal_penalty_weight if method == "stacked_lora" else 0.0,
            "task_gate_adapters": model.task_gate_adapters if method == "stacked_lora" else {},
            "orthogonal_schedule": args.orthogonal_schedule if method == "stacked_lora" else None,
            "orthogonal_delay_epochs": args.orthogonal_delay_epochs if method == "stacked_lora" else 0,
        }
        if getattr(args, "save_checkpoints", True):
            checkpoint = method_dir / f"{method if method != 'full' else 'full'}_after_{task}"
            model.save_checkpoint(checkpoint, metadata)
            tokenizer.save_pretrained(checkpoint)

        for eval_task in args.task_sequence[: stage_index + 1]:
            if method == "stacked_lora" and args.adapter_eval_mode == "learned_gates":
                model.set_task_gate(eval_task)
            elif method == "stacked_lora":
                active = adapters if args.adapter_eval_mode == "all" else [eval_task]
                model.set_active_adapters(active)
            accuracy, loss = evaluate(model, eval_task, data[eval_task][1], tokenizer, args, device)
            result_rows.append({"method": method, "stage": task, "eval_task": eval_task,
                                "accuracy": accuracy, "loss": loss})
            print(f"[eval] method={method} stage={task} task={eval_task} accuracy={accuracy:.4f}", flush=True)
        if method == "stacked_lora" and args.adapter_eval_mode == "learned_gates":
            model.set_task_gate(task)
        elif method == "stacked_lora":
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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] device={device} tasks={args.task_sequence} methods={args.methods}", flush=True)
    for method in args.methods:
        seed_everything(args.seed)
        run_method(method, data, tokenizer, args, device)


if __name__ == "__main__":
    main()
