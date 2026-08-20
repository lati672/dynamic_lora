#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from itertools import product
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoTokenizer

REPO_PARENT = Path(__file__).resolve().parent.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from dynamic_lora.intruder_experiment.data import TASK_SPECS, load_and_sample_tasks
from dynamic_lora.run_intruder_experiment import run_method, seed_everything


def arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-name", default="Kt672/Dolma_pretain")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/intruder_stacked_advanced_sweep"))
    parser.add_argument("--train-samples-per-task", type=int, default=1000)
    parser.add_argument("--eval-samples-per-task", type=int, default=500)
    parser.add_argument("--task-sequence", nargs="+", default=list(TASK_SPECS))
    parser.add_argument("--ranks", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--evaluation-modes", nargs="+", choices=("all", "task_specific", "fixed_gates"), default=["all", "task_specific", "fixed_gates"])
    parser.add_argument("--penalty-types", nargs="+", choices=("a_subspace", "effective_update"), default=["a_subspace", "effective_update"])
    parser.add_argument("--orthogonal-weight", type=float, default=0.1)
    parser.add_argument("--orthogonal-delay-epochs", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--lora-alpha", type=float, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+",
                        default=["q_proj", "v_proj", "up_proj", "down_proj"])
    parser.add_argument("--adapter-eval-mode", choices=("all", "task_specific", "fixed_gates"), default="all")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--focused-suggestions", action="store_true",
        help="Run eight fixed-gate variants covering orthogonal weight, rank, alpha, LR, and epochs.")
    parser.add_argument(
        "--save-checkpoints", action=argparse.BooleanOptionalAction, default=False,
        help="Keep per-task checkpoints; disabled by default to minimize disk use.",
    )
    args = parser.parse_args()
    if args.orthogonal_weight < 0:
        parser.error("orthogonal weight must be non-negative")
    if any(rank <= 0 for rank in args.ranks):
        parser.error("ranks must be positive")
    return args


def variants(args):
    if args.focused_suggestions:
        return [
            (16, "fixed_gates", "effective_update", 0.10, 32, 2e-5, 2),
            (16, "fixed_gates", "effective_update", 0.01, 32, 2e-5, 2),
            (16, "fixed_gates", "effective_update", 0.03, 32, 2e-5, 2),
            (32, "fixed_gates", "effective_update", 0.03, 32, 2e-5, 2),
            (16, "fixed_gates", "effective_update", 0.03, 16, 2e-5, 2),
            (16, "fixed_gates", "effective_update", 0.03, 32, 5e-5, 2),
            (16, "fixed_gates", "effective_update", 0.03, 32, 2e-5, 4),
            (32, "fixed_gates", "effective_update", 0.03, 32, 5e-5, 4),
        ]
    return [
        (rank, mode, penalty, args.orthogonal_weight, args.lora_alpha,
         args.learning_rate, args.epochs)
        for rank, mode, penalty in product(args.ranks, args.evaluation_modes, args.penalty_types)
    ]


def variant_name(rank, evaluation_mode, penalty_type, weight, alpha, learning_rate, epochs):
    return (f"r{rank}_{evaluation_mode}_{penalty_type}_w{weight:g}_a{alpha:g}_"
            f"lr{learning_rate:g}_e{epochs}")


def complete(path, task_count):
    if not path.exists():
        return False
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle)) == task_count * (task_count + 1) // 2


def summary(path, name, rank, evaluation_mode, penalty_type, weight, alpha,
            learning_rate, epochs):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    stages = list(dict.fromkeys(row["stage"] for row in rows))
    scores = {(row["stage"], row["eval_task"]): float(row["accuracy"]) for row in rows}
    final = [scores[(stages[-1], task)] for task in stages]
    acquisition = [scores[(task, task)] for task in stages]
    forgetting = []
    for index, task in enumerate(stages[:-1]):
        history = [scores[(stage, task)] for stage in stages[index:]]
        forgetting.append(max(history) - history[-1])
    result = {
        "variant": name,
        "rank": rank,
        "evaluation_mode": evaluation_mode,
        "penalty_type": penalty_type,
        "orthogonal_weight": weight,
        "lora_alpha": alpha,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "final_average_accuracy": mean(final),
        "acquisition_average_accuracy": mean(acquisition),
        "average_forgetting": mean(forgetting),
    }
    result.update({f"final_{task}": score for task, score in zip(stages, final)})
    return result


def write_summary(path, rows):
    rows.sort(key=lambda row: (-row["final_average_accuracy"], row["average_forgetting"]))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summary] wrote {path}", flush=True)


def main():
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    data = load_and_sample_tasks(
        args.task_sequence, args.train_samples_per_task, args.eval_samples_per_task,
        args.seed, args.output_dir / "sampled_data",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries = []
    print(f"[sweep] device={device} variants={len(variants(args))}", flush=True)
    for variant in variants(args):
        rank, evaluation_mode, penalty_type, weight, alpha, learning_rate, epochs = variant
        name = variant_name(*variant)
        variant_dir = args.output_dir / name
        results = variant_dir / "stacked_lora" / "results.csv"
        run_args = argparse.Namespace(**vars(args))
        run_args.output_dir = variant_dir
        run_args.methods = ["stacked_lora"]
        run_args.lora_rank = rank
        run_args.adapter_eval_mode = evaluation_mode
        run_args.orthogonal_penalty_type = penalty_type
        run_args.orthogonal_penalty_weight = weight
        run_args.lora_alpha = alpha
        run_args.learning_rate = learning_rate
        run_args.epochs = epochs
        run_args.orthogonal_schedule = "linear"
        run_args.save_checkpoints = args.save_checkpoints
        variant_dir.mkdir(parents=True, exist_ok=True)
        config = vars(run_args) | {
            "output_dir": str(variant_dir),
            "sweep_variant": name,
            "orthogonal_penalty_normalization": ("global_element_mean" if penalty_type == "a_subspace" else "mean_squared_frobenius_cosine"),
        }
        (variant_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        if args.overwrite or not complete(results, len(args.task_sequence)):
            print(f"[variant:start] {name}", flush=True)
            seed_everything(args.seed)
            run_method("stacked_lora", data, tokenizer, run_args, device)
        else:
            print(f"[variant:skip] {name}", flush=True)
        summaries.append(summary(results, name, rank, evaluation_mode, penalty_type,
                                 weight, alpha, learning_rate, epochs))
        write_summary(args.output_dir / "sweep_summary.csv", summaries)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
