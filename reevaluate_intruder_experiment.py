#!/usr/bin/env python3
"""Reevaluate saved continual-learning checkpoints without retraining."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO_PARENT = Path(__file__).resolve().parent.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from dynamic_lora.intruder_experiment.data import load_and_sample_tasks
from dynamic_lora.intruder_experiment.modeling import ContinualClassifier
from dynamic_lora.run_intruder_experiment import evaluate, seed_everything


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--experiment-dir", type=Path,
        default=Path("outputs/intruder_experiment_dolma"),
    )
    parser.add_argument(
        "--method", choices=("full", "single_lora", "stacked_lora"),
        default="stacked_lora",
    )
    parser.add_argument(
        "--adapter-eval-mode", choices=("all", "task_specific"),
        default="all",
        help="For stacked LoRA, activate all accumulated adapters or only the evaluated task adapter.",
    )
    parser.add_argument("--output-name", default="results.csv")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args()


def method_paths(experiment_dir: Path, method: str) -> tuple[Path, str]:
    folder = "full_finetune" if method == "full" else method
    prefix = "full" if method == "full" else method
    return experiment_dir / folder, prefix


def write_results(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "stage", "eval_task", "accuracy", "loss"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cli = arguments()
    config = json.loads((cli.experiment_dir / "config.json").read_text(encoding="utf-8"))
    config["output_dir"] = str(cli.experiment_dir)
    if cli.batch_size is not None:
        config["batch_size"] = cli.batch_size
    if cli.num_workers is not None:
        config["num_workers"] = cli.num_workers
    args = argparse.Namespace(**config)

    seed_everything(args.seed)
    data = load_and_sample_tasks(
        args.task_sequence,
        args.train_samples_per_task,
        args.eval_samples_per_task,
        args.seed,
        cli.experiment_dir / "sampled_data",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    method_dir, prefix = method_paths(cli.experiment_dir, cli.method)
    rows: list[dict] = []
    for stage_index, stage in enumerate(args.task_sequence):
        checkpoint = method_dir / f"{prefix}_after_{stage}"
        model, metadata = ContinualClassifier.load_checkpoint(checkpoint)
        model.to(device)
        if cli.method == "stacked_lora":
            if cli.adapter_eval_mode == "all":
                model.set_active_adapters(metadata["adapters"])
            else:
                model.set_active_adapters([stage])
        print(
            f"[reeval:stage] method={cli.method} stage={stage} "
            f"active_adapters={model.active_adapters}",
            flush=True,
        )
        for eval_task in args.task_sequence[: stage_index + 1]:
            accuracy, loss = evaluate(model, eval_task, data[eval_task][1], tokenizer, args, device)
            rows.append({
                "method": cli.method,
                "stage": stage,
                "eval_task": eval_task,
                "accuracy": accuracy,
                "loss": loss,
            })
            print(
                f"[reeval] stage={stage} task={eval_task} accuracy={accuracy:.4f}",
                flush=True,
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output = method_dir / cli.output_name
    write_results(output, rows)
    eval_config = {
        "method": cli.method,
        "adapter_eval_mode": cli.adapter_eval_mode,
        "checkpoint_source": str(method_dir),
        "output": str(output),
        "task_sequence": args.task_sequence,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "seed": args.seed,
    }
    (method_dir / "evaluation_config.json").write_text(
        json.dumps(eval_config, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[done] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
