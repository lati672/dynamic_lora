#!/usr/bin/env python3
"""Compare the pretrained Dolma model with the three final continual checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch

from eval_all_dolma import (
    apply_training_metadata,
    evaluate,
    evaluate_baseline,
    prepare_cache,
    resolve_dtype,
)

FINAL_CHECKPOINTS = {
    "full": ("full_finetune", "full_after_fever"),
    "single_lora": ("single_lora", "single_lora_after_fever"),
    "stacked_lora": ("stacked_lora", "stacked_lora_after_fever"),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("outputs/intruder_experiment_dolma"),
    )
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=Path("artifacts/dolma_full_20k/final"),
    )
    parser.add_argument("--training-metadata", type=Path)
    parser.add_argument("--dataset-id")
    parser.add_argument("--dataset-config")
    parser.add_argument("--num-documents", type=int, default=1000)
    parser.add_argument("--skip-documents", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--shuffle-buffer-size", type=int)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--projection-chunk-size", type=int, default=128)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    args.output_dir = (
        args.output_dir
        or args.experiment_dir / "dolma_pretrained_comparison"
    )
    args.cache_dir = (
        args.cache_dir
        or args.output_dir / "heldout_token_cache"
    )
    for name in (
        "num_documents",
        "max_length",
        "batch_size",
        "projection_chunk_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.log_every < 0:
        parser.error("--log-every must be non-negative")
    return args


def final_checkpoints(experiment_dir: Path) -> list[tuple[Path, dict]]:
    found = []
    for method, (folder, checkpoint_name) in FINAL_CHECKPOINTS.items():
        checkpoint = experiment_dir / folder / checkpoint_name
        metadata_file = checkpoint / "metadata.json"
        model_file = checkpoint / "model.pt"
        if not metadata_file.is_file() or not model_file.is_file():
            raise FileNotFoundError(f"Missing final checkpoint for {method}: {checkpoint}")
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if metadata["stage"] != "fever":
            raise ValueError(f"Expected final FEVER checkpoint: {checkpoint}")
        found.append((checkpoint, metadata))
    return found


def write_outputs(output_dir: Path, results: list[dict], config: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "stage",
        "checkpoint",
        "active_adapters",
        "num_documents",
        "num_predicted_tokens",
        "max_length",
        "mean_nll",
        "perplexity",
        "dtype",
        "elapsed_seconds",
    ]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(
                row | {"active_adapters": ",".join(row["active_adapters"])}
            )
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    baseline = results[0]
    lines = [
        "# Dolma pretrained-model comparison",
        "",
        "This evaluation measures held-out Dolma next-token negative log-likelihood",
        "and perplexity. Lower values are better. The three continual models are",
        "their final checkpoints after FEVER; stacked LoRA activates all six",
        "accumulated adapters.",
        "",
        "| Model | Mean NLL | Perplexity | PPL change vs. pretrained |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "pretrained_dolma": "Pretrained Dolma",
        "full": "Full-weight after FEVER",
        "single_lora": "Single LoRA after FEVER",
        "stacked_lora": "Stacked LoRA after FEVER",
    }
    for row in results:
        change = 100.0 * (row["perplexity"] / baseline["perplexity"] - 1.0)
        lines.append(
            f"| {labels[row['method']]} | {row['mean_nll']:.6f} | "
            f"{row['perplexity']:.4f} | {change:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "The evaluation uses the same held-out token cache for every model.",
            f"Documents: {config['num_documents']}; maximum length: "
            f"{config['max_length']}; seed: {config['seed']}.",
            "",
            "Classification heads are ignored. Each encoder is evaluated as a",
            "causal language model with the model's tied token-embedding output",
            "projection, matching the pretrained Llama configuration.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = arguments()
    training_metadata = apply_training_metadata(args)
    checkpoints = final_checkpoints(args.experiment_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment_dir": str(args.experiment_dir),
        "baseline_model": str(args.baseline_model),
        "dataset_id": args.dataset_id,
        "dataset_config": args.dataset_config,
        "num_documents": args.num_documents,
        "skip_documents": args.skip_documents,
        "seed": args.seed,
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "projection_chunk_size": args.projection_chunk_size,
        "dtype": args.dtype,
        "final_checkpoints_only": True,
        "stacked_adapter_eval_mode": "all",
        "pretraining_metadata": training_metadata,
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    cache = prepare_cache(args, str(args.baseline_model))
    dtype = resolve_dtype(args.dtype)
    results = []

    baseline_file = args.output_dir / "pretrained_dolma.json"
    if baseline_file.is_file() and not args.overwrite:
        baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
    else:
        print(f"[eval:start] pretrained Dolma: {args.baseline_model}", flush=True)
        baseline = evaluate_baseline(args, cache, dtype)
        baseline["method"] = "pretrained_dolma"
        baseline_file.write_text(
            json.dumps(baseline, indent=2) + "\n",
            encoding="utf-8",
        )
    results.append(baseline)
    print(
        f"[eval:done] method=pretrained_dolma nll={baseline['mean_nll']:.6f} "
        f"ppl={baseline['perplexity']:.4f}",
        flush=True,
    )

    for checkpoint, metadata in checkpoints:
        result_file = args.output_dir / f"{metadata['method']}_after_fever.json"
        if result_file.is_file() and not args.overwrite:
            result = json.loads(result_file.read_text(encoding="utf-8"))
        else:
            print(f"[eval:start] {checkpoint}", flush=True)
            result = evaluate(args, checkpoint, metadata, cache, dtype)
            result_file.write_text(
                json.dumps(result, indent=2) + "\n",
                encoding="utf-8",
            )
        results.append(result)
        print(
            f"[eval:done] method={result['method']} nll={result['mean_nll']:.6f} "
            f"ppl={result['perplexity']:.4f}",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_outputs(args.output_dir, results, config)
    print(f"[done] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
