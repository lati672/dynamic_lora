#!/usr/bin/env python3
"""Scale full-finetuning intruder singular components and reevaluate checkpoints.

For every selected full-weight matrix, this script computes reduced SVDs and
considers their top-k components. A tuned left singular vector is an intruder
when its maximum absolute cosine similarity with the pretrained top-k left
singular vectors is below epsilon. All detected components are scaled with

    W_scaled = W_tuned + (lambda_scale - 1) * sum_i u_i sigma_i v_i^T

where W_tuned = W_0 + Delta W and i ranges over detected intruders.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

REPO_PARENT = Path(__file__).resolve().parent.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

from dynamic_lora.intruder_experiment.data import load_and_sample_tasks
from dynamic_lora.intruder_experiment.modeling import ContinualClassifier, matches_target
from dynamic_lora.run_intruder_experiment import evaluate, write_results

LAYER_RE = re.compile(r"(?:^|\.)layers?\.(\d+)(?:\.|$)")
LAMBDA_OPTIONS = (0.0, 0.5)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Scale full-weight intruder singular components, then reevaluate.",
    )
    parser.add_argument(
        "--experiment-dir", type=Path,
        default=Path("outputs/intruder_experiment_dolma"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/intruder_experiment_dolma/full_finetune_intruder_scaled"),
    )
    parser.add_argument("--base-model", help="Override model_name from experiment config.")
    parser.add_argument("--epsilon", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--lambda-scale", type=float, choices=LAMBDA_OPTIONS, default=0.5,
        help="Multiplier selected from the discrete lambda option set.",
    )
    parser.add_argument(
        "--modules", nargs="+", default=["q_proj", "v_proj", "up_proj", "down_proj"],
    )
    parser.add_argument(
        "--layers", nargs="+", type=int,
        help="Transformer layers to modify. Omit to process every matching layer.",
    )
    parser.add_argument(
        "--svd-device", choices=("auto", "cuda", "cpu"), default="auto",
        help="Device used one matrix at a time for float32 SVD and reconstruction.",
    )
    parser.add_argument(
        "--similarity-chunk-size", type=int, default=256,
        help="Tuned singular vectors per pretrained-top-k similarity chunk.",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--save-scaled-checkpoints", action=argparse.BooleanOptionalAction, default=False,
        help="Save each scaled model.pt. Disabled by default because six full checkpoints need about 14 GB.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.epsilon <= 1.0:
        parser.error("--epsilon must be between 0 and 1")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.similarity_chunk_size <= 0:
        parser.error("--similarity-chunk-size must be positive")
    if args.layers is not None and any(layer < 0 for layer in args.layers):
        parser.error("--layers values must be non-negative")
    if args.svd_device == "cuda" and not torch.cuda.is_available():
        parser.error("--svd-device cuda requested, but CUDA is unavailable")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def selected_linears(
    model: nn.Module,
    modules: list[str],
    layers: list[int] | None,
) -> dict[str, tuple[int, str, nn.Linear]]:
    """Return matching encoder linears keyed by their full module path."""
    selected = {}
    layer_filter = None if layers is None else set(layers)
    for path, module in model.named_modules():
        match = LAYER_RE.search(path)
        if match is None or not isinstance(module, nn.Linear):
            continue
        layer = int(match.group(1))
        if layer_filter is not None and layer not in layer_filter:
            continue
        suffix = next((name for name in modules if matches_target(path, name)), None)
        if suffix is not None:
            selected[path] = (layer, suffix, module)
    return selected


def resolve_svd_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


@torch.inference_mode()
def reduced_svd(weight: torch.Tensor, device: torch.device):
    """Compute every reduced-SVD triplet in float32 on the requested device."""
    matrix = weight.detach().to(device=device, dtype=torch.float32)
    return torch.linalg.svd(matrix, full_matrices=False)


@torch.inference_mode()
def all_vector_max_similarities(
    base_u: torch.Tensor,
    tuned_u: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """For every tuned u_i, maximize |u_0_j^T u_i| over every base vector j."""
    if base_u.shape[0] != tuned_u.shape[0]:
        raise ValueError(f"Left singular-vector dimensions differ: {base_u.shape} vs {tuned_u.shape}")
    maxima = []
    base_u_t = base_u.T
    for start in range(0, tuned_u.shape[1], chunk_size):
        chunk = tuned_u[:, start : start + chunk_size]
        maxima.append((base_u_t @ chunk).abs().amax(dim=0).cpu())
    return torch.cat(maxima)


@torch.inference_mode()
def scale_matrix_intruders(
    tuned_weight: torch.Tensor,
    base_u_cpu: torch.Tensor,
    epsilon: float,
    lambda_scale: float,
    top_k: int,
    device: torch.device,
    chunk_size: int,
) -> dict[str, float | int]:
    """Apply the requested singular-component scaling to one tuned matrix."""
    original_device = tuned_weight.device
    original_dtype = tuned_weight.dtype
    tuned_u, tuned_s, tuned_vh = reduced_svd(tuned_weight, device)
    matrix_reduced_rank = int(tuned_s.numel())
    k = min(top_k, matrix_reduced_rank, base_u_cpu.shape[1])
    tuned_u = tuned_u[:, :k]
    tuned_s = tuned_s[:k]
    tuned_vh = tuned_vh[:k, :]
    base_u = base_u_cpu.to(device=device, dtype=torch.float32)
    maxima = all_vector_max_similarities(base_u, tuned_u, chunk_size)
    intruder_mask_cpu = maxima < epsilon
    intruder_count = int(intruder_mask_cpu.sum().item())

    correction_norm = 0.0
    if intruder_count:
        intruder_mask = intruder_mask_cpu.to(device=device)
        selected_u = tuned_u[:, intruder_mask]
        selected_s = tuned_s[intruder_mask]
        selected_vh = tuned_vh[intruder_mask, :]
        correction = (lambda_scale - 1.0) * ((selected_u * selected_s) @ selected_vh)
        correction_norm = float(torch.linalg.vector_norm(correction).item())
        scaled = tuned_weight.detach().to(device=device, dtype=torch.float32) + correction
        tuned_weight.copy_(scaled.to(device=original_device, dtype=original_dtype))

    stats = {
        "matrix_reduced_rank": matrix_reduced_rank,
        "top_k": k,
        "num_singular_vectors": int(tuned_s.numel()),
        "num_intruders": intruder_count,
        "intruder_fraction": intruder_count / int(tuned_s.numel()),
        "mean_max_similarity": float(maxima.mean().item()),
        "min_max_similarity": float(maxima.min().item()),
        "correction_frobenius_norm": correction_norm,
    }
    del tuned_u, tuned_s, tuned_vh, base_u
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return stats


@torch.inference_mode()
def build_base_left_vectors(
    model_name: str,
    modules: list[str],
    layers: list[int] | None,
    top_k: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, tuple[int, str]]]:
    """Compute and CPU-cache pretrained top-k left singular vectors once."""
    print(f"[base] loading {model_name}", flush=True)
    base = AutoModel.from_pretrained(model_name)
    selected = selected_linears(base, modules, layers)
    if not selected:
        raise ValueError(f"No base matrices matched modules={modules}, layers={layers or 'all'}")
    vectors = {}
    identities = {}
    for index, (path, (layer, suffix, module)) in enumerate(selected.items(), start=1):
        print(f"[base:svd] {index}/{len(selected)} layer={layer} module={suffix}", flush=True)
        u, singular_values, vh = reduced_svd(module.weight, device)
        vectors[path] = u[:, : min(top_k, u.shape[1])].cpu()
        identities[path] = (layer, suffix)
        del u, singular_values, vh
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del base
    return vectors, identities


def save_scaled_checkpoint(
    model: ContinualClassifier,
    source: Path,
    target: Path,
    metadata: dict,
    scaling_config: dict,
) -> None:
    target.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target / "model.pt")
    saved_metadata = dict(metadata)
    saved_metadata["intruder_scaling"] = scaling_config
    (target / "metadata.json").write_text(json.dumps(saved_metadata, indent=2) + "\n")
    for path in source.iterdir():
        if path.is_file() and path.name not in {"model.pt", "metadata.json"}:
            shutil.copy2(path, target / path.name)


def write_scaling_rows(path: Path, rows: list[dict]) -> None:
    fields = [
        "stage", "layer", "module", "module_path", "epsilon", "lambda_scale",
        "lambda_options",
        "matrix_reduced_rank", "top_k",
        "num_singular_vectors", "num_intruders", "intruder_fraction",
        "mean_max_similarity", "min_max_similarity", "correction_frobenius_norm",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cli = arguments()
    experiment_config = json.loads((cli.experiment_dir / "config.json").read_text())
    model_name = cli.base_model or experiment_config["model_name"]
    if cli.batch_size is not None:
        experiment_config["batch_size"] = cli.batch_size
    if cli.num_workers is not None:
        experiment_config["num_workers"] = cli.num_workers
    eval_args = argparse.Namespace(**experiment_config)
    seed_everything(eval_args.seed)

    cli.output_dir.mkdir(parents=True, exist_ok=True)
    svd_device = resolve_svd_device(cli.svd_device)
    base_u, identities = build_base_left_vectors(
        model_name, cli.modules, cli.layers, cli.top_k, svd_device,
    )
    print(f"[base] cached top-{cli.top_k} reduced-SVD U for {len(base_u)} matrices", flush=True)

    data = load_and_sample_tasks(
        eval_args.task_sequence,
        eval_args.train_samples_per_task,
        eval_args.eval_samples_per_task,
        eval_args.seed,
        cli.experiment_dir / "sampled_data",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_root = cli.experiment_dir / "full_finetune"
    result_rows = []
    scaling_rows = []
    for stage_index, stage in enumerate(eval_args.task_sequence):
        checkpoint = source_root / f"full_after_{stage}"
        print(f"[checkpoint] loading {checkpoint}", flush=True)
        model, metadata = ContinualClassifier.load_checkpoint(checkpoint)
        tuned = selected_linears(model.encoder, cli.modules, cli.layers)
        missing = set(base_u) - set(tuned)
        extra = set(tuned) - set(base_u)
        if missing or extra:
            raise ValueError(f"Matrix mismatch for {checkpoint}: missing={sorted(missing)}, extra={sorted(extra)}")

        for index, path in enumerate(base_u, start=1):
            layer, suffix = identities[path]
            print(
                f"[scale:svd] stage={stage} {index}/{len(base_u)} "
                f"layer={layer} module={suffix}",
                flush=True,
            )
            stats = scale_matrix_intruders(
                tuned[path][2].weight,
                base_u[path],
                cli.epsilon,
                cli.lambda_scale,
                cli.top_k,
                svd_device,
                cli.similarity_chunk_size,
            )
            scaling_rows.append({
                "stage": stage, "layer": layer, "module": suffix, "module_path": path,
                "epsilon": cli.epsilon, "lambda_scale": cli.lambda_scale,
                "lambda_options": json.dumps(LAMBDA_OPTIONS), **stats,
            })
            print(
                f"[scale] stage={stage} layer={layer} module={suffix} "
                f"intruders={stats['num_intruders']}/{stats['num_singular_vectors']}",
                flush=True,
            )

        model.to(eval_device)
        for eval_task in eval_args.task_sequence[: stage_index + 1]:
            accuracy, loss = evaluate(model, eval_task, data[eval_task][1], tokenizer, eval_args, eval_device)
            result_rows.append({
                "method": "full_intruder_scaled", "stage": stage,
                "eval_task": eval_task, "accuracy": accuracy, "loss": loss,
            })
            print(f"[eval] stage={stage} task={eval_task} accuracy={accuracy:.4f}", flush=True)

        if cli.save_scaled_checkpoints:
            save_scaled_checkpoint(
                model, checkpoint, cli.output_dir / f"full_scaled_after_{stage}", metadata,
                {"epsilon": cli.epsilon, "lambda_scale": cli.lambda_scale,
                 "lambda_options": list(LAMBDA_OPTIONS),
                 "top_k": cli.top_k, "all_singular_vectors": False,
                 "modules": cli.modules,
                 "layers": cli.layers or "all"},
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        write_results(cli.output_dir / "results.csv", result_rows)
        write_scaling_rows(cli.output_dir / "intruder_scaling.csv", scaling_rows)

    output_config = {
        "source_experiment": str(cli.experiment_dir),
        "source_method": "full_finetune",
        "base_model": model_name,
        "epsilon": cli.epsilon,
        "lambda_scale": cli.lambda_scale,
        "lambda_options": list(LAMBDA_OPTIONS),
        "top_k": cli.top_k,
        "all_singular_vectors": False,
        "svd": "complete reduced SVD (full_matrices=False)",
        "intruder_definition": "top-k tuned u_i with max over pretrained top-k |u_base_j^T u_tuned_i| < epsilon",
        "formula": "W_scaled = W_tuned + (lambda_scale - 1) * sum_i u_i sigma_i v_i^T",
        "modules": cli.modules,
        "layers": cli.layers or "all",
        "svd_device": str(svd_device),
        "save_scaled_checkpoints": cli.save_scaled_checkpoints,
        "task_sequence": eval_args.task_sequence,
        "results": str(cli.output_dir / "results.csv"),
        "scaling_details": str(cli.output_dir / "intruder_scaling.csv"),
    }
    (cli.output_dir / "config.json").write_text(
        json.dumps(output_config, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[done] wrote {cli.output_dir}", flush=True)


if __name__ == "__main__":
    main()
