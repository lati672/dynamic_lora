"""Evaluate every saved continual-learning checkpoint on one Dolma sample.

Intruder-experiment checkpoints are classifiers whose ``encoder`` is the
original causal-LM backbone. This evaluates that backbone with its tied token
embedding as the language-model head; classification heads are ignored.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from dynamic_lora.intruder_experiment.modeling import ContinualClassifier

METHOD_DIRS = ("full_finetune", "single_lora", "stacked_lora")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-dir", type=Path, required=True)
    p.add_argument("--dataset-id", default="allenai/dolma")
    p.add_argument("--dataset-config", default="v1_6-sample")
    p.add_argument("--num-documents", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--shuffle-buffer-size", type=int, default=10_000)
    p.add_argument("--max-length", type=int, default=1_024)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--projection-chunk-size", type=int, default=128,
                   help="Token positions projected onto the vocabulary at once (controls VRAM).")
    p.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    p.add_argument("--token", default=os.getenv("HF_TOKEN"))
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--overwrite-cache", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--methods", nargs="+", choices=METHOD_DIRS, default=list(METHOD_DIRS))
    p.add_argument("--list-only", action="store_true")
    a = p.parse_args()
    for name in ("num_documents", "shuffle_buffer_size", "max_length", "batch_size",
                 "projection_chunk_size"):
        if getattr(a, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if a.log_every < 0:
        p.error("--log-every cannot be negative")
    a.output_dir = a.output_dir or a.experiment_dir / "dolma_eval"
    a.cache_dir = a.cache_dir or a.output_dir / "token_cache"
    return a


def resolve_dtype(name: str) -> torch.dtype:
    if name != "auto":
        return getattr(torch, name)
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def discover_checkpoints(root: Path, methods: list[str]) -> list[tuple[Path, dict[str, Any]]]:
    found = []
    for family in methods:
        family_dir = root / family
        if not family_dir.is_dir():
            print(f"[discover:skip] missing directory: {family_dir}", flush=True)
            continue
        for metadata_file in family_dir.glob("*/metadata.json"):
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            if (metadata_file.parent / "model.pt").is_file():
                found.append((metadata_file.parent, metadata))
    order = {"full": 0, "single_lora": 1, "stacked_lora": 2}
    return sorted(found, key=lambda item: (order.get(item[1]["method"], 99), item[1]["stage_index"]))


def dolma_rows(a: argparse.Namespace) -> Iterator[dict[str, Any]]:
    manifest = hf_hub_download(repo_id=a.dataset_id, filename=f"urls/{a.dataset_config}.txt",
                               repo_type="dataset", token=a.token)
    urls = [line.strip() for line in Path(manifest).read_text().splitlines() if line.strip()]
    if not urls:
        raise RuntimeError(f"Dolma URL manifest is empty: {manifest}")
    stream = load_dataset("json", data_files={"train": urls}, split="train",
                          streaming=True, token=a.token)
    stream = stream.shuffle(seed=a.seed, buffer_size=a.shuffle_buffer_size)
    return (row for row in stream if str(row.get("text", "")).strip())


def cache_spec(a: argparse.Namespace, model_name: str) -> dict[str, Any]:
    return {"dataset_id": a.dataset_id, "dataset_config": a.dataset_config,
            "num_documents": a.num_documents, "seed": a.seed,
            "shuffle_buffer_size": a.shuffle_buffer_size, "max_length": a.max_length,
            "batch_size": a.batch_size, "tokenizer": model_name}


def prepare_cache(a: argparse.Namespace, model_name: str) -> dict[str, Any]:
    metadata_file = a.cache_dir / "metadata.json"
    wanted = cache_spec(a, model_name)
    if metadata_file.is_file() and not a.overwrite_cache:
        existing = json.loads(metadata_file.read_text(encoding="utf-8"))
        mismatch = {k: (existing.get(k), v) for k, v in wanted.items() if existing.get(k) != v}
        if mismatch:
            raise RuntimeError(f"Incompatible token cache {a.cache_dir}: {mismatch}. "
                               "Use another --cache-dir or --overwrite-cache.")
        missing = [i for i in range(existing["num_batches"])
                   if not (a.cache_dir / f"batch_{i:06d}.pt").is_file()]
        if missing:
            raise RuntimeError(f"Token cache is incomplete; first missing batch is {missing[0]}")
        print(f"[cache:reuse] {a.cache_dir} batches={existing['num_batches']}", flush=True)
        return existing

    a.cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=a.token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    rows, seen, batch_index = dolma_rows(a), 0, 0
    sampled_ids: list[str] = []
    source_counts: dict[str, int] = {}
    while seen < a.num_documents:
        batch = []
        try:
            for _ in range(min(a.batch_size, a.num_documents - seen)):
                batch.append(next(rows))
        except StopIteration as error:
            raise RuntimeError(f"Dolma ended after {seen + len(batch):,} usable documents") from error
        encoded = tokenizer([row["text"] for row in batch], padding=True, truncation=True,
                            max_length=a.max_length, return_tensors="pt")
        torch.save({"input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"]},
                   a.cache_dir / f"batch_{batch_index:06d}.pt")
        for row in batch:
            sampled_ids.append(str(row.get("id", "")))
            source = str(row.get("source", "")) or "unknown"
            source_counts[source] = source_counts.get(source, 0) + 1
        seen += len(batch)
        batch_index += 1
        if a.log_every and batch_index % a.log_every == 0:
            print(f"[cache] documents={seen:,}/{a.num_documents:,}", flush=True)
    metadata = wanted | {"num_batches": batch_index, "sampled_document_ids": sampled_ids,
                         "source_counts": dict(sorted(source_counts.items()))}
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[cache:done] {a.cache_dir} batches={batch_index}", flush=True)
    return metadata


def batch_nll(model: ContinualClassifier, input_ids: torch.Tensor,
              attention_mask: torch.Tensor, chunk_size: int) -> tuple[float, int]:
    hidden = model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, :-1]
    labels, valid = input_ids[:, 1:], attention_mask[:, 1:].bool()
    hidden, labels = hidden[valid], labels[valid]
    count = labels.numel()
    if not count:
        return 0.0, 0
    embedding = model.encoder.get_input_embeddings().weight
    nll = 0.0
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        logits = F.linear(hidden[start:stop], embedding)
        nll += float(F.cross_entropy(logits.float(), labels[start:stop], reduction="sum"))
    return nll, count


def evaluate(a: argparse.Namespace, checkpoint: Path, metadata: dict[str, Any],
             cache: dict[str, Any], dtype: torch.dtype) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, loaded_metadata = ContinualClassifier.load_checkpoint(checkpoint, map_location="cpu")
    if loaded_metadata != metadata:
        raise RuntimeError(f"Metadata changed while loading {checkpoint}")
    model.to(device=device, dtype=dtype).eval()
    # Dolma has no task identity. Apply every adapter accumulated at this stage,
    # rather than a classification task's learned gate.
    active = list(metadata.get("adapters", []))
    if metadata["method"] in {"single_lora", "stacked_lora"}:
        model.set_active_adapters(active)
    total_nll, total_tokens, started = 0.0, 0, time.time()
    with torch.inference_mode():
        for index in range(cache["num_batches"]):
            batch = torch.load(a.cache_dir / f"batch_{index:06d}.pt", map_location="cpu",
                               weights_only=True)
            nll, count = batch_nll(model, batch["input_ids"].to(device),
                                   batch["attention_mask"].to(device), a.projection_chunk_size)
            total_nll += nll
            total_tokens += count
            if a.log_every and (index + 1) % a.log_every == 0:
                mean = total_nll / total_tokens
                print(f"[eval] checkpoint={checkpoint.name} batches={index + 1}/{cache['num_batches']} "
                      f"tokens={total_tokens:,} loss={mean:.6f} ppl={math.exp(mean):.4f}", flush=True)
    if not total_tokens:
        raise RuntimeError("The sample contains no predicted tokens")
    mean = total_nll / total_tokens
    return {"method": metadata["method"], "stage": metadata["stage"],
            "stage_index": metadata["stage_index"], "checkpoint": str(checkpoint),
            "base_model": metadata["model_name"], "active_adapters": active,
            "num_documents": cache["num_documents"], "num_predicted_tokens": total_tokens,
            "max_length": cache["max_length"], "mean_nll": mean,
            "perplexity": math.exp(mean) if mean < 709 else float("inf"),
            "dtype": str(dtype).removeprefix("torch."),
            "elapsed_seconds": time.time() - started, "dataset_id": cache["dataset_id"],
            "dataset_config": cache["dataset_config"], "seed": cache["seed"]}


def write_summary(output_dir: Path, results: list[dict[str, Any]]) -> None:
    fields = ["method", "stage", "stage_index", "checkpoint", "active_adapters",
              "num_documents", "num_predicted_tokens", "max_length", "mean_nll",
              "perplexity", "dtype", "elapsed_seconds"]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow(result | {"active_adapters": ",".join(result["active_adapters"])})
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    a = parse_args()
    checkpoints = discover_checkpoints(a.experiment_dir, a.methods)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found under {a.experiment_dir}")
    print(f"[discover] checkpoints={len(checkpoints)}", flush=True)
    for checkpoint, metadata in checkpoints:
        print(f"  {metadata['method']:12s} stage={metadata['stage']:12s} {checkpoint}")
    if a.list_only:
        return
    model_names = {metadata["model_name"] for _, metadata in checkpoints}
    if len(model_names) != 1:
        raise RuntimeError(f"Expected one base model/tokenizer; found {sorted(model_names)}")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    cache = prepare_cache(a, next(iter(model_names)))
    dtype, results = resolve_dtype(a.dtype), []
    for checkpoint, metadata in checkpoints:
        result_file = a.output_dir / f"{metadata['method']}_after_{metadata['stage']}.json"
        if result_file.is_file() and not a.overwrite:
            print(f"[eval:reuse] {result_file}", flush=True)
            results.append(json.loads(result_file.read_text(encoding="utf-8")))
            continue
        print(f"[eval:start] {checkpoint}", flush=True)
        result = evaluate(a, checkpoint, metadata, cache, dtype)
        result_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        results.append(result)
        print(f"[eval:done] method={result['method']} stage={result['stage']} "
              f"loss={result['mean_nll']:.6f} ppl={result['perplexity']:.4f}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_summary(a.output_dir, results)
    print(f"[done] wrote {a.output_dir / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
