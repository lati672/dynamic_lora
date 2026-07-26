"""Robust loading, normalization, and reproducible sampling of the six tasks."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset


@dataclass(frozen=True)
class TaskSpec:
    dataset: str
    config: str | None
    num_labels: int
    train_splits: tuple[str, ...] = ("train",)
    eval_splits: tuple[str, ...] = ("validation", "test")


TASK_SPECS = {
    "mnli": TaskSpec("glue", "mnli", 3, eval_splits=("validation_matched", "validation_mismatched")),
    "qqp": TaskSpec("glue", "qqp", 2),
    "sst2": TaskSpec("glue", "sst2", 2),
    "siqa": TaskSpec("social_i_qa", None, 3),
    "winogrande": TaskSpec("winogrande", "winogrande_debiased", 2),
    "fever": TaskSpec("fever", "v1.0", 3),
}


def _first(record: dict[str, Any], *names: str, default: str = "") -> str:
    for name in names:
        value = record.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _label(record: dict[str, Any], task: str) -> int | None:
    raw = record.get("label", record.get("gold_label"))
    if raw is None or raw == -1 or str(raw).strip() in {"", "-1"}:
        return None
    if isinstance(raw, int):
        # SIQA and WinoGrande encode their answers as 1..N; GLUE is already 0..N-1.
        return raw - 1 if task in {"siqa", "winogrande"} and raw > 0 else raw
    value = str(raw).strip().upper().replace("_", " ")
    maps = {
        "mnli": {"ENTAILMENT": 0, "NEUTRAL": 1, "CONTRADICTION": 2},
        "qqp": {"NOT DUPLICATE": 0, "DUPLICATE": 1, "0": 0, "1": 1},
        "sst2": {"NEGATIVE": 0, "POSITIVE": 1, "0": 0, "1": 1},
        "siqa": {"1": 0, "2": 1, "3": 2},
        "winogrande": {"1": 0, "2": 1},
        "fever": {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2},
    }
    return maps[task].get(value)


def normalize_example(record: dict[str, Any], task: str) -> dict[str, Any] | None:
    """Convert a source row into ``text, label, task_name`` or skip it with a warning."""
    if task == "mnli":
        a, b = _first(record, "premise"), _first(record, "hypothesis")
        text = f"Premise: {a}\nHypothesis: {b}\nQuestion: What is the relation between the premise and hypothesis?"
        required = (a, b)
    elif task == "qqp":
        a, b = _first(record, "question1"), _first(record, "question2")
        text = f"Question 1: {a}\nQuestion 2: {b}\nAre these questions duplicates?"
        required = (a, b)
    elif task == "sst2":
        sentence = _first(record, "sentence", "text")
        text, required = f"Sentence: {sentence}\nSentiment?", (sentence,)
    elif task == "siqa":
        context, question = _first(record, "context"), _first(record, "question")
        a, b, c = (_first(record, key) for key in ("answerA", "answerB", "answerC"))
        text = f"Context: {context}\nQuestion: {question}\nA: {a}\nB: {b}\nC: {c}\nChoose the best answer."
        required = (context, question, a, b, c)
    elif task == "winogrande":
        sentence = _first(record, "sentence")
        a, b = _first(record, "option1"), _first(record, "option2")
        text = f"Sentence: {sentence}\nOption 1: {a}\nOption 2: {b}\nChoose the correct option."
        required = (sentence, a, b)
    elif task == "fever":
        claim = _first(record, "claim")
        evidence = _first(record, "evidence", "evidence_text")
        text = f"Claim: {claim}" + (f"\nEvidence: {evidence}" if evidence else "") + "\nVerify the claim."
        required = (claim,)
    else:
        raise KeyError(f"Unknown task: {task}")
    label = _label(record, task)
    if not all(required) or label is None or not 0 <= label < TASK_SPECS[task].num_labels:
        warnings.warn(f"Skipping malformed/unlabelled {task} row; fields={sorted(record)}", stacklevel=2)
        return None
    return {"text": text, "label": label, "task_name": task}


def _choose_split(dataset: DatasetDict, candidates: tuple[str, ...], task: str, purpose: str) -> str:
    for name in candidates:
        if name in dataset:
            return name
    available = ", ".join(dataset.keys())
    raise ValueError(f"{task}: no usable {purpose} split (tried {candidates}; available: {available})")


def _candidate_indices(length: int, seed: int, cache: Path) -> tuple[list[int], bool]:
    import random

    if cache.exists():
        indices = [int(index) for index in json.loads(cache.read_text())]
        valid = [index for index in indices if 0 <= index < length]
        if len(valid) != len(indices):
            warnings.warn(f"Ignoring invalid cached indices in {cache}", stacklevel=2)
        remainder = [index for index in range(length) if index not in set(valid)]
        random.Random(seed).shuffle(remainder)
        return valid + remainder, True
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    return indices, False


def load_and_sample_task(
    task: str, train_count: int, eval_count: int, seed: int, sample_dir: Path
) -> tuple[Dataset, Dataset]:
    spec = TASK_SPECS[task]
    try:
        raw = load_dataset(spec.dataset, spec.config) if spec.config else load_dataset(spec.dataset)
    except Exception as exc:
        raise RuntimeError(f"Failed to load {task} ({spec.dataset}/{spec.config}): {exc}") from exc
    train_split = _choose_split(raw, spec.train_splits, task, "training")
    eval_split = _choose_split(raw, spec.eval_splits, task, "evaluation")

    def convert(source: Dataset, purpose: str, count: int, offset: int) -> Dataset:
        cache = sample_dir / f"{task}_{purpose}_indices.json"
        indices, _ = _candidate_indices(len(source), seed + offset, cache)
        rows = []
        selected = []
        for index in indices:
            row = normalize_example(dict(source[index]), task)
            if row is not None:
                rows.append(row)
                selected.append(index)
            if len(rows) == count:
                break
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(selected, indent=2) + "\n")
        if len(rows) < min(count, len(source)):
            warnings.warn(f"{task}: requested {count} {purpose} rows but obtained {len(rows)} valid rows", stacklevel=2)
        return Dataset.from_list(rows)

    return convert(raw[train_split], "train", train_count, 0), convert(raw[eval_split], "eval", eval_count, 10_000)


def load_and_sample_tasks(
    tasks: list[str], train_count: int, eval_count: int, seed: int, sample_dir: Path
) -> dict[str, tuple[Dataset, Dataset]]:
    unknown = set(tasks) - TASK_SPECS.keys()
    if unknown:
        raise ValueError(f"Unknown tasks: {sorted(unknown)}")
    return {task: load_and_sample_task(task, train_count, eval_count, seed, sample_dir) for task in tasks}
