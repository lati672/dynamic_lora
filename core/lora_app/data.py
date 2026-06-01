from dataclasses import dataclass

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoTokenizer

from dynamic_lora.core.lora_app.config import TrainingConfig


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def build_user_message(question_text: str, config: TrainingConfig) -> str:
    return question_text


def build_prompt(question_text: str, config: TrainingConfig) -> str:
    user_message = build_user_message(question_text, config)
    if not config.apply_chat_template:
        return user_message

    return (
        f"{config.system_prompt_with_special_tokens}"
        f"{config.user_start_tag}{user_message}{config.user_end_tag}"
        f"{config.asst_start_tag}"
    )


def prepare_example(
    tokenizer: AutoTokenizer,
    config: TrainingConfig,
    question_text: str,
    answer_text: str,
    max_length: int,
) -> dict:
    prompt = build_prompt(question_text, config)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_text = prompt + answer_text
    if config.apply_chat_template:
        full_text += config.asst_end_tag
    else:
        full_text += tokenizer.eos_token
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    labels = tokenized["input_ids"][:]
    prompt_length = min(len(prompt_ids), len(labels))
    for index in range(prompt_length):
        labels[index] = -100

    pad_token_id = tokenizer.pad_token_id
    labels = [token if token != pad_token_id else -100 for token in labels]

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
        "pad_token_id": tokenizer.pad_token_id,
    }


def collate_batch(features: list[dict]) -> Batch:
    max_length = max(len(feature["input_ids"]) for feature in features)
    pad_token_id = features[0]["pad_token_id"]
    input_ids = []
    attention_mask = []
    labels = []

    for feature in features:
        pad_length = max_length - len(feature["input_ids"])
        input_ids.append(feature["input_ids"] + [pad_token_id] * pad_length)
        attention_mask.append(feature["attention_mask"] + [0] * pad_length)
        labels.append(feature["labels"] + [-100] * pad_length)

    return Batch(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
    )


def _parse_dataset_subsets(dataset_subset: str) -> tuple[str, ...]:
    normalized = dataset_subset.replace(",", "+")
    subsets = tuple(part.strip() for part in normalized.split("+") if part.strip())
    if not subsets:
        raise ValueError("dataset_subset must contain at least one subset name")
    return subsets


def load_source_dataset(config: TrainingConfig) -> Dataset:
    subsets = _parse_dataset_subsets(config.dataset_subset)
    split_datasets = []

    for subset in subsets:
        dataset = load_dataset(config.dataset_id, subset)[config.dataset_split]
        split_datasets.append(dataset.add_column("subset_name", [subset] * len(dataset)))

    return split_datasets[0] if len(split_datasets) == 1 else concatenate_datasets(split_datasets)


def create_dataset(tokenizer: AutoTokenizer, config: TrainingConfig) -> Dataset:
    source_dataset = load_source_dataset(config)

    rows = source_dataset.map(
        lambda example: prepare_example(
            tokenizer=tokenizer,
            config=config,
            question_text=example["question"],
            answer_text=example["answer"],
            max_length=config.max_length,
        )
    )
    return rows.remove_columns(source_dataset.column_names)
