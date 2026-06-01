import re
from typing import Sequence

import torch
from datasets import Dataset
from peft import PeftModel

from dynamic_lora.core.lora_app.config import TrainingConfig
from dynamic_lora.core.lora_app.data import build_prompt


def normalize_prediction(task_name: str, text: str) -> str:
    cleaned = text.strip().splitlines()[0].strip() if text.strip() else ""
    lowered = cleaned.lower()
    if task_name == "ag_news":
        aliases = {
            "world": "World",
            "sports": "Sports",
            "sport": "Sports",
            "business": "Business",
            "sci/tech": "Sci/Tech",
            "sci-tech": "Sci/Tech",
            "science/technology": "Sci/Tech",
            "science and technology": "Sci/Tech",
            "technology": "Sci/Tech",
            "sci": "Sci/Tech",
        }
    elif task_name == "yelp_review_full":
        aliases = {
            "1": "1 star",
            "1 star": "1 star",
            "one star": "1 star",
            "2": "2 stars",
            "2 stars": "2 stars",
            "two stars": "2 stars",
            "3": "3 stars",
            "3 stars": "3 stars",
            "three stars": "3 stars",
            "4": "4 stars",
            "4 stars": "4 stars",
            "four stars": "4 stars",
            "5": "5 stars",
            "5 stars": "5 stars",
            "five stars": "5 stars",
        }
    elif task_name == "dbpedia_14":
        aliases = {
            "company": "Company",
            "educational institution": "Educational Institution",
            "education institution": "Educational Institution",
            "school": "Educational Institution",
            "artist": "Artist",
            "athlete": "Athlete",
            "office holder": "Office Holder",
            "officeholder": "Office Holder",
            "politician": "Office Holder",
            "mean of transportation": "Mean of Transportation",
            "means of transportation": "Mean of Transportation",
            "transportation": "Mean of Transportation",
            "transport": "Mean of Transportation",
            "vehicle": "Mean of Transportation",
            "building": "Building",
            "natural place": "Natural Place",
            "place": "Natural Place",
            "village": "Village",
            "animal": "Animal",
            "plant": "Plant",
            "album": "Album",
            "film": "Film",
            "movie": "Film",
            "written work": "Written Work",
            "writtenwork": "Written Work",
            "book": "Written Work",
        }
    else:
        raise ValueError(f"Unsupported task for normalization: {task_name}")

    for key, label in aliases.items():
        if lowered == key or re.search(rf"\b{re.escape(key)}\b", lowered):
            return label
    return "unknown"


def generate_label(
    tokenizer,
    model,
    config: TrainingConfig,
    question_text: str,
    max_new_tokens: int,
    task_name: str,
) -> str:
    prediction, _ = generate_label_with_raw(
        tokenizer=tokenizer,
        model=model,
        config=config,
        question_text=question_text,
        max_new_tokens=max_new_tokens,
        task_name=task_name,
    )
    return prediction


def generate_label_with_raw(
    tokenizer,
    model,
    config: TrainingConfig,
    question_text: str,
    max_new_tokens: int,
    task_name: str,
) -> tuple[str, str]:
    prompt = build_prompt(question_text, config)
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][encoded["input_ids"].shape[-1] :]
    decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return normalize_prediction(task_name, decoded), decoded


def evaluate_task(
    tokenizer,
    model: PeftModel,
    config: TrainingConfig,
    task_name: str,
    eval_dataset: Dataset,
    max_new_tokens: int,
    stack_adapter_name: str,
    build_question,
    label_text,
    task_spec,
    set_active_adapters,
) -> dict:
    set_active_adapters(model, stack_adapter_name, inference_mode=True)
    model.eval()
    rows = []
    correct = 0
    for index, example in enumerate(eval_dataset):
        expected = label_text(task_name, int(example["label"]))
        prediction, raw_decoded = generate_label_with_raw(
            tokenizer=tokenizer,
            model=model,
            config=config,
            question_text=build_question(task_name, example[task_spec(task_name)["text_field"]]),
            max_new_tokens=max_new_tokens,
            task_name=task_name,
        )
        is_correct = prediction == expected
        correct += int(is_correct)
        rows.append(
            {
                "index": index,
                "expected": expected,
                "prediction": prediction,
                "raw_decoded": raw_decoded,
                "correct": is_correct,
            }
        )
    accuracy = correct / len(rows) if rows else 0.0
    print(f"[eval] task={task_name} num_examples={len(rows)} mean_accuracy={accuracy:.4f}")
    return {
        "task_name": task_name,
        "num_examples": len(rows),
        "accuracy": accuracy,
        "correct": correct,
        "rows": rows,
    }


def evaluate_task_sequence(
    tokenizer,
    model: PeftModel,
    config: TrainingConfig,
    task_names: Sequence[str],
    eval_datasets: dict[str, Dataset],
    max_new_tokens: int,
    stack_adapter_name: str,
    build_question,
    label_text,
    task_spec,
    set_active_adapters,
) -> list[dict]:
    results = []
    for task_name in task_names:
        results.append(
            evaluate_task(
                tokenizer=tokenizer,
                model=model,
                config=config,
                task_name=task_name,
                eval_dataset=eval_datasets[task_name],
                max_new_tokens=max_new_tokens,
                stack_adapter_name=stack_adapter_name,
                build_question=build_question,
                label_text=label_text,
                task_spec=task_spec,
                set_active_adapters=set_active_adapters,
            )
        )
    return results
