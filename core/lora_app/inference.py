import json
import os

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from dynamic_lora.core.lora_app.config import TEST_QUESTION, TrainingConfig
from dynamic_lora.core.lora_app.data import build_prompt, load_source_dataset
from dynamic_lora.core.lora_app.modeling import load_base_model, resolve_device

try:
    from peft import PeftModel
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "Missing dependency: peft. Install it with `pip install peft` before running this script."
    ) from exc


def load_full_model(
    config: TrainingConfig,
    token: str | None,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    device = resolve_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model_dir = config.run_output_dir("full")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        token=token,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )

    if device != "cuda":
        model.to(device)

    model.eval()
    return tokenizer, model


def generate_answer(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM | PeftModel,
    config: TrainingConfig,
    question_text: str,
) -> str:
    prompt = build_prompt(question_text, config)
    device = next(model.parameters()).device

    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][encoded["input_ids"].shape[-1] :]
    decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if config.apply_chat_template and config.asst_end_tag in decoded:
        decoded = decoded.split(config.asst_end_tag, 1)[0].strip()
    first_line = decoded.splitlines()[0].strip()
    cleaned = first_line.split("Note:", 1)[0].strip()
    return cleaned or first_line


def translate_text(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM | PeftModel,
    english_text: str,
) -> str:
    return generate_answer(
        tokenizer=tokenizer,
        model=model,
        config=TrainingConfig(),
        question_text=english_text,
    )


def sample_training_examples(
    config: TrainingConfig,
    limit: int = 20,
    seed: int = 42,
) -> Dataset:
    dataset = load_source_dataset(config)
    if len(dataset) <= limit:
        return dataset
    return dataset.shuffle(seed=seed).select(range(limit))


def collect_dual_model_inference(
    config: TrainingConfig,
    sample_size: int = 20,
    seed: int = 42,
) -> list[dict[str, str | int]]:
    token = os.environ.get("HF_TOKEN")
    full_tokenizer, full_model = load_full_model(config, token)
    lora_tokenizer, lora_model = load_base_model(config, token)
    lora_model = PeftModel.from_pretrained(lora_model, config.run_output_dir("lora"))
    lora_model.eval()
    examples = sample_training_examples(config, limit=sample_size, seed=seed)

    rows = []
    for index, example in enumerate(examples, start=1):
        rows.append(
            {
                "index": index,
                "subset": example.get("subset_name", config.dataset_subset),
                "question": example["question"],
                "ground_truth": example["answer"],
                "full_prediction": generate_answer(
                    full_tokenizer,
                    full_model,
                    config,
                    example["question"],
                ),
                "lora_prediction": generate_answer(
                    lora_tokenizer,
                    lora_model,
                    config,
                    example["question"],
                ),
            }
        )
    return rows


def format_dual_model_inference(
    config: TrainingConfig,
    rows: list[dict[str, str | int]],
) -> str:
    lines = [
        f"full_outputs= {config.run_output_dir('full')}",
        f"lora_outputs= {config.run_output_dir('lora')}",
        f"samples={len(rows)}",
        "-" * 80,
    ]

    for row in rows:
        lines.append(f"[{row['index']}] subset={row['subset']}")
        lines.append(f"Q: {row['question']}")
        lines.append(f"GT: {row['ground_truth']}")
        lines.append(f"FULL: {row['full_prediction']}")
        lines.append(f"LORA: {row['lora_prediction']}")
        lines.append("-" * 80)

    return "\n".join(lines)


def save_dual_model_inference(
    config: TrainingConfig,
    rows: list[dict[str, str | int]],
    txt_path: str,
    json_path: str,
) -> None:
    with open(txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(format_dual_model_inference(config, rows) + "\n")

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(
            {
                "model_id": config.model_id,
                "dataset_id": config.dataset_id,
                "dataset_subset": config.dataset_subset,
                "dataset_split": config.dataset_split,
                "full_output_dir": config.run_output_dir("full"),
                "lora_output_dir": config.run_output_dir("lora"),
                "rows": rows,
            },
            json_file,
            indent=2,
            ensure_ascii=False,
        )


def run_dual_model_inference(
    config: TrainingConfig,
    sample_size: int = 20,
    seed: int = 42,
) -> None:
    rows = collect_dual_model_inference(config=config, sample_size=sample_size, seed=seed)
    print(format_dual_model_inference(config, rows))


def run_base_inference() -> None:
    config = TrainingConfig()
    tokenizer, model = load_base_model(config, os.environ.get("HF_TOKEN"))
    model.eval()
    examples = sample_training_examples(config, limit=5)

    print(f"Model: {config.model_id}")
    print(f"Dataset: {config.dataset_id} [{config.dataset_subset}/{config.dataset_split}]")
    print(f"Running inference on {len(examples)} examples")
    print("-" * 80)

    for index, example in enumerate(examples, start=1):
        prediction = generate_answer(
            tokenizer=tokenizer,
            model=model,
            config=config,
            question_text=example["question"],
        )
        print(f"[{index}] Q: {example['question']}")
        print(f"    GT: {example['answer']}")
        print(f"    PR: {prediction}")
        print("-" * 80)

    print("single_test_question=", TEST_QUESTION)
    print(
        "single_test_prediction=",
        generate_answer(tokenizer=tokenizer, model=model, config=config, question_text=TEST_QUESTION),
    )


if __name__ == "__main__":
    run_base_inference()
