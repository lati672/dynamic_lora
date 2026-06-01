import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dynamic_lora.core.lora_app.config import TARGET_MODULES, TrainingConfig
from typing import Sequence

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "Missing dependency: peft. Install it with `pip install peft` before running this script."
    ) from exc


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_base_model(
    config: TrainingConfig,
    token: str | None,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    device = resolve_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        token=token,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )

    if device != "cuda":
        model.to(device)

    return tokenizer, model


def add_lora_adapter(
    model: AutoModelForCausalLM,
    rank: int = 8,
    alpha: int = 16,
    target_modules: Sequence[str] | None = None,
) -> PeftModel:
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules or TARGET_MODULES),
    )
    return get_peft_model(model, config)


def load_trained_adapter(
    config: TrainingConfig,
    token: str | None,
) -> tuple[AutoTokenizer, PeftModel]:
    tokenizer, base_model = load_base_model(config, token)
    adapter_dir = config.run_output_dir("lora")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return tokenizer, model
