from typing import Literal

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_HF_REPO_ID = "Kt672/dynamic_lora"
DEFAULT_HF_LORA_SUBFOLDER = "artifacts/dynamic_lora/ag_news_yelp_dbpedia/final"
DEFAULT_HF_FULL_SUBFOLDER = "artifacts/dynamic_lora/ag_news_yelp_dbpedia_full_finetune/final"


def _load_kwargs(token: str | None) -> dict:
    if torch.cuda.is_available():
        return {
            "token": token,
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        }
    return {
        "token": token,
        "torch_dtype": torch.float32,
    }


def _prepare_tokenizer(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_hf_continual_model(
    mode: Literal["lora", "full"],
    repo_id: str = DEFAULT_HF_REPO_ID,
    subfolder: str | None = None,
    token: str | None = None,
    merge_lora: bool = False,
):
    """Load a continual-learning checkpoint stored within a Hugging Face repo.

    LoRA checkpoints are expected to have tokenizer files in ``subfolder`` and
    PEFT adapter files in ``subfolder/stack``. Full checkpoints are expected to
    contain both the tokenizer and full model weights directly in ``subfolder``.
    """
    if mode not in {"lora", "full"}:
        raise ValueError(f"Unsupported mode: {mode!r}. Expected 'lora' or 'full'.")

    load_kwargs = _load_kwargs(token)

    if mode == "full":
        model_subfolder = subfolder or DEFAULT_HF_FULL_SUBFOLDER
        tokenizer = _prepare_tokenizer(
            AutoTokenizer.from_pretrained(repo_id, subfolder=model_subfolder, token=token)
        )
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            subfolder=model_subfolder,
            **load_kwargs,
        )
    else:
        tokenizer_subfolder = subfolder or DEFAULT_HF_LORA_SUBFOLDER
        adapter_subfolder = f"{tokenizer_subfolder.rstrip('/')}/stack"
        peft_config = PeftConfig.from_pretrained(
            repo_id,
            subfolder=adapter_subfolder,
            token=token,
        )
        tokenizer = _prepare_tokenizer(
            AutoTokenizer.from_pretrained(repo_id, subfolder=tokenizer_subfolder, token=token)
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **load_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model,
            repo_id,
            subfolder=adapter_subfolder,
            token=token,
            is_trainable=False,
        )
        if merge_lora:
            model = model.merge_and_unload()

    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()
    return tokenizer, model
