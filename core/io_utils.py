import gc
import json
from pathlib import Path

import torch
from peft import PeftModel


def save_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_lora_ab(model: PeftModel, output_dir: Path) -> None:
    lora_ab = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }
    torch.save(lora_ab, output_dir / "lora_ab.pt")
