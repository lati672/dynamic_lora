"""Shared encoder, task heads, and explicit additive stacked-LoRA layers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


def matches_target(path: str, target: str) -> bool:
    """Match a RoBERTa module suffix without conflating its two output.dense paths."""
    if target == "output.dense":
        return path.endswith(".output.dense") and not path.endswith(".attention.output.dense")
    return path.endswith(target)


class AdditiveLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.rank, self.scale = rank, alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.adapters = nn.ModuleDict()
        self.active: list[str] = []
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def add_adapter(self, name: str) -> None:
        if name in self.adapters:
            raise ValueError(f"Adapter already exists: {name}")
        pair = nn.ModuleDict({"A": nn.Linear(self.base.in_features, self.rank, bias=False),
                              "B": nn.Linear(self.rank, self.base.out_features, bias=False)})
        nn.init.kaiming_uniform_(pair["A"].weight, a=math.sqrt(5))
        nn.init.zeros_(pair["B"].weight)
        self.adapters[name] = pair

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base(inputs)
        for name in self.active:
            pair = self.adapters[name]
            output = output + pair["B"](pair["A"](self.dropout(inputs))) * self.scale
        return output

    def effective_weight(self, active: list[str] | None = None) -> torch.Tensor:
        weight = self.base.weight.detach()
        for name in self.active if active is None else active:
            pair = self.adapters[name]
            weight = weight + pair["B"].weight @ pair["A"].weight * self.scale
        return weight


class ContinualClassifier(nn.Module):
    def __init__(self, model_name: str, task_labels: dict[str, int]):
        super().__init__()
        self.model_name, self.task_labels = model_name, task_labels
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = AutoConfig.from_pretrained(model_name).hidden_size
        self.heads = nn.ModuleDict({task: nn.Linear(hidden, labels) for task, labels in task_labels.items()})
        self.active_adapters: list[str] = []

    def forward(self, task: str, **inputs) -> torch.Tensor:
        output = self.encoder(**inputs)
        pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            pooled = output.last_hidden_state[:, 0]
        return self.heads[task](pooled)

    def install_lora(self, target_suffixes: list[str], rank: int, alpha: float, dropout: float) -> None:
        replacements = []
        for path, module in self.encoder.named_modules():
            if isinstance(module, nn.Linear) and any(matches_target(path, suffix) for suffix in target_suffixes):
                replacements.append((path, module))
        if not replacements:
            raise ValueError(f"No linear modules matched target suffixes: {target_suffixes}")
        for path, module in replacements:
            parent_path, name = path.rsplit(".", 1)
            parent = self.encoder.get_submodule(parent_path)
            setattr(parent, name, AdditiveLoRALinear(module, rank, alpha, dropout))

    def add_adapter(self, name: str) -> None:
        for module in self.modules():
            if isinstance(module, AdditiveLoRALinear):
                module.add_adapter(name)
        self.set_active_adapters([name])

    def set_active_adapters(self, names: list[str]) -> None:
        self.active_adapters = list(names)
        for module in self.modules():
            if isinstance(module, AdditiveLoRALinear):
                module.active = list(names)

    def set_trainable(self, task: str, adapter: str | None) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.heads[task].parameters():
            parameter.requires_grad = True
        if adapter:
            for module in self.modules():
                if isinstance(module, AdditiveLoRALinear):
                    for parameter in module.adapters[adapter].parameters():
                        parameter.requires_grad = True
        else:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = True

    def save_checkpoint(self, path: Path, metadata: dict) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    @classmethod
    def load_checkpoint(cls, path: Path, map_location: str = "cpu") -> tuple["ContinualClassifier", dict]:
        metadata = json.loads((path / "metadata.json").read_text())
        model = cls(metadata["model_name"], metadata["task_labels"])
        if metadata["method"] == "stacked_lora":
            model.install_lora(metadata["target_modules"], metadata["lora_rank"], metadata["lora_alpha"], 0.0)
            for adapter in metadata["adapters"]:
                model.add_adapter(adapter)
        model.load_state_dict(torch.load(path / "model.pt", map_location=map_location, weights_only=True))
        return model, metadata
