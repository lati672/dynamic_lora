"""Shared encoder, task heads, and explicit additive stacked-LoRA layers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


def matches_target(path: str, target: str) -> bool:
    """Match a module suffix without conflating RoBERTa's two output.dense paths."""
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
        self.active_weights = None
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def add_adapter(self, name: str) -> None:
        if name in self.adapters:
            raise ValueError(f"Adapter already exists: {name}")
        pair = nn.ModuleDict({"A": nn.Linear(self.base.in_features, self.rank, bias=False),
                              "B": nn.Linear(self.rank, self.base.out_features, bias=False)})
        nn.init.kaiming_uniform_(pair["A"].weight, a=math.sqrt(5))
        nn.init.zeros_(pair["B"].weight)
        pair.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        self.adapters[name] = pair

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base(inputs)
        weights = self.active_weights
        for index, name in enumerate(self.active):
            weight = 1.0 if weights is None else weights[index]
            pair = self.adapters[name]
            output = output + pair["B"](pair["A"](self.dropout(inputs))) * self.scale * weight
        return output

    def effective_weight(self, active: list[str] | None = None) -> torch.Tensor:
        weight = self.base.weight.detach()
        names = self.active if active is None else active
        gate_weights = self.active_weights if active is None else None
        for index, name in enumerate(names):
            factor = 1.0 if gate_weights is None else gate_weights[index]
            pair = self.adapters[name]
            weight = weight + pair["B"].weight @ pair["A"].weight * self.scale * factor
        return weight


class ContinualClassifier(nn.Module):
    def __init__(self, model_name: str, task_labels: dict[str, int]):
        super().__init__()
        self.model_name, self.task_labels = model_name, task_labels
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = AutoConfig.from_pretrained(model_name).hidden_size
        self.heads = nn.ModuleDict({task: nn.Linear(hidden, labels) for task, labels in task_labels.items()})
        self.task_gates = nn.ParameterDict()
        self.task_gate_adapters = {}
        self.active_adapters: list[str] = []

    def forward(self, task: str, **inputs) -> torch.Tensor:
        output = self.encoder(**inputs)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            pooled = output.last_hidden_state[:, -1]
        else:
            last_token = attention_mask.sum(dim=1).sub(1).clamp_min(0)
            batch = torch.arange(output.last_hidden_state.shape[0], device=last_token.device)
            pooled = output.last_hidden_state[batch, last_token]
        head = self.heads[task]
        return head(pooled.to(dtype=head.weight.dtype))

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

    def set_active_adapters(self, names: list[str], weights=None) -> None:
        self.active_adapters = list(names)
        active_weights = weights
        if active_weights is not None and len(active_weights) != len(names):
            raise ValueError("Adapter names and weights must have the same length")
        for module in self.modules():
            if isinstance(module, AdditiveLoRALinear):
                module.active = list(names)
                object.__setattr__(module, "active_weights", active_weights)

    def configure_task_gate(self, task: str, adapters: list[str]) -> None:
        if task in self.task_gates:
            return
        device = next(self.parameters()).device
        initial = torch.zeros(len(adapters), dtype=torch.float32, device=device)
        initial[-1] = 1.0
        self.task_gates[task] = nn.Parameter(initial)
        self.task_gate_adapters[task] = list(adapters)

    def set_task_gate(self, task: str) -> None:
        self.set_active_adapters(self.task_gate_adapters[task], self.task_gates[task])

    def set_trainable(self, task: str, adapter: str | None) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.heads[task].parameters():
            parameter.requires_grad = True
        if task in self.task_gates:
            self.task_gates[task].requires_grad = True
        if adapter:
            for module in self.modules():
                if isinstance(module, AdditiveLoRALinear):
                    for parameter in module.adapters[adapter].parameters():
                        parameter.requires_grad = True
        else:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = True

    def snapshot_adapters_a(self, adapters: list[str]) -> dict[str, torch.Tensor]:
        """Return concatenated, detached LoRA A matrices for prior adapters."""
        if not adapters:
            raise ValueError("At least one prior adapter is required")
        snapshot = {}
        for path, module in self.named_modules():
            if isinstance(module, AdditiveLoRALinear):
                snapshot[path] = torch.cat(
                    [module.adapters[name]["A"].weight.detach().float() for name in adapters], dim=0
                ).clone()
        if not snapshot:
            raise ValueError("Prior adapters have no LoRA A matrices")
        return snapshot

    def orthogonal_penalty(
        self,
        adapter: str,
        previous_a: dict[str, torch.Tensor] | None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Compute an L1 penalty between prior and new stacked-adapter A subspaces."""
        device = next(self.parameters()).device
        penalty = torch.zeros((), dtype=torch.float32, device=device)
        element_count = 0
        if previous_a is None:
            return penalty
        for path, module in self.named_modules():
            if not isinstance(module, AdditiveLoRALinear):
                continue
            if path not in previous_a:
                raise ValueError(f"Previous adapter snapshot is missing module: {path}")
            current = module.adapters[adapter]["A"].weight.float()
            previous = previous_a[path].to(device=current.device, dtype=torch.float32)
            interaction = previous @ current.T
            penalty = penalty + interaction.abs().sum()
            element_count += interaction.numel()
        if normalize and element_count:
            penalty = penalty / element_count
        return penalty

    def orthogonal_penalty_effective(self, adapter: str, previous_adapters: list[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        penalty = torch.zeros((), dtype=torch.float32, device=device)
        comparisons = 0
        eps = 1e-12
        for module in self.modules():
            if not isinstance(module, AdditiveLoRALinear):
                continue
            new_a = module.adapters[adapter]["A"].weight.float()
            new_b = module.adapters[adapter]["B"].weight.float()
            new_norm_sq = ((new_b.T @ new_b) * (new_a @ new_a.T).T).sum().clamp_min(eps)
            for old_name in previous_adapters:
                old_a = module.adapters[old_name]["A"].weight.detach().float()
                old_b = module.adapters[old_name]["B"].weight.detach().float()
                old_norm_sq = ((old_b.T @ old_b) * (old_a @ old_a.T).T).sum().clamp_min(eps)
                cross_b = old_b.T @ new_b
                cross_a = new_a @ old_a.T
                dot = (cross_b * cross_a.T).sum()
                penalty = penalty + dot.square() / (old_norm_sq * new_norm_sq)
                comparisons += 1
        return penalty / comparisons if comparisons else penalty

    def save_checkpoint(self, path: Path, metadata: dict) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    @classmethod
    def load_checkpoint(cls, path: Path, map_location: str = "cpu") -> tuple["ContinualClassifier", dict]:
        metadata = json.loads((path / "metadata.json").read_text())
        model = cls(metadata["model_name"], metadata["task_labels"])
        if metadata["method"] in {"single_lora", "stacked_lora"}:
            model.install_lora(metadata["target_modules"], metadata["lora_rank"], metadata["lora_alpha"], 0.0)
            for adapter in metadata["adapters"]:
                model.add_adapter(adapter)
            for task, adapters in metadata.get("task_gate_adapters", {}).items():
                model.configure_task_gate(task, adapters)
        state = torch.load(path / "model.pt", map_location=map_location, weights_only=True)
        state = {key: value for key, value in state.items() if not key.endswith(".active_weights")}
        model.load_state_dict(state)
        return model, metadata
