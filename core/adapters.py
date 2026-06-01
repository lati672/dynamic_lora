import argparse
from typing import Sequence

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from dynamic_lora.core.constants import STACK_ADAPTER_NAME
from dynamic_lora.core.lora_app.config import TrainingConfig
from dynamic_lora.core.lora_app.modeling import load_base_model


def parse_target_modules(value: str) -> tuple[str, ...]:
    modules = tuple(part.strip() for part in value.split(",") if part.strip())
    if not modules:
        raise ValueError("--target-modules must include at least one module")
    return modules


def lora_config(args: argparse.Namespace, target_modules: Sequence[str]) -> LoraConfig:
    return LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules),
    )


def stacked_lora_config(args: argparse.Namespace, target_modules: Sequence[str], stack_rank: int) -> LoraConfig:
    if stack_rank % args.lora_rank != 0:
        raise ValueError(
            f"stack_rank={stack_rank} must be divisible by the per-task rank={args.lora_rank} "
            "to preserve effective alpha / r scaling."
        )
    stack_alpha = args.lora_alpha * (stack_rank // args.lora_rank)
    return LoraConfig(
        r=stack_rank,
        lora_alpha=stack_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules),
    )


def set_only_adapter_trainable(model: PeftModel, adapter_name: str) -> None:
    marker = f".{adapter_name}."
    for name, parameter in model.named_parameters():
        parameter.requires_grad = ".lora_" in name and marker in name


def set_active_adapters(model: PeftModel, adapter_names: str | list[str], inference_mode: bool = False) -> None:
    names = [adapter_names] if isinstance(adapter_names, str) else adapter_names
    for adapter_name in names:
        if adapter_name not in model.peft_config:
            raise ValueError(f"Adapter {adapter_name} not found.")
    model.base_model.set_adapter(adapter_names, inference_mode=inference_mode)


def extract_adapter_state(model: PeftModel, adapter_name: str) -> dict[str, torch.Tensor]:
    state = {}
    marker = f".{adapter_name}."
    for name, parameter in model.named_parameters():
        if ".lora_A." not in name and ".lora_B." not in name:
            continue
        if marker not in name:
            continue
        state[name.replace(marker, ".<adapter>.")] = parameter.detach().float().cpu().clone()
    if not state:
        raise ValueError(f"No LoRA weights found for adapter {adapter_name}")
    return state


def load_adapter_state(model: PeftModel, adapter_name: str, state: dict[str, torch.Tensor]) -> None:
    remaining = set(state.keys())
    marker = f".{adapter_name}."
    for name, parameter in model.named_parameters():
        if ".lora_A." not in name and ".lora_B." not in name:
            continue
        if marker not in name:
            continue
        key = name.replace(marker, ".<adapter>.")
        if key not in state:
            raise ValueError(f"Missing tensor for {key} while loading adapter {adapter_name}")
        with torch.no_grad():
            parameter.copy_(state[key].to(device=parameter.device, dtype=parameter.dtype))
        remaining.discard(key)
    if remaining:
        unresolved = ", ".join(sorted(remaining))
        raise ValueError(f"Unused adapter tensors while loading {adapter_name}: {unresolved}")


def adapter_rank(state: dict[str, torch.Tensor]) -> int:
    for key, tensor in state.items():
        if ".lora_A." in key:
            return int(tensor.shape[0])
    raise ValueError("Could not infer adapter rank from state")


def concatenate_adapter_states(
    old_state: dict[str, torch.Tensor] | None,
    new_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if old_state is None:
        return {key: value.clone() for key, value in new_state.items()}
    if set(old_state.keys()) != set(new_state.keys()):
        raise ValueError("Old and new adapter states must cover the same tensors")
    merged = {}
    for key in sorted(new_state.keys()):
        old_tensor = old_state[key]
        new_tensor = new_state[key]
        if ".lora_A." in key:
            merged[key] = torch.cat([old_tensor, new_tensor], dim=0)
        elif ".lora_B." in key:
            merged[key] = torch.cat([old_tensor, new_tensor], dim=1)
        else:
            raise ValueError(f"Unexpected adapter tensor key: {key}")
    return merged


def orthogonal_penalty(model: PeftModel, old_adapter_name: str | None, new_adapter_name: str) -> torch.Tensor:
    device = next(model.parameters()).device
    penalty = torch.zeros((), dtype=torch.float32, device=device)
    if old_adapter_name is None:
        return penalty
    old_state = extract_adapter_state(model, old_adapter_name)
    marker = f".{new_adapter_name}."
    for name, parameter in model.named_parameters():
        if ".lora_A." not in name or marker not in name:
            continue
        key = name.replace(marker, ".<adapter>.")
        previous = old_state[key].to(device=device, dtype=torch.float32)
        penalty = penalty + torch.abs(previous @ parameter.float().T).sum()
    return penalty


def orthogonal_penalty_first_task_slices(
    model: PeftModel,
    old_adapter_name: str | None,
    new_adapter_name: str,
    task_rank: int,
    task_count: int,
) -> torch.Tensor:
    device = next(model.parameters()).device
    penalty = torch.zeros((), dtype=torch.float32, device=device)
    if old_adapter_name is None:
        return penalty
    if task_rank <= 0:
        raise ValueError("task_rank must be positive")
    if task_count <= 0:
        raise ValueError("task_count must be positive")

    old_state = extract_adapter_state(model, old_adapter_name)
    marker = f".{new_adapter_name}."
    rows_to_keep = task_rank * task_count
    for name, parameter in model.named_parameters():
        if ".lora_A." not in name or marker not in name:
            continue
        key = name.replace(marker, ".<adapter>.")
        previous = old_state[key].to(device=device, dtype=torch.float32)
        stack_rank = previous.shape[0]
        if rows_to_keep > stack_rank:
            raise ValueError(
                f"Cannot apply orthogonal penalty to {task_count} tasks with rank {task_rank}; "
                f"stacked adapter rank is only {stack_rank}."
            )
        first_previous = previous[:rows_to_keep]
        penalty = penalty + torch.abs(first_previous @ parameter.float().T).sum()
    return penalty


def l2_penalty(model: PeftModel, adapter_name: str) -> torch.Tensor:
    device = next(model.parameters()).device
    penalty = torch.zeros((), dtype=torch.float32, device=device)
    marker = f".{adapter_name}."
    for name, parameter in model.named_parameters():
        if marker not in name:
            continue
        if ".lora_A." not in name and ".lora_B." not in name:
            continue
        penalty = penalty + torch.linalg.norm(parameter.float(), ord=2)
    return penalty


def build_model_for_task(
    config: TrainingConfig,
    token: str | None,
    args: argparse.Namespace,
    target_modules: Sequence[str],
    stacked_state: dict[str, torch.Tensor] | None,
    train_adapter_name: str,
):
    tokenizer, base_model = load_base_model(config, token)
    if stacked_state is None:
        model = get_peft_model(base_model, lora_config(args, target_modules), adapter_name=train_adapter_name)
        active_adapters = [train_adapter_name]
        old_adapter_name = None
    else:
        stack_rank = adapter_rank(stacked_state)
        model = get_peft_model(
            base_model,
            stacked_lora_config(args, target_modules, stack_rank),
            adapter_name=STACK_ADAPTER_NAME,
        )
        load_adapter_state(model, STACK_ADAPTER_NAME, stacked_state)
        model.add_adapter(train_adapter_name, lora_config(args, target_modules))
        active_adapters = [STACK_ADAPTER_NAME, train_adapter_name]
        old_adapter_name = STACK_ADAPTER_NAME
    set_active_adapters(model, active_adapters)
    return tokenizer, model, old_adapter_name, active_adapters


def build_model_from_stacked_state(
    config: TrainingConfig,
    token: str | None,
    args: argparse.Namespace,
    target_modules: Sequence[str],
    stacked_state: dict[str, torch.Tensor],
):
    tokenizer, base_model = load_base_model(config, token)
    stack_rank = adapter_rank(stacked_state)
    model = get_peft_model(
        base_model,
        stacked_lora_config(args, target_modules, stack_rank),
        adapter_name=STACK_ADAPTER_NAME,
    )
    load_adapter_state(model, STACK_ADAPTER_NAME, stacked_state)
    set_active_adapters(model, STACK_ADAPTER_NAME)
    return tokenizer, model
