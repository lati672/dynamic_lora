from collections.abc import Iterable

import torch
from peft import PeftModel


def previous_tasks_for_unlearning(task_sequence: tuple[str, ...], unlearn_task: str) -> tuple[str, ...]:
    return task_sequence[: task_sequence.index(unlearn_task)]


def collect_retain_batches(dataloaders: Iterable) -> list:
    batches = []
    for dataloader in dataloaders:
        batches.extend(list(dataloader))
    return batches


def retain_projection_penalty(
    model: PeftModel,
    adapter_name: str,
    retain_batch,
    ridge: float = 1e-6,
) -> torch.Tensor:
    captures: dict[str, torch.Tensor] = {}
    handles = []

    for module_name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
            continue
        lora_a = module.lora_A[adapter_name]

        def capture_input(_, inputs, name=module_name):
            captures[name] = inputs[0].detach()

        handles.append(lora_a.register_forward_pre_hook(capture_input))

    if not handles:
        raise ValueError(f"No LoRA A/B modules found for adapter {adapter_name}")

    device = next(model.parameters()).device
    try:
        model(
            input_ids=retain_batch.input_ids.to(device),
            attention_mask=retain_batch.attention_mask.to(device),
        )
    finally:
        for handle in handles:
            handle.remove()

    penalties = []
    module_lookup = dict(model.named_modules())
    for module_name, retain_x in captures.items():
        module = module_lookup[module_name]
        lora_a_weight = module.lora_A[adapter_name].weight.float()
        lora_b_weight = module.lora_B[adapter_name].weight.float()

        x_flat = retain_x.to(device=device, dtype=torch.float32).reshape(-1, retain_x.shape[-1])
        a_columns = lora_a_weight.T
        gram = a_columns.T @ a_columns
        eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        inverse_gram = torch.linalg.pinv(gram + ridge * eye)
        x_projection = (x_flat @ a_columns @ inverse_gram) @ a_columns.T
        x_residual = x_flat - x_projection
        unlearn_delta = (x_residual @ lora_a_weight.T) @ lora_b_weight.T
        penalties.append(torch.linalg.vector_norm(unlearn_delta, dim=-1).mean())

    if not penalties:
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.stack(penalties).mean()
