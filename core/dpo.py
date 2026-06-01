import copy
import re
from typing import Any

import torch
import torch.nn.functional as F


IGNORE_INDEX = -100


def _select_model_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {"input_ids", "attention_mask", "labels"}
    return {key: value for key, value in batch.items() if key in allowed_keys}


def compute_batch_nll(
    model,
    inputs: dict[str, Any],
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, Any]:
    model_inputs = _select_model_inputs(inputs)
    outputs = model(**model_inputs)

    logits = outputs.logits[..., :-1, :].contiguous()
    labels = model_inputs["labels"][..., 1:].contiguous()

    token_losses = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).view(labels.shape)

    valid_mask = labels.ne(ignore_index)
    token_counts = valid_mask.sum(dim=-1)
    token_loss_sums = token_losses.sum(dim=-1)

    per_sample_nll = torch.zeros_like(token_loss_sums)
    valid_rows = token_counts > 0
    if valid_rows.any():
        per_sample_nll[valid_rows] = (
            token_loss_sums[valid_rows] / token_counts[valid_rows].to(token_loss_sums.dtype)
        )

    return per_sample_nll, outputs


def compute_dpo_loss(model, ref_model, win_inputs=None, lose_inputs=None, beta: float = 1.0):
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs can't be None")
    if beta <= 0:
        raise ValueError("beta must be positive")

    device = next(model.parameters()).device
    win_log_ratio = torch.tensor(0.0, device=device)
    lose_log_ratio = torch.tensor(0.0, device=device)
    win_outputs, lose_outputs = None, None

    if win_inputs is not None:
        win_loss, win_outputs = compute_batch_nll(model, win_inputs)
        with torch.no_grad():
            win_ref_loss, _ = compute_batch_nll(ref_model, win_inputs)
        win_log_ratio = -(win_loss - win_ref_loss)

    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
        with torch.no_grad():
            lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
        lose_log_ratio = -(lose_loss - lose_ref_loss)

    loss = -2.0 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    return loss, (win_outputs, lose_outputs)


def _build_padded_batch(
    tokenizer,
    input_id_rows: list[list[int]],
    label_rows: list[list[int]],
) -> dict[str, torch.Tensor]:
    if not input_id_rows:
        raise ValueError("Cannot build a batch from empty rows")

    max_length = max(len(row) for row in input_id_rows)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id")

    input_ids = []
    attention_mask = []
    labels = []
    for input_row, label_row in zip(input_id_rows, label_rows, strict=True):
        pad_length = max_length - len(input_row)
        input_ids.append(input_row + [pad_token_id] * pad_length)
        attention_mask.append([1] * len(input_row) + [0] * pad_length)
        labels.append(label_row + [IGNORE_INDEX] * pad_length)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _normalize_label_text(text: str) -> str:
    normalized = " ".join(text.strip().lower().split())
    # Label spans can inherit prompt punctuation at the token boundary
    # (for example ":Educational Institution" from "Category:" + label).
    # Strip only boundary punctuation so internal label characters remain intact.
    normalized = re.sub(r"^[^a-z0-9]+", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+$", "", normalized)
    return normalized


def _extract_prompt_and_true_label(
    tokenizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    candidate_labels: tuple[str, ...],
) -> tuple[str, str] | None:
    valid_positions = labels.ne(IGNORE_INDEX).nonzero(as_tuple=False).flatten()
    if valid_positions.numel() == 0:
        return None

    prompt_end = int(valid_positions[0].item())
    prompt_ids = input_ids[:prompt_end]
    valid_token_ids = labels[labels.ne(IGNORE_INDEX)]

    prompt_text = tokenizer.decode(
        prompt_ids.tolist(),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    decoded_label = tokenizer.decode(
        valid_token_ids.tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    normalized_decoded = _normalize_label_text(decoded_label)
    if not normalized_decoded:
        return None

    for candidate in candidate_labels:
        normalized_candidate = _normalize_label_text(candidate)
        if (
            normalized_decoded == normalized_candidate
            or normalized_decoded.startswith(normalized_candidate)
        ):
            return prompt_text, candidate

    # Robust fallback for truncated numeric labels (e.g., "2", "2.", "\n\n2")
    # when candidates are text labels like "2 stars".
    numeric_match = re.search(r"\b([1-9][0-9]*)\b", normalized_decoded)
    if numeric_match is not None:
        numeric_token = numeric_match.group(1)
        matching_candidates = [
            candidate
            for candidate in candidate_labels
            if _normalize_label_text(candidate).startswith(f"{numeric_token} ")
            or _normalize_label_text(candidate) == numeric_token
        ]
        if len(matching_candidates) == 1:
            return prompt_text, matching_candidates[0]

    observed = attention_mask.sum().item()
    raise ValueError(
        "Could not map decoded label "
        f"{decoded_label!r} to any candidate label for a sequence of length {observed}."
    )


def _encode_prompt_with_answer(
    tokenizer,
    prompt_text: str,
    answer_text: str,
    max_length: int,
    assistant_end_tag: str | None = None,
) -> tuple[list[int], list[int]]:
    full_text = prompt_text + answer_text
    if assistant_end_tag:
        full_text += assistant_end_tag
    else:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer must define eos_token when no assistant_end_tag is provided")
        full_text += tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    input_ids = tokenized["input_ids"]
    labels = input_ids[:]
    prompt_length = min(len(prompt_ids), len(labels))
    for index in range(prompt_length):
        labels[index] = IGNORE_INDEX

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is not None:
        labels = [token if token != pad_token_id else IGNORE_INDEX for token in labels]
    return input_ids, labels


def build_classification_dpo_inputs(
    model,
    tokenizer,
    batch,
    candidate_labels: tuple[str, ...],
    max_length: int,
    assistant_end_tag: str | None = None,
) -> dict[str, dict[str, torch.Tensor]] | None:
    prompt_texts: list[str] = []
    true_labels: list[str] = []
    for input_ids, labels, attention_mask in zip(
        batch.input_ids,
        batch.labels,
        batch.attention_mask,
        strict=True,
    ):
        extracted = _extract_prompt_and_true_label(
            tokenizer=tokenizer,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            candidate_labels=candidate_labels,
        )
        if extracted is None:
            continue
        prompt_text, true_label = extracted
        prompt_texts.append(prompt_text)
        true_labels.append(true_label)

    if not prompt_texts:
        return None

    device = next(model.parameters()).device
    candidate_losses: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for candidate in candidate_labels:
            input_rows = []
            label_rows = []
            for prompt_text in prompt_texts:
                input_row, label_row = _encode_prompt_with_answer(
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    answer_text=candidate,
                    max_length=max_length,
                    assistant_end_tag=assistant_end_tag,
                )
                input_rows.append(input_row)
                label_rows.append(label_row)
            candidate_batch = _build_padded_batch(tokenizer, input_rows, label_rows)
            candidate_batch = {
                key: value.to(device)
                for key, value in candidate_batch.items()
            }
            candidate_loss, _ = compute_batch_nll(model, candidate_batch)
            candidate_losses[candidate] = candidate_loss.detach().cpu()

    win_rows = []
    win_label_rows = []
    lose_rows = []
    lose_label_rows = []
    batch_size = batch.input_ids.shape[0]
    for index in range(len(prompt_texts)):
        ranked_candidates = sorted(
            candidate_labels,
            key=lambda candidate: float(candidate_losses[candidate][index].item()),
        )
        win_label = next(
            candidate for candidate in ranked_candidates if candidate != true_labels[index]
        )
        lose_label = true_labels[index]

        prompt_text = prompt_texts[index]
        win_row, win_label_row = _encode_prompt_with_answer(
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            answer_text=win_label,
            max_length=max_length,
            assistant_end_tag=assistant_end_tag,
        )
        lose_row, lose_label_row = _encode_prompt_with_answer(
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            answer_text=lose_label,
            max_length=max_length,
            assistant_end_tag=assistant_end_tag,
        )
        win_rows.append(win_row)
        win_label_rows.append(win_label_row)
        lose_rows.append(lose_row)
        lose_label_rows.append(lose_label_row)

    return {
        "forget": {
            "alternate": _build_padded_batch(tokenizer, win_rows, win_label_rows),
            "original": _build_padded_batch(tokenizer, lose_rows, lose_label_rows),
        }
    }


def build_yelp_dpo_inputs(
    model,
    tokenizer,
    batch,
    candidate_labels: tuple[str, ...],
    max_length: int,
    assistant_end_tag: str | None = None,
) -> dict[str, dict[str, torch.Tensor]] | None:
    return build_classification_dpo_inputs(
        model=model,
        tokenizer=tokenizer,
        batch=batch,
        candidate_labels=candidate_labels,
        max_length=max_length,
        assistant_end_tag=assistant_end_tag,
    )


class GradDiff:
    def __init__(self, model, ref_model=None, gamma: float = 1.0):
        self.model = model
        self.ref_model = ref_model
        self.gamma = gamma

    def _prepare_ref_model(self, model):
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        return ref_model


class DPO(GradDiff):
    def __init__(self, beta: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        alternate_inputs = inputs["forget"]["alternate"]
        original_inputs = inputs["forget"]["original"]

        forget_loss, forget_outputs = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=alternate_inputs,
            lose_inputs=original_inputs,
            beta=self.beta,
        )

        loss = self.gamma * forget_loss
        return (loss, forget_outputs) if return_outputs else loss
