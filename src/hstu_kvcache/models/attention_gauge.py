from __future__ import annotations

import math
from typing import Any

import torch

from .hstu import HSTU


def _rotation(
    width: int,
    angle: float,
    parity: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = torch.eye(width, device=device, dtype=dtype)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    for start in range(0, width - 1, 2):
        direction = -1.0 if (start // 2 + parity) % 2 else 1.0
        value[start, start] = cosine
        value[start, start + 1] = -direction * sine
        value[start + 1, start] = direction * sine
        value[start + 1, start + 1] = cosine
    return value


@torch.no_grad()
def apply_attention_coordinate_gauge_(
    model: HSTU,
    angle: float,
    key_scale: float = 1.0,
    value_scale: float = 1.0,
) -> dict[str, Any]:
    if not math.isfinite(angle) or not math.isfinite(key_scale) or not math.isfinite(value_scale):
        raise ValueError("attention coordinate gauge values must be finite")
    maximum_orthogonality_error = 0.0
    transformed_tensors = 0
    for layer, block in enumerate(model.blocks):
        attention = block.attn
        heads = attention.num_heads
        width = attention.head_dim
        hidden = attention.q_proj.weight.shape[1]
        q_weight = attention.q_proj.weight.view(heads, width, hidden)
        k_weight = attention.k_proj.weight.view(heads, width, hidden)
        v_weight = attention.v_proj.weight.view(heads, width, hidden)
        out_weight = attention.out_proj.weight.view(hidden, heads, width)
        for head in range(heads):
            parity = layer * heads + head
            key_rotation = _rotation(
                width,
                angle * key_scale,
                parity,
                q_weight.device,
                q_weight.dtype,
            )
            value_rotation = _rotation(
                width,
                angle * value_scale,
                parity + 1,
                v_weight.device,
                v_weight.dtype,
            )
            identity = torch.eye(width, device=q_weight.device, dtype=q_weight.dtype)
            maximum_orthogonality_error = max(
                maximum_orthogonality_error,
                float((key_rotation @ key_rotation.T - identity).abs().max().item()),
                float((value_rotation @ value_rotation.T - identity).abs().max().item()),
            )
            q_weight[head].copy_(key_rotation.T @ q_weight[head])
            k_weight[head].copy_(key_rotation.T @ k_weight[head])
            v_weight[head].copy_(value_rotation.T @ v_weight[head])
            out_weight[:, head, :].copy_(out_weight[:, head, :] @ value_rotation)
            if attention.q_proj.bias is not None:
                q_bias = attention.q_proj.bias.view(heads, width)
                k_bias = attention.k_proj.bias.view(heads, width)
                v_bias = attention.v_proj.bias.view(heads, width)
                q_bias[head].copy_(key_rotation.T @ q_bias[head])
                k_bias[head].copy_(key_rotation.T @ k_bias[head])
                v_bias[head].copy_(value_rotation.T @ v_bias[head])
        transformed_tensors += 4
        if attention.q_proj.bias is not None:
            transformed_tensors += 3
    return {
        "angle": float(angle),
        "key_scale": float(key_scale),
        "value_scale": float(value_scale),
        "layers": len(model.blocks),
        "transformed_tensors": transformed_tensors,
        "maximum_orthogonality_error": maximum_orthogonality_error,
    }


@torch.no_grad()
def apply_attention_coordinate_scale_(
    model: HSTU,
    key_log_scale: float = 0.0,
    value_log_scale: float = 0.0,
) -> dict[str, Any]:
    if not math.isfinite(key_log_scale) or not math.isfinite(value_log_scale):
        raise ValueError("attention coordinate log scales must be finite")
    key_factor = math.exp(key_log_scale)
    value_factor = math.exp(value_log_scale)
    transformed_tensors = 0
    for block in model.blocks:
        attention = block.attn
        attention.q_proj.weight.div_(key_factor)
        attention.k_proj.weight.mul_(key_factor)
        attention.v_proj.weight.mul_(value_factor)
        attention.out_proj.weight.div_(value_factor)
        transformed_tensors += 4
        if attention.q_proj.bias is not None:
            attention.q_proj.bias.div_(key_factor)
            attention.k_proj.bias.mul_(key_factor)
            attention.v_proj.bias.mul_(value_factor)
            transformed_tensors += 3
    return {
        "key_log_scale": float(key_log_scale),
        "value_log_scale": float(value_log_scale),
        "key_factor": key_factor,
        "value_factor": value_factor,
        "layers": len(model.blocks),
        "transformed_tensors": transformed_tensors,
    }
