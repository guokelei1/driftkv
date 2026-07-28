from __future__ import annotations

from dataclasses import dataclass

import torch

from ..models import HSTU
from .cohort_jagged import JaggedMigratedKVBatch
from .stage45_oldkv import DirectOldKVProgram

RENEWAL_CALIBRATION_MODES = (
    "inverse_norm_ridge",
    "direct_kv_residual_ridge",
)


@dataclass(frozen=True)
class RenewalCalibrationMetrics:
    mode: str
    paired_records: int
    paired_tokens: int
    sampled_tokens: int
    ridge: float
    source_width: int
    target_width: int
    program_fp16_bytes: int
    labels_used: bool = False
    semantic_gate_used: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "paired_records": self.paired_records,
            "paired_tokens": self.paired_tokens,
            "sampled_tokens": self.sampled_tokens,
            "ridge": self.ridge,
            "source_width": self.source_width,
            "target_width": self.target_width,
            "program_fp16_bytes": self.program_fp16_bytes,
            "labels_used": self.labels_used,
            "semantic_gate_used": self.semantic_gate_used,
        }


def _validate_pairs(
    actual_old: JaggedMigratedKVBatch,
    fresh_target: JaggedMigratedKVBatch,
    source_version: str,
    target_version: str,
) -> None:
    if (
        source_version == target_version
        or actual_old.record_ids != fresh_target.record_ids
        or actual_old.k.shape != fresh_target.k.shape
        or actual_old.v.shape != fresh_target.v.shape
        or actual_old.k.device != fresh_target.k.device
        or actual_old.lengths.device != fresh_target.lengths.device
        or not torch.equal(actual_old.lengths, fresh_target.lengths)
        or not torch.equal(actual_old.offsets, fresh_target.offsets)
        or actual_old.served_kv_target != source_version
        or fresh_target.served_kv_target != target_version
    ):
        raise ValueError("renewal calibration pairs differ")


def _projection(
    model: HSTU,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = []
    biases = []
    for block in model.blocks:
        weights.append(
            torch.cat(
                (
                    block.attn.k_proj.weight.detach(),
                    block.attn.v_proj.weight.detach(),
                ),
                dim=0,
            )
            .T.to(device=device, dtype=torch.float32)
        )
        parts = []
        for value in (block.attn.k_proj, block.attn.v_proj):
            if value.bias is None:
                parts.append(
                    torch.zeros(
                        value.out_features,
                        device=device,
                        dtype=torch.float32,
                    )
                )
            else:
                parts.append(
                    value.bias.detach().to(
                        device=device,
                        dtype=torch.float32,
                    )
                )
        biases.append(torch.cat(parts))
    return torch.stack(weights), torch.stack(biases)


def _sample_indices(
    tokens: int,
    max_fit_tokens: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if tokens <= max_fit_tokens:
        return torch.arange(tokens, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(tokens, generator=generator)[:max_fit_tokens].sort().values.to(
        device
    )


def _ridge_correction(
    features: torch.Tensor,
    residuals: torch.Tensor,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_mean = features.mean(dim=1)
    residual_mean = residuals.mean(dim=1)
    centered_features = features - feature_mean[:, None, :]
    centered_residuals = residuals - residual_mean[:, None, :]
    rows = features.shape[1]
    gram = torch.bmm(
        centered_features.transpose(1, 2),
        centered_features,
    ) / rows
    cross = torch.bmm(
        centered_features.transpose(1, 2),
        centered_residuals,
    ) / rows
    scale = gram.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(
        torch.finfo(gram.dtype).eps
    )
    identity = torch.eye(
        gram.shape[1],
        device=gram.device,
        dtype=gram.dtype,
    ).expand(gram.shape[0], -1, -1)
    correction = torch.linalg.solve(
        gram + ridge * scale[:, None, None] * identity,
        cross,
    )
    correction_bias = residual_mean - torch.bmm(
        feature_mean[:, None, :],
        correction,
    )[:, 0]
    return correction, correction_bias


@torch.inference_mode()
def fit_renewal_calibrated_direct_oldkv_program(
    actual_old: JaggedMigratedKVBatch,
    fresh_target: JaggedMigratedKVBatch,
    *,
    source_version: str,
    target_version: str,
    mode: str,
    ridge: float = 1e-3,
    max_fit_tokens: int = 8192,
    seed: int = 0,
    source_model: HSTU | None = None,
    target_model: HSTU | None = None,
) -> tuple[DirectOldKVProgram, RenewalCalibrationMetrics]:
    if (
        mode not in RENEWAL_CALIBRATION_MODES
        or ridge <= 0
        or max_fit_tokens < 2
        or seed < 0
    ):
        raise ValueError("renewal calibration settings differ")
    _validate_pairs(
        actual_old,
        fresh_target,
        source_version,
        target_version,
    )
    tokens = actual_old.token_count
    if tokens < 2:
        raise ValueError("renewal calibration needs at least two tokens")
    indices = _sample_indices(
        tokens,
        max_fit_tokens,
        seed,
        actual_old.k.device,
    )
    actual = torch.cat((actual_old.k, actual_old.v), dim=-1)[
        :, indices
    ].float()
    fresh = torch.cat((fresh_target.k, fresh_target.v), dim=-1)[
        :, indices
    ].float()
    width = actual.shape[-1]
    if mode == "direct_kv_residual_ridge":
        correction, correction_bias = _ridge_correction(
            actual,
            fresh - actual,
            ridge,
        )
        identity = torch.eye(
            width,
            device=actual.device,
            dtype=actual.dtype,
        ).expand(actual.shape[0], -1, -1)
        weights = identity + correction
        biases = correction_bias
        source_width = width
    else:
        if source_model is None or target_model is None:
            raise ValueError("inverse-Norm calibration needs both models")
        source_projection, source_bias = _projection(
            source_model,
            actual.device,
        )
        target_projection, target_bias = _projection(
            target_model,
            actual.device,
        )
        if (
            source_projection.shape[0] != actual.shape[0]
            or source_projection.shape[2] != width
            or target_projection.shape != source_projection.shape
        ):
            raise ValueError("inverse-Norm projection signature differs")
        source_gram = torch.bmm(
            source_projection,
            source_projection.transpose(1, 2),
        )
        inverse_projection = torch.linalg.solve(
            source_gram,
            source_projection,
        ).transpose(1, 2)
        approximate_norm = torch.bmm(
            actual - source_bias[:, None, :],
            inverse_projection,
        )
        cheap = (
            torch.bmm(approximate_norm, target_projection)
            + target_bias[:, None, :]
        )
        correction, correction_bias = _ridge_correction(
            approximate_norm,
            fresh - cheap,
            ridge,
        )
        norm_weights = target_projection + correction
        norm_biases = target_bias + correction_bias
        weights = torch.bmm(inverse_projection, norm_weights)
        biases = norm_biases - torch.bmm(
            source_bias[:, None, :],
            weights,
        )[:, 0]
        source_width = approximate_norm.shape[-1]
    program = DirectOldKVProgram(
        source_version=source_version,
        target_version=target_version,
        weights=weights.to(torch.float16).contiguous(),
        biases=biases.to(torch.float16).contiguous(),
    )
    return program, RenewalCalibrationMetrics(
        mode=mode,
        paired_records=actual_old.batch_size,
        paired_tokens=tokens,
        sampled_tokens=len(indices),
        ridge=ridge,
        source_width=source_width,
        target_width=width,
        program_fp16_bytes=program.nbytes,
    )
