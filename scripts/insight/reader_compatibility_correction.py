"""Stage-wise signed HSTU reader compatibility-correction diagnostics.

The corrections in this module are oracle observations derived from coherent
Current-Exact and Parent-Reuse reader traces over the same candidate bank.
They are not persistent states and are never fitted to target K/V tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

if __package__:  # Package import from the rolling evaluator.
    from .candidate_shared_causal import _block_update, _cached_prefix_heads
else:  # Direct script/test import with scripts/insight on sys.path.
    from candidate_shared_causal import _block_update, _cached_prefix_heads
from hstu_kvcache.models import HSTUKVCache


LAYER_STAGES = (
    "kv_prefix_contribution",
    "av_aggregation",
    "u_gated_update",
    "layer_hidden",
)
STAGES = (*LAYER_STAGES, "final_readout")


@dataclass(frozen=True)
class ReaderCorrectionTrace:
    exact_scores: torch.Tensor
    reuse_scores: torch.Tensor
    exact_readout: torch.Tensor
    reuse_readout: torch.Tensor
    stage_scores: dict[str, torch.Tensor]
    corrections: dict[str, tuple[torch.Tensor, ...]]
    energy_metrics: tuple[dict[str, torch.Tensor | str | int], ...]
    correctness: dict[str, float]


def _candidate_split(delta: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Split signed ``[B,C,...]`` delta into candidate mean and residual."""
    if delta.ndim < 3 or delta.shape[1] < 1:
        raise ValueError("stage delta must have shape [B,C,...]")
    delta = delta.float()
    shared = delta.mean(dim=1, keepdim=True)
    residual = delta - shared
    reduce = tuple(range(1, delta.ndim))
    total = delta.square().sum(dim=reduce)
    shared_energy = shared.expand_as(delta).square().sum(dim=reduce)
    residual_energy = residual.square().sum(dim=reduce)
    cross = (shared.expand_as(delta) * residual).sum(dim=reduce)
    return shared[:, 0], {
        "total_energy": total,
        "shared_energy": shared_energy,
        "residual_energy": residual_energy,
        "shared_energy_fraction": shared_energy / total.clamp_min(1e-20),
        "residual_energy_fraction": residual_energy / total.clamp_min(1e-20),
        "orthogonality_error": cross.abs() / total.clamp_min(1e-20),
    }


def _prefix_contributions(
    attention,
    q: torch.Tensor,
    cache: HSTUKVCache,
    layer: int,
    candidate_count: int,
) -> torch.Tensor:
    """Return signed activated-qK-times-V before the history-position sum."""
    flat = q.shape[0]
    if flat % candidate_count:
        raise ValueError("flattened query batch is not divisible by candidate count")
    batch = flat // candidate_count
    length = cache.seq_len
    cached_k = cache.k[layer].repeat_interleave(candidate_count, dim=0)
    cached_v = cache.v[layer].repeat_interleave(candidate_count, dim=0)
    cached_k = cached_k.view(
        flat, length, attention.num_heads, attention.head_dim
    ).transpose(1, 2)
    cached_v = cached_v.view(
        flat, length, attention.num_heads, attention.head_dim
    ).transpose(1, 2)
    weights = attention._activate(
        torch.matmul(q, cached_k.transpose(-2, -1)) * attention.scale
    )
    if attention.block_variant == "hstu_reference":
        weights = weights / attention.cfg.max_seq_len
    weights = attention.attn_dropout(weights).squeeze(2)
    contribution = weights.unsqueeze(-1) * cached_v
    return contribution.reshape(
        batch,
        candidate_count,
        attention.num_heads,
        length,
        attention.head_dim,
    )


def _self_heads(block, q: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor) -> torch.Tensor:
    if block.attn.causal_diagonal != "inclusive":
        return torch.zeros_like(v_new)
    weight = block.attn._activate(
        (q * k_new).sum(dim=-1, keepdim=True) * block.attn.scale
    )
    if block.attn.block_variant == "hstu_reference":
        weight = weight / block.attn.cfg.max_seq_len
    return weight * v_new


def _reshape_stage(value: torch.Tensor, batch: int, candidates: int) -> torch.Tensor:
    return value.reshape(batch, candidates, *value.shape[1:])


@torch.inference_mode()
def _stage_path(
    model,
    exact_cache: HSTUKVCache,
    reuse_cache: HSTUKVCache,
    candidate_ids: torch.Tensor,
    query_time_deltas: torch.Tensor,
    *,
    stage: str,
    mode: str,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[dict[str, torch.Tensor | str | int], ...]]:
    """Run one dynamic stage intervention at a common hidden state.

    At each layer, Exact and Reuse stage tensors are evaluated from the same
    current hidden state.  The chosen path then advances with either the signed
    candidate-mean correction (``shared``) or the complete Exact tensor
    (``full``).  This avoids applying an upstream error twice at later layers.
    """
    if stage not in LAYER_STAGES:
        raise ValueError(f"stage path requires a layered stage, got {stage}")
    if mode not in {"shared", "full"}:
        raise ValueError(f"unknown stage-path mode: {mode}")
    batch, candidates = candidate_ids.shape
    x = model.embed_query_tokens(candidate_ids, query_time_deltas).reshape(
        batch * candidates, 1, model.cfg.hidden_size
    )
    corrections: list[torch.Tensor] = []
    metrics: list[dict[str, torch.Tensor | str | int]] = []

    for layer, block in enumerate(model.blocks):
        residual_x = x
        x_norm = block.norm(x)
        q, k_new, v_new = block.attn._project(x_norm)
        exact_prefix = _prefix_contributions(
            block.attn, q, exact_cache, layer, candidates
        )
        reuse_prefix = _prefix_contributions(
            block.attn, q, reuse_cache, layer, candidates
        )
        exact_prefix_heads = exact_prefix.sum(dim=3)
        reuse_prefix_heads = reuse_prefix.sum(dim=3)
        self_heads = _self_heads(block, q, k_new, v_new).squeeze(2).reshape(
            batch, candidates, block.attn.num_heads, block.attn.head_dim
        )
        exact_av = exact_prefix_heads + self_heads
        reuse_av = reuse_prefix_heads + self_heads
        exact_update = _reshape_stage(
            _block_update(
                block,
                x_norm,
                exact_av.reshape(
                    batch * candidates, block.attn.num_heads, 1, block.attn.head_dim
                ),
            ).squeeze(1),
            batch,
            candidates,
        )
        reuse_update = _reshape_stage(
            _block_update(
                block,
                x_norm,
                reuse_av.reshape(
                    batch * candidates, block.attn.num_heads, 1, block.attn.head_dim
                ),
            ).squeeze(1),
            batch,
            candidates,
        )
        residual = _reshape_stage(residual_x.squeeze(1), batch, candidates)
        exact_hidden = residual + exact_update
        reuse_hidden = residual + reuse_update
        exact_stage, reuse_stage = {
            "kv_prefix_contribution": (exact_prefix, reuse_prefix),
            "av_aggregation": (exact_av, reuse_av),
            "u_gated_update": (exact_update, reuse_update),
            "layer_hidden": (exact_hidden, reuse_hidden),
        }[stage]
        shared, stage_metrics = _candidate_split(exact_stage - reuse_stage)
        if stage == "kv_prefix_contribution":
            # The fixed intervention enters immediately after the history sum;
            # keeping the pre-sum tensor for energy avoids changing the stage
            # definition while keeping the correction compact.
            shared = shared.sum(dim=2)
        shared = shared.to(exact_stage.dtype)
        corrections.append(shared)
        metrics.append({"stage": stage, "layer": layer, **stage_metrics})

        if stage == "kv_prefix_contribution":
            selected_heads = (
                exact_prefix_heads
                if mode == "full"
                else reuse_prefix_heads + shared[:, None]
            )
            selected_av = selected_heads + self_heads
            selected_update = _reshape_stage(
                _block_update(
                    block,
                    x_norm,
                    selected_av.reshape(
                        batch * candidates,
                        block.attn.num_heads,
                        1,
                        block.attn.head_dim,
                    ),
                ).squeeze(1),
                batch,
                candidates,
            )
            selected_hidden = residual + selected_update
        elif stage == "av_aggregation":
            selected_av = exact_av if mode == "full" else reuse_av + shared[:, None]
            selected_update = _reshape_stage(
                _block_update(
                    block,
                    x_norm,
                    selected_av.reshape(
                        batch * candidates,
                        block.attn.num_heads,
                        1,
                        block.attn.head_dim,
                    ),
                ).squeeze(1),
                batch,
                candidates,
            )
            selected_hidden = residual + selected_update
        elif stage == "u_gated_update":
            selected_update = exact_update if mode == "full" else reuse_update + shared[:, None]
            selected_hidden = residual + selected_update
        else:
            selected_hidden = exact_hidden if mode == "full" else reuse_hidden + shared[:, None]
        x = selected_hidden.reshape(batch * candidates, 1, model.cfg.hidden_size)

    readout = model.final_norm(x).reshape(batch, candidates, model.cfg.hidden_size)
    return readout, tuple(corrections), tuple(metrics)


@torch.inference_mode()
def intervene_reader_correction(
    model,
    reuse_cache: HSTUKVCache,
    candidate_ids: torch.Tensor,
    query_time_deltas: torch.Tensor,
    *,
    stage: str,
    corrections: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inject one frozen broadcast correction into a coherent Reuse reader."""
    if stage not in STAGES:
        raise ValueError(f"unknown reader-correction stage: {stage}")
    batch, candidates = candidate_ids.shape
    expected = 1 if stage == "final_readout" else len(model.blocks)
    if len(corrections) != expected:
        raise ValueError("correction layer count differs")
    x = model.embed_query_tokens(candidate_ids, query_time_deltas).reshape(
        batch * candidates, 1, model.cfg.hidden_size
    )
    for layer, block in enumerate(model.blocks):
        residual_x = x
        x_norm = block.norm(x)
        q, k_new, v_new = block.attn._project(x_norm)
        prefix_heads = _cached_prefix_heads(
            block.attn, q, reuse_cache, layer, candidates
        ).reshape(batch * candidates, block.attn.num_heads, 1, block.attn.head_dim)
        if stage == "kv_prefix_contribution":
            addition = corrections[layer].to(prefix_heads.device, prefix_heads.dtype)
            prefix_heads = prefix_heads + addition[:, None].expand(
                batch, candidates, *addition.shape[1:]
            ).reshape_as(prefix_heads)
        av_heads = prefix_heads + _self_heads(block, q, k_new, v_new)
        if stage == "av_aggregation":
            addition = corrections[layer].to(av_heads.device, av_heads.dtype)
            av_heads = av_heads + addition[:, None].expand(
                batch, candidates, *addition.shape[1:]
            ).reshape_as(av_heads)
        update = _block_update(block, x_norm, av_heads)
        if stage == "u_gated_update":
            addition = corrections[layer].to(update.device, update.dtype)
            update = update + addition[:, None].expand(
                batch, candidates, addition.shape[-1]
            ).reshape(batch * candidates, 1, addition.shape[-1])
        x = residual_x + update
        if stage == "layer_hidden":
            addition = corrections[layer].to(x.device, x.dtype)
            x = x + addition[:, None].expand(
                batch, candidates, addition.shape[-1]
            ).reshape(batch * candidates, 1, addition.shape[-1])
    readout = model.final_norm(x).reshape(batch, candidates, model.cfg.hidden_size)
    if stage == "final_readout":
        addition = corrections[0].to(readout.device, readout.dtype)
        readout = readout + addition[:, None]
    return model.cc_score_head(readout).squeeze(-1), readout


@torch.inference_mode()
def trace_reader_correction(
    model,
    exact_cache: HSTUKVCache,
    reuse_cache: HSTUKVCache,
    candidate_ids: torch.Tensor,
    query_time_deltas: torch.Tensor,
    *,
    verify_full_delta: bool = True,
) -> ReaderCorrectionTrace:
    """Trace coherent paths and execute every same-request shared correction."""
    if exact_cache.seq_len != reuse_cache.seq_len:
        raise ValueError("Exact and Reuse cache lengths differ")
    native_exact, exact_readout = model.observe_cc_reuse(
        exact_cache, candidate_ids, query_time_deltas
    )
    native_reuse, reuse_readout = model.observe_cc_reuse(
        reuse_cache, candidate_ids, query_time_deltas
    )
    exact_scores = model.cc_score_head(exact_readout).squeeze(-1)
    reuse_scores = model.cc_score_head(reuse_readout).squeeze(-1)
    corrections: dict[str, tuple[torch.Tensor, ...]] = {}
    stage_scores: dict[str, torch.Tensor] = {}
    metrics: list[dict[str, torch.Tensor | str | int]] = []
    full_errors: list[torch.Tensor] = []
    for stage in LAYER_STAGES:
        shared_readout, shared_corrections, stage_metrics = _stage_path(
            model,
            exact_cache,
            reuse_cache,
            candidate_ids,
            query_time_deltas,
            stage=stage,
            mode="shared",
        )
        corrections[stage] = shared_corrections
        stage_scores[stage] = model.cc_score_head(shared_readout).squeeze(-1)
        metrics.extend(stage_metrics)
        if verify_full_delta:
            full_readout, _, _ = _stage_path(
                model,
                exact_cache,
                reuse_cache,
                candidate_ids,
                query_time_deltas,
                stage=stage,
                mode="full",
            )
            full_errors.append(torch.max(torch.abs(full_readout - exact_readout)))
    shared_final, final_metrics = _candidate_split(exact_readout - reuse_readout)
    shared_final = shared_final.to(exact_readout.dtype)
    corrections["final_readout"] = (shared_final,)
    metrics.append({"stage": "final_readout", "layer": -1, **final_metrics})
    stage_scores["final_readout"] = model.cc_score_head(
        reuse_readout + shared_final[:, None]
    ).squeeze(-1)
    full_readout = reuse_readout + (exact_readout - reuse_readout)
    full_scores = model.cc_score_head(full_readout).squeeze(-1)
    return ReaderCorrectionTrace(
        exact_scores=exact_scores,
        reuse_scores=reuse_scores,
        exact_readout=exact_readout,
        reuse_readout=reuse_readout,
        stage_scores=stage_scores,
        corrections=corrections,
        energy_metrics=tuple(metrics),
        correctness={
            "native_exact": float(torch.max(torch.abs(native_exact - exact_scores))),
            "native_reuse": float(torch.max(torch.abs(native_reuse - reuse_scores))),
            "final_full_delta": float(torch.max(torch.abs(native_exact - full_scores))),
            "layer_stage_full_delta": (
                float(torch.stack(full_errors).max()) if full_errors else float("nan")
            ),
        },
    )


def correction_cosine(
    current: tuple[torch.Tensor, ...],
    previous: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    if len(current) != len(previous):
        raise ValueError("correction layer counts differ")
    current_flat = torch.cat([value.float().flatten(1) for value in current], dim=1)
    previous_flat = torch.cat([value.float().flatten(1) for value in previous], dim=1)
    return torch.nn.functional.cosine_similarity(current_flat, previous_flat, dim=1)


def correction_norm(correction: tuple[torch.Tensor, ...]) -> torch.Tensor:
    flat = torch.cat([value.float().flatten(1) for value in correction], dim=1)
    return flat.norm(dim=1)


def scale_correction(
    correction: tuple[torch.Tensor, ...], factors: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    return tuple(
        value * factors.to(value.device, value.dtype).view(
            value.shape[0], *([1] * (value.ndim - 1))
        )
        for value in correction
    )
