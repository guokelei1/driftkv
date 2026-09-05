#!/usr/bin/env python3
"""Run the frozen Medium attention-cone response-moment diagnostic.

The runner is raw-first and four-rank only.  It never fits a mapper or reads
labels.  Full and sampled Current moments are oracle mechanisms; every Parent
moment and every serving control response uses the complete Parent cache.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "insight"))

from evaluate_yambda500m_foundation_raw import load_histories, load_model  # noqa: E402
from insight_one_locality.common import histories_at_cutover  # noqa: E402
from insight_two.address_response_memory import select_address_landmarks  # noqa: E402
from insight_two.attention_cone_moments import (  # noqa: E402
    decompose_elu_plus_one_response,
    scaled_qk_logits,
)
from insight_two.common import (  # noqa: E402
    ANCHOR_INDICES,
    CUTOVER_DAYS,
    DATASET,
    DAY,
    EDGES,
    HELDOUT_INDICES,
    HISTORY,
    KNOWN_ITEMS,
    OOV_BUCKETS,
    RESULT_ROOT,
    checkpoint,
    load_frozen_inputs,
    metrics_row,
    score_metrics,
    sha256_file,
    verify_contract as verify_boundary_contract,
    verify_model_payload,
)
from insight_two.cone_response_memory import (  # noqa: E402
    REQUIRED_ANCHOR_COUNT,
    ConeResponseMemory,
    build_layer_signed_cone_moment,
    intervene_cone_response_memory,
)
from insight_two.signed_response_memory import (  # noqa: E402
    SUPPORTED_SAMPLE_COUNTS,
    fixed_midpoint_strata,
)


CONTRACT = (
    ROOT
    / "configs/contracts/"
    "yambda500m_medium_hstu_native_insight2_attention_cone_moments_v1.yaml"
)
OUTPUT_ROOT = RESULT_ROOT / "diagnostic_attention_cone_moments_v1"
RESOURCE_ESTIMATE = OUTPUT_ROOT / "resource_estimate.json"
CANARY_USERS = 32
DISCOVERY_USERS = 512
EXPECTED_LAYERS = 6
EXPECTED_HEADS = 6
EXPECTED_HIDDEN = 192
EXPECTED_HEAD_DIM = 32
EXPECTED_MOMENT_SCALARS = 38_016
EXPECTED_MOMENT_RATIO = 0.01611328125
METHODS_PER_USER_EDGE = 2 + 2 * len(SUPPORTED_SAMPLE_COUNTS)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def verify_contract() -> tuple[dict[str, Any], str]:
    verify_boundary_contract()
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    scope = contract["scope"]
    if scope["edges"] != list(EDGES) or scope["cutover_days"] != list(CUTOVER_DAYS):
        raise RuntimeError("attention-cone edge scope differs")
    if int(scope["history_positions"]) != HISTORY:
        raise RuntimeError("attention-cone history length differs")
    split = contract["candidate_split"]
    if tuple(split["anchor_indices"]) != ANCHOR_INDICES:
        raise RuntimeError("attention-cone anchor split differs")
    if tuple(split["heldout_indices"]) != HELDOUT_INDICES:
        raise RuntimeError("attention-cone held-out split differs")
    if int(split["anchor_count"]) != REQUIRED_ANCHOR_COUNT:
        raise RuntimeError("attention-cone anchor count differs")
    oracles = contract["current_moment_construction_oracles"]
    if tuple(oracles["chronological"]["sample_counts"]) != SUPPORTED_SAMPLE_COUNTS:
        raise RuntimeError("chronological moment grid differs")
    if tuple(oracles["address"]["sample_counts"]) != SUPPORTED_SAMPLE_COUNTS:
        raise RuntimeError("address moment grid differs")
    state = contract["moment_state"]
    if int(state["floating_scalars_per_user"]) != EXPECTED_MOMENT_SCALARS:
        raise RuntimeError("attention-cone moment scalar count differs")
    if float(state["ratio_to_full_Current_KV"]) != EXPECTED_MOMENT_RATIO:
        raise RuntimeError("attention-cone persistent ratio differs")
    population = contract["population"]
    if population["canary_indices_half_open"] != [0, CANARY_USERS]:
        raise RuntimeError("attention-cone canary population differs")
    if population["discovery_indices_half_open"] != [0, DISCOVERY_USERS]:
        raise RuntimeError("attention-cone discovery population differs")
    execution = contract["execution"]
    if execution["GPUs"] != [0, 1, 2, 3]:
        raise RuntimeError("attention-cone GPU allowlist differs")
    if int(execution["canary_users"]) != CANARY_USERS:
        raise RuntimeError("attention-cone canary user count differs")
    if int(execution["discovery_users"]) != DISCOVERY_USERS:
        raise RuntimeError("attention-cone discovery user count differs")
    if contract["outputs"]["root"] != OUTPUT_ROOT.relative_to(ROOT).as_posix():
        raise RuntimeError("attention-cone output root differs")
    for record in contract["frozen_inputs"].values():
        path = ROOT / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"frozen attention-cone input differs: {path}")
    return contract, sha256_file(CONTRACT)


def verify_candidate_partition(candidates: np.ndarray) -> None:
    anchors = candidates[:, :, ANCHOR_INDICES]
    heldout = candidates[:, :, HELDOUT_INDICES]
    if anchors.shape[-1] != REQUIRED_ANCHOR_COUNT or heldout.shape[-1] != 32:
        raise RuntimeError("candidate partition width differs")
    if np.any(anchors[..., :, None] == heldout[..., None, :]):
        raise RuntimeError("anchor and held-out candidate identities overlap")


def verify_model(model, payload: dict[str, Any]) -> None:
    verify_model_payload(payload)
    cfg = model.cfg
    expected = {
        "hidden_size": EXPECTED_HIDDEN,
        "num_layers": EXPECTED_LAYERS,
        "num_heads": EXPECTED_HEADS,
        "max_seq_len": HISTORY,
    }
    for name, value in expected.items():
        if int(getattr(cfg, name)) != value:
            raise RuntimeError(f"model {name} differs: {getattr(cfg, name)}")
    resolved_head_dim = cfg.head_dim or (cfg.hidden_size // cfg.num_heads)
    if int(resolved_head_dim) != EXPECTED_HEAD_DIM:
        raise RuntimeError(f"resolved model head_dim differs: {resolved_head_dim}")
    if cfg.activation != "elu_plus1":
        raise RuntimeError("model attention activation is not elu_plus1")
    if cfg.block_variant != "legacy":
        raise RuntimeError("model block variant is not legacy")
    if bool(cfg.relative_position_bias):
        raise RuntimeError("model unexpectedly has relative-position bias")
    if model.training:
        raise RuntimeError("loaded model must be in eval mode")
    for block in model.blocks:
        attention = block.attn
        if attention.activation != "elu_plus1":
            raise RuntimeError("block attention activation differs")
        if attention.block_variant != "legacy":
            raise RuntimeError("block attention variant differs")
        if attention.position_bias is not None:
            raise RuntimeError("block has a relative-position bias table")
        if attention.num_heads != EXPECTED_HEADS or attention.head_dim != EXPECTED_HEAD_DIM:
            raise RuntimeError("block head layout differs")


def require_discovery_gate(contract_hash: str) -> None:
    analysis_path = OUTPUT_ROOT / "canary/analysis/summary.json"
    if not analysis_path.is_file() or not RESOURCE_ESTIMATE.is_file():
        raise RuntimeError(
            "attention-cone discovery requires canary adjudication and resource estimate"
        )
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    estimate = json.loads(RESOURCE_ESTIMATE.read_text(encoding="utf-8"))
    if analysis.get("contract_sha256") != contract_hash or not analysis.get(
        "discovery_launch_gate_passed"
    ):
        raise RuntimeError("attention-cone canary did not pass the discovery gate")
    if estimate.get("contract_sha256") != contract_hash:
        raise RuntimeError("attention-cone resource estimate contract differs")
    if "estimated_512_user_minutes" not in estimate:
        raise RuntimeError("attention-cone resource estimate is incomplete")


def write_resource_estimate() -> None:
    _, contract_hash = verify_contract()
    canary_path = OUTPUT_ROOT / "canary/summary.json"
    analysis_path = OUTPUT_ROOT / "canary/analysis/summary.json"
    if RESOURCE_ESTIMATE.exists():
        raise FileExistsError(f"refusing to overwrite {RESOURCE_ESTIMATE}")
    if not canary_path.is_file() or not analysis_path.is_file():
        raise RuntimeError("resource estimate requires canary run and adjudication")
    canary = json.loads(canary_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if not canary.get("passed") or not analysis.get("discovery_launch_gate_passed"):
        raise RuntimeError("attention-cone canary did not unlock resource estimation")
    if canary.get("contract_sha256") != contract_hash or analysis.get(
        "contract_sha256"
    ) != contract_hash:
        raise RuntimeError("attention-cone canary evidence contract differs")
    ratio = DISCOVERY_USERS / CANARY_USERS
    seconds = float(canary["elapsed_seconds"]) * ratio
    payload = {
        "status": "attention_cone_moment_discovery_resource_estimate",
        "contract_sha256": contract_hash,
        "canary_users": CANARY_USERS,
        "discovery_users": DISCOVERY_USERS,
        "conservative_user_ratio": ratio,
        "canary_elapsed_seconds": float(canary["elapsed_seconds"]),
        "estimated_512_user_seconds": seconds,
        "estimated_512_user_minutes": seconds / 60.0,
        "canary_peak_allocated_mib": float(canary["peak_allocated_mib"]),
        "canary_peak_reserved_mib": float(canary["peak_reserved_mib"]),
        "recommended_launch_mode": "detached" if seconds > 30 * 60 else "interactive_ok",
        "exceeds_30_minutes": seconds > 30 * 60,
    }
    RESOURCE_ESTIMATE.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(RESOURCE_ESTIMATE, payload)
    print(json.dumps(payload, indent=2), flush=True)


def distributed_context() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        raise RuntimeError("attention-cone diagnostic requires torchrun")
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 4:
        raise RuntimeError(f"attention-cone diagnostic requires four ranks, got {world}")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world


def _cache_heads(attention, values: torch.Tensor) -> torch.Tensor:
    if values.shape != (1, HISTORY, attention.inner):
        raise RuntimeError(f"layer cache shape differs: {tuple(values.shape)}")
    return values.view(
        1, HISTORY, attention.num_heads, attention.head_dim
    ).transpose(1, 2)


def _native_prefix_heads(attention, q, layer_k, layer_v) -> torch.Tensor:
    keys = _cache_heads(attention, layer_k).expand(q.shape[0], -1, -1, -1)
    values = _cache_heads(attention, layer_v).expand(q.shape[0], -1, -1, -1)
    weights = attention._activate(scaled_qk_logits(q, keys, scale=attention.scale))
    return torch.matmul(attention.attn_dropout(weights), values)


def _native_self_heads(attention, q, k_new, v_new) -> torch.Tensor:
    if attention.causal_diagonal != "inclusive":
        return torch.zeros_like(v_new)
    weights = attention._activate(
        (q * k_new).sum(dim=-1, keepdim=True) * attention.scale
    )
    return attention.attn_dropout(weights) * v_new


def _block_update(block, x_norm: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
    attention_out = block.attn._finish(heads)
    if block.gating == "silu_gate":
        return attention_out * torch.nn.functional.silu(block.gate_proj(x_norm))
    if block.gating == "glu":
        return attention_out * torch.sigmoid(block.gate_proj(x_norm))
    if block.gating == "ffn":
        return block.fc2(
            torch.nn.functional.silu(block.fc1(x_norm)) * block.fc3(x_norm)
        )
    return attention_out


@torch.inference_mode()
def trace_coherent_reuse_queries(
    model,
    reuse_cache,
    candidates: torch.Tensor,
    query_delta: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Trace Current q at every layer while consuming complete Parent K/V."""

    if candidates.ndim != 2 or candidates.shape[0] != 1:
        raise ValueError("trace candidates must have shape [1,C]")
    count = candidates.shape[1]
    x = model.embed_query_tokens(candidates, query_delta).reshape(
        count, 1, model.cfg.hidden_size
    )
    queries: list[torch.Tensor] = []
    for layer, block in enumerate(model.blocks):
        residual = x
        x_norm = block.norm(x)
        q, k_new, v_new = block.attn._project(x_norm)
        queries.append(q.detach())
        prefix = _native_prefix_heads(
            block.attn, q, reuse_cache.k[layer], reuse_cache.v[layer]
        )
        self_heads = _native_self_heads(block.attn, q, k_new, v_new)
        x = residual + _block_update(block, x_norm, prefix + self_heads)
    return tuple(queries)


def build_memory_from_queries(
    model,
    exact_cache,
    reuse_cache,
    anchor_queries: tuple[torch.Tensor, ...],
    *,
    positions: torch.Tensor | None,
    weights: torch.Tensor | None,
) -> ConeResponseMemory:
    if len(anchor_queries) != len(model.blocks):
        raise ValueError("anchor query trace and model layer counts differ")
    layers = tuple(
        build_layer_signed_cone_moment(
            block.attn,
            anchor_queries[layer],
            exact_cache.k[layer],
            exact_cache.v[layer],
            reuse_cache.k[layer],
            reuse_cache.v[layer],
            current_sample_positions=positions,
            current_sample_weights=weights,
        )
        for layer, block in enumerate(model.blocks)
    )
    return ConeResponseMemory(
        layers=layers,
        source_length=exact_cache.seq_len,
        anchor_count=REQUIRED_ANCHOR_COUNT,
        source_kv_scalars=exact_cache.k.numel() + exact_cache.v.numel(),
    )


def _split_cone_metrics(
    attention,
    q: torch.Tensor,
    exact_k: torch.Tensor,
    exact_v: torch.Tensor,
    reuse_k: torch.Tensor,
    reuse_v: torch.Tensor,
    current_majority: torch.Tensor,
    parent_majority: torch.Tensor,
) -> dict[str, torch.Tensor]:
    count = q.shape[0]
    exact_keys = _cache_heads(attention, exact_k).expand(count, -1, -1, -1)
    exact_values = _cache_heads(attention, exact_v).expand(count, -1, -1, -1)
    reuse_keys = _cache_heads(attention, reuse_k).expand(count, -1, -1, -1)
    reuse_values = _cache_heads(attention, reuse_v).expand(count, -1, -1, -1)
    current = decompose_elu_plus_one_response(
        q, exact_keys, exact_values, scale=attention.scale
    )
    parent = decompose_elu_plus_one_response(
        q, reuse_keys, reuse_values, scale=attention.scale
    )
    current_sign = current.positive_mask.squeeze(2)
    parent_sign = parent.positive_mask.squeeze(2)
    current_reference = current_majority.expand(count, -1, -1)
    parent_reference = parent_majority.expand(count, -1, -1)

    def quantile(logits: torch.Tensor, probability: float) -> torch.Tensor:
        by_head = logits.abs().squeeze(2).permute(1, 0, 2).reshape(
            attention.num_heads, -1
        )
        return torch.quantile(by_head.float(), probability, dim=1)

    def pooled_quantile(probability: float) -> torch.Tensor:
        logits = torch.cat((current.logits, parent.logits), dim=-1)
        return quantile(logits, probability)

    return {
        "current_agreement": (current_sign == current_reference).float().mean(dim=(0, 2)),
        "parent_agreement": (parent_sign == parent_reference).float().mean(dim=(0, 2)),
        "sign_crossing": (current_sign != parent_sign).float().mean(dim=(0, 2)),
        "current_negative_activation": current.negative_activation_fraction.squeeze(2).mean(dim=0),
        "parent_negative_activation": parent.negative_activation_fraction.squeeze(2).mean(dim=0),
        "current_negative_response": current.negative_response_fraction.squeeze(2).mean(dim=0),
        "parent_negative_response": parent.negative_response_fraction.squeeze(2).mean(dim=0),
        "current_abs_qk_p50": quantile(current.logits, 0.50),
        "current_abs_qk_p95": quantile(current.logits, 0.95),
        "parent_abs_qk_p50": quantile(parent.logits, 0.50),
        "parent_abs_qk_p95": quantile(parent.logits, 0.95),
        "qk_abs_p50": pooled_quantile(0.50),
        "qk_abs_p95": pooled_quantile(0.95),
    }


def cone_diagnostic_rows(
    *,
    uid: int,
    edge: str,
    model,
    exact_cache,
    reuse_cache,
    anchor_queries: tuple[torch.Tensor, ...],
    heldout_queries: tuple[torch.Tensor, ...],
    full_memory: ConeResponseMemory,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer, block in enumerate(model.blocks):
        moment = full_memory.layers[layer]
        anchor = _split_cone_metrics(
            block.attn,
            anchor_queries[layer],
            exact_cache.k[layer],
            exact_cache.v[layer],
            reuse_cache.k[layer],
            reuse_cache.v[layer],
            moment.current_positive_mask,
            moment.parent_positive_mask,
        )
        heldout = _split_cone_metrics(
            block.attn,
            heldout_queries[layer],
            exact_cache.k[layer],
            exact_cache.v[layer],
            reuse_cache.k[layer],
            reuse_cache.v[layer],
            moment.current_positive_mask,
            moment.parent_positive_mask,
        )
        for head in range(block.attn.num_heads):
            row: dict[str, Any] = {
                "edge": edge,
                "uid": uid,
                "layer": layer,
                "head": head,
                "anchor_count": REQUIRED_ANCHOR_COUNT,
                "heldout_count": len(HELDOUT_INDICES),
            }
            for split, values in (("anchor", anchor), ("heldout", heldout)):
                for name, tensor in values.items():
                    row[f"{split}_{name}"] = float(tensor[head].detach())
            # Stable adjudication names use held-out queries for defects.  The
            # split-prefixed columns above preserve the full anchor/held-out
            # contrast needed to distinguish cone stability from response
            # representation and sampled construction quality.
            row.update(
                {
                    "anchor_current_majority_agreement": float(
                        anchor["current_agreement"][head].detach()
                    ),
                    "anchor_parent_majority_agreement": float(
                        anchor["parent_agreement"][head].detach()
                    ),
                    "heldout_current_majority_agreement": float(
                        heldout["current_agreement"][head].detach()
                    ),
                    "heldout_parent_majority_agreement": float(
                        heldout["parent_agreement"][head].detach()
                    ),
                    "current_parent_sign_crossing_fraction": float(
                        heldout["sign_crossing"][head].detach()
                    ),
                    "current_negative_activation_fraction": float(
                        heldout["current_negative_activation"][head].detach()
                    ),
                    "parent_negative_activation_fraction": float(
                        heldout["parent_negative_activation"][head].detach()
                    ),
                    "current_negative_response_fraction": float(
                        heldout["current_negative_response"][head].detach()
                    ),
                    "parent_negative_response_fraction": float(
                        heldout["parent_negative_response"][head].detach()
                    ),
                    "qk_abs_p50": float(heldout["qk_abs_p50"][head].detach()),
                    "qk_abs_p95": float(heldout["qk_abs_p95"][head].detach()),
                }
            )
            rows.append(row)
    return rows


def _method_row(
    *,
    edge: str,
    uid: int,
    method: str,
    sample_kind: str,
    sample_count: int,
    full_kv_scalars: int,
    memory: ConeResponseMemory | None,
    exact_scores: torch.Tensor,
    reuse_scores: torch.Tensor,
    observed_scores: torch.Tensor,
) -> dict[str, Any]:
    persistent = 0 if memory is None else memory.stored_scalars
    temporary = 0 if memory is None else 2 * EXPECTED_LAYERS * sample_count * EXPECTED_HIDDEN
    return {
        "edge": edge,
        "uid": uid,
        "source": "heldout_odd32",
        "method": method,
        "sample_kind": sample_kind,
        "sample_count": sample_count,
        "persistent_moment_scalars": persistent,
        "persistent_moment_bytes": 0 if memory is None else memory.stored_bytes,
        "persistent_storage_ratio_to_current_KV": persistent / full_kv_scalars,
        "temporary_Current_sample_KV_scalars": temporary,
        "temporary_current_sample_KV_ratio": temporary / full_kv_scalars,
        **metrics_row(score_metrics(exact_scores, reuse_scores, observed_scores)),
    }


@torch.inference_mode()
def evaluate_user(
    *,
    uid: int,
    edge: str,
    parent,
    current,
    items: torch.Tensor,
    actions: torch.Tensor,
    deltas: torch.Tensor,
    query_delta: torch.Tensor,
    anchors: torch.Tensor,
    heldout: torch.Tensor,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    exact_cache = current.compute_kv(items, actions, deltas)
    reuse_cache = parent.compute_kv(items, actions, deltas)
    if exact_cache.seq_len != HISTORY or reuse_cache.seq_len != HISTORY:
        raise RuntimeError("attention-cone cache history length differs")
    exact_before = (exact_cache.k.clone(), exact_cache.v.clone())
    reuse_before = (reuse_cache.k.clone(), reuse_cache.v.clone())
    exact_scores, _ = current.observe_cc_reuse(exact_cache, heldout, query_delta)
    reuse_scores, _ = current.observe_cc_reuse(reuse_cache, heldout, query_delta)
    anchor_queries = trace_coherent_reuse_queries(
        current, reuse_cache, anchors, query_delta
    )
    heldout_queries = trace_coherent_reuse_queries(
        current, reuse_cache, heldout, query_delta
    )
    full_kv_scalars = exact_cache.k.numel() + exact_cache.v.numel()

    records = [
        _method_row(
            edge=edge,
            uid=uid,
            method="Current_Reuse",
            sample_kind="none",
            sample_count=0,
            full_kv_scalars=full_kv_scalars,
            memory=None,
            exact_scores=exact_scores,
            reuse_scores=reuse_scores,
            observed_scores=reuse_scores,
        )
    ]
    full_memory = build_memory_from_queries(
        current,
        exact_cache,
        reuse_cache,
        anchor_queries,
        positions=None,
        weights=None,
    )
    full_observed = intervene_cone_response_memory(
        current, reuse_cache, full_memory, heldout, query_delta
    )
    records.append(
        _method_row(
            edge=edge,
            uid=uid,
            method="full_cone_moment",
            sample_kind="full",
            sample_count=HISTORY,
            full_kv_scalars=full_kv_scalars,
            memory=full_memory,
            exact_scores=exact_scores,
            reuse_scores=reuse_scores,
            observed_scores=full_observed.scores,
        )
    )

    all_memories = [full_memory]
    all_scores = [exact_scores, reuse_scores, full_observed.scores]
    for sample_count in SUPPORTED_SAMPLE_COUNTS:
        strata = fixed_midpoint_strata(HISTORY, sample_count)
        positions = strata.midpoints.to(device=items.device)
        weights = strata.inverse_inclusion_probabilities.to(
            device=items.device, dtype=exact_cache.v.dtype
        )
        memory = build_memory_from_queries(
            current,
            exact_cache,
            reuse_cache,
            anchor_queries,
            positions=positions,
            weights=weights,
        )
        observed = intervene_cone_response_memory(
            current, reuse_cache, memory, heldout, query_delta
        )
        all_memories.append(memory)
        all_scores.append(observed.scores)
        records.append(
            _method_row(
                edge=edge,
                uid=uid,
                method=f"chronological_R{sample_count}",
                sample_kind="chronological",
                sample_count=sample_count,
                full_kv_scalars=full_kv_scalars,
                memory=memory,
                exact_scores=exact_scores,
                reuse_scores=reuse_scores,
                observed_scores=observed.scores,
            )
        )

    for sample_count in SUPPORTED_SAMPLE_COUNTS:
        selection = select_address_landmarks(
            exact_cache, reuse_cache, sample_count=sample_count
        )
        positions = selection.selected_positions
        weights = selection.cluster_masses.to(
            device=items.device, dtype=exact_cache.v.dtype
        )
        memory = build_memory_from_queries(
            current,
            exact_cache,
            reuse_cache,
            anchor_queries,
            positions=positions,
            weights=weights,
        )
        observed = intervene_cone_response_memory(
            current, reuse_cache, memory, heldout, query_delta
        )
        all_memories.append(memory)
        all_scores.append(observed.scores)
        records.append(
            _method_row(
                edge=edge,
                uid=uid,
                method=f"address_R{sample_count}",
                sample_kind="address",
                sample_count=sample_count,
                full_kv_scalars=full_kv_scalars,
                memory=memory,
                exact_scores=exact_scores,
                reuse_scores=reuse_scores,
                observed_scores=observed.scores,
            )
        )

    diagnostics = cone_diagnostic_rows(
        uid=uid,
        edge=edge,
        model=current,
        exact_cache=exact_cache,
        reuse_cache=reuse_cache,
        anchor_queries=anchor_queries,
        heldout_queries=heldout_queries,
        full_memory=full_memory,
    )
    persistent_counts = {memory.stored_scalars for memory in all_memories}
    persistent_ratios = {
        memory.storage_ratio_to_current_kv for memory in all_memories
    }
    finite_metrics = all(
        np.isfinite(value)
        for record in records
        for name, value in record.items()
        if name not in {"edge", "source", "method", "sample_kind"}
    )
    finite_diagnostics = all(
        np.isfinite(value)
        for row in diagnostics
        for name, value in row.items()
        if name != "edge"
    )
    correctness = {
        "edge": edge,
        "uid": uid,
        "finite_all_paths": bool(
            all(torch.isfinite(scores).all() for scores in all_scores)
            and finite_metrics
            and finite_diagnostics
        ),
        "input_exact_cache_unchanged": bool(
            torch.equal(exact_cache.k, exact_before[0])
            and torch.equal(exact_cache.v, exact_before[1])
        ),
        "input_reuse_cache_unchanged": bool(
            torch.equal(reuse_cache.k, reuse_before[0])
            and torch.equal(reuse_cache.v, reuse_before[1])
        ),
        "persistent_scalars_exact": persistent_counts == {EXPECTED_MOMENT_SCALARS},
        "persistent_ratio_exact": persistent_ratios == {EXPECTED_MOMENT_RATIO},
        "method_count": len(records),
        "cone_diagnostic_rows": len(diagnostics),
    }
    return records, diagnostics, correctness


def main() -> None:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--scope", choices=("canary", "discovery"))
    action.add_argument("--write-resource-estimate", action="store_true")
    args = parser.parse_args()
    if args.write_resource_estimate:
        write_resource_estimate()
        return

    rank, local_rank, world = distributed_context()
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(4)
    started = time.perf_counter()
    verification: list[dict[str, Any] | None] = [None]
    if rank == 0:
        try:
            _, contract_hash = verify_contract()
            if args.scope == "discovery":
                require_discovery_gate(contract_hash)
            verification[0] = {"ok": True, "contract_sha256": contract_hash}
        except BaseException as error:
            verification[0] = {"ok": False, "error": repr(error)}
    dist.broadcast_object_list(verification, src=0)
    assert verification[0] is not None
    if not verification[0]["ok"]:
        raise RuntimeError(f"contract verification failed: {verification[0]['error']}")
    contract_hash = str(verification[0]["contract_sha256"])

    all_uids, all_candidates, _ = load_frozen_inputs()
    verify_candidate_partition(all_candidates)
    user_count = CANARY_USERS if args.scope == "canary" else DISCOVERY_USERS
    selected_indices = np.arange(user_count, dtype=np.int64)
    local_indices = selected_indices[rank::world]
    local_uids = all_uids[local_indices]
    output = OUTPUT_ROOT / str(args.scope)
    partial = output.with_name(output.name + ".partial")
    if rank == 0:
        if output.exists() or partial.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        partial.mkdir(parents=True)
        atomic_json(
            partial / "configuration.json",
            {
                "contract_sha256": contract_hash,
                "scope": args.scope,
                "users": user_count,
                "edges": list(EDGES),
                "anchor_indices": list(ANCHOR_INDICES),
                "heldout_indices": list(HELDOUT_INDICES),
                "sample_counts": list(SUPPORTED_SAMPLE_COUNTS),
                "persistent_moment_scalars": EXPECTED_MOMENT_SCALARS,
                "persistent_moment_ratio": EXPECTED_MOMENT_RATIO,
                "labels_read": False,
                "oracle_exact_cache_used": True,
                "sampled_Current_upper_layer_KV_oracle": True,
                "Parent_moments_use_complete_cache": True,
            },
        )
    dist.barrier()
    rank_output = partial / f"rank{rank}"
    rank_output.mkdir()

    history = load_histories(
        local_uids.tolist(),
        oov_buckets=OOV_BUCKETS,
        dataset_path=DATASET,
        known_vocab_size=KNOWN_ITEMS,
        end_timestamp=(CUTOVER_DAYS[-1] + 1) * DAY,
        threads=8,
    )
    metric_records: list[dict[str, Any]] = []
    diagnostic_records: list[dict[str, Any]] = []
    correctness_records: list[dict[str, Any]] = []
    edge_records: list[dict[str, Any]] = []
    peak_allocated_mib = 0.0
    peak_reserved_mib = 0.0

    for edge_index, edge in enumerate(EDGES):
        edge_started = time.perf_counter()
        print(json.dumps({"phase": "edge_start", "rank": rank, "edge": edge}), flush=True)
        parent, parent_payload = load_model(checkpoint(edge_index), device)
        current, current_payload = load_model(checkpoint(edge_index + 1), device)
        verify_model(parent, parent_payload)
        verify_model(current, current_payload)
        _, items_np, actions_np, deltas_np, query_np = histories_at_cutover(
            history, local_uids, CUTOVER_DAYS[edge_index] * DAY
        )
        candidate_np = all_candidates[edge_index, local_indices]
        torch.cuda.reset_peak_memory_stats(device)

        for local_offset, uid_value in enumerate(local_uids):
            items = torch.as_tensor(
                items_np[local_offset : local_offset + 1], dtype=torch.long, device=device
            )
            actions_tensor = torch.as_tensor(
                actions_np[local_offset : local_offset + 1], dtype=torch.long, device=device
            )
            deltas = torch.as_tensor(
                deltas_np[local_offset : local_offset + 1], dtype=torch.float32, device=device
            )
            query_delta = torch.as_tensor(
                query_np[local_offset : local_offset + 1], dtype=torch.float32, device=device
            )
            anchors = torch.as_tensor(
                candidate_np[local_offset : local_offset + 1, ANCHOR_INDICES],
                dtype=torch.long,
                device=device,
            )
            heldout = torch.as_tensor(
                candidate_np[local_offset : local_offset + 1, HELDOUT_INDICES],
                dtype=torch.long,
                device=device,
            )
            metrics, diagnostics, correctness = evaluate_user(
                uid=int(uid_value),
                edge=edge,
                parent=parent,
                current=current,
                items=items,
                actions=actions_tensor,
                deltas=deltas,
                query_delta=query_delta,
                anchors=anchors,
                heldout=heldout,
            )
            metric_records.extend(metrics)
            diagnostic_records.extend(diagnostics)
            correctness_records.append(correctness)
            peak_allocated_mib = max(
                peak_allocated_mib,
                torch.cuda.max_memory_allocated(device) / (1 << 20),
            )
            peak_reserved_mib = max(
                peak_reserved_mib,
                torch.cuda.max_memory_reserved(device) / (1 << 20),
            )

        seconds = time.perf_counter() - edge_started
        edge_records.append({"edge": edge, "users": len(local_uids), "seconds": seconds})
        print(
            json.dumps(
                {"phase": "edge_complete", "rank": rank, "edge": edge, "users": len(local_uids), "seconds": seconds}
            ),
            flush=True,
        )
        del parent, current, parent_payload, current_payload
        torch.cuda.empty_cache()

    pd.DataFrame(metric_records).to_parquet(rank_output / "metrics.parquet", index=False)
    pd.DataFrame(diagnostic_records).to_parquet(
        rank_output / "cone_diagnostics.parquet", index=False
    )
    pd.DataFrame(correctness_records).to_parquet(
        rank_output / "correctness.parquet", index=False
    )
    atomic_json(
        rank_output / "summary.json",
        {
            "rank": rank,
            "uids": local_uids.tolist(),
            "edge_records": edge_records,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_mib": peak_allocated_mib,
            "peak_reserved_mib": peak_reserved_mib,
            "labels_read": False,
        },
    )
    dist.barrier()

    if rank == 0:
        metrics = pd.concat(
            [pd.read_parquet(partial / f"rank{i}/metrics.parquet") for i in range(world)],
            ignore_index=True,
        )
        diagnostics = pd.concat(
            [
                pd.read_parquet(partial / f"rank{i}/cone_diagnostics.parquet")
                for i in range(world)
            ],
            ignore_index=True,
        )
        correctness = pd.concat(
            [pd.read_parquet(partial / f"rank{i}/correctness.parquet") for i in range(world)],
            ignore_index=True,
        )
        expected_metrics = user_count * len(EDGES) * METHODS_PER_USER_EDGE
        expected_diagnostics = user_count * len(EDGES) * EXPECTED_LAYERS * EXPECTED_HEADS
        expected_correctness = user_count * len(EDGES)
        if (
            len(metrics) != expected_metrics
            or len(diagnostics) != expected_diagnostics
            or len(correctness) != expected_correctness
        ):
            raise RuntimeError("attention-cone raw row count differs")
        if metrics[["edge", "uid", "method"]].duplicated().any():
            raise RuntimeError("duplicate attention-cone metric key")
        if diagnostics[["edge", "uid", "layer", "head"]].duplicated().any():
            raise RuntimeError("duplicate attention-cone diagnostic key")
        passed = bool(
            correctness.finite_all_paths.all()
            and correctness.input_exact_cache_unchanged.all()
            and correctness.input_reuse_cache_unchanged.all()
            and correctness.persistent_scalars_exact.all()
            and correctness.persistent_ratio_exact.all()
            and (correctness.method_count == METHODS_PER_USER_EDGE).all()
            and (
                correctness.cone_diagnostic_rows
                == EXPECTED_LAYERS * EXPECTED_HEADS
            ).all()
        )
        combined_paths = (
            partial / "metrics.parquet",
            partial / "cone_diagnostics.parquet",
            partial / "correctness.parquet",
        )
        metrics.to_parquet(combined_paths[0], index=False)
        diagnostics.to_parquet(combined_paths[1], index=False)
        correctness.to_parquet(combined_paths[2], index=False)
        rank_summaries = [
            json.loads((partial / f"rank{i}/summary.json").read_text(encoding="utf-8"))
            for i in range(world)
        ]
        artifacts = {
            path.name: {
                "rows": len(metrics)
                if path.name == "metrics.parquet"
                else len(diagnostics)
                if path.name == "cone_diagnostics.parquet"
                else len(correctness),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in combined_paths
        }
        summary = {
            "status": (
                "attention_cone_moment_instrumentation_passed"
                if passed
                else "attention_cone_moment_instrumentation_failed"
            ),
            "passed": passed,
            "contract_sha256": contract_hash,
            "scope": args.scope,
            "users": user_count,
            "edges": list(EDGES),
            "labels_read": False,
            "oracle_exact_cache_used": True,
            "sampled_Current_upper_layer_KV_oracle": True,
            "all_numeric_rows_finite": bool(correctness.finite_all_paths.all()),
            "input_caches_unchanged": bool(
                correctness.input_exact_cache_unchanged.all()
                and correctness.input_reuse_cache_unchanged.all()
            ),
            "persistent_moment_scalars": EXPECTED_MOMENT_SCALARS,
            "persistent_moment_ratio": EXPECTED_MOMENT_RATIO,
            "elapsed_seconds": max(float(row["elapsed_seconds"]) for row in rank_summaries),
            "peak_allocated_mib": max(
                float(row["peak_allocated_mib"]) for row in rank_summaries
            ),
            "peak_reserved_mib": max(
                float(row["peak_reserved_mib"]) for row in rank_summaries
            ),
            "artifacts": artifacts,
        }
        atomic_json(partial / "summary.json", summary)
        partial.replace(output)
        print(json.dumps(summary, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
