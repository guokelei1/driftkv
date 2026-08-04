from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist

from ..models import HSTUKVCache
from .low_rank import LowRankLayerAdapter
from .stage45_oldkv import DirectOldKVProgram
from .xp_d1_quality import apply_direct_oldkv
from .xp_exact_baseline import canonical_sha256

RECURSIVE_D1_PROTOCOL = "evokv_qk_recursive_d1_round_a_development_v0"
RECURSIVE_ACTION_PLAN_PROTOCOL = (
    "evokv_qk_recursive_d1_action_plan_development_v0"
)
RECURSIVE_METHODS = (
    "reuse_exact_baselines",
    "incumbent_rank16_recursive",
    "rollout_only_exact0",
    "ract_kv_exact0",
    "ract_kv_exact10",
    "ract_kv_exact20",
)


@dataclass
class RecursiveBatchState:
    cache: HSTUKVCache
    record_ids: torch.Tensor
    depths: torch.Tensor
    last_exact_versions: torch.Tensor

    def __post_init__(self) -> None:
        batch = self.cache.k.shape[1]
        if (
            self.cache.k.shape != self.cache.v.shape
            or self.cache.k.ndim != 4
            or self.record_ids.shape != (batch,)
            or self.depths.shape != (batch,)
            or self.last_exact_versions.shape != (batch,)
            or any(
                value.device.type != "cpu"
                for value in (
                    self.cache.k,
                    self.cache.v,
                    self.record_ids,
                    self.depths,
                    self.last_exact_versions,
                )
            )
            or self.cache.k.dtype != torch.float16
            or self.cache.v.dtype != torch.float16
        ):
            raise ValueError("recursive D1 batch state differs")


def storage_cache(cache: HSTUKVCache) -> HSTUKVCache:
    return HSTUKVCache(
        k=cache.k.detach().to(device="cpu", dtype=torch.float16),
        v=cache.v.detach().to(device="cpu", dtype=torch.float16),
        seq_len=cache.seq_len,
    )


def mix_exact_cache(
    compiled: HSTUKVCache,
    exact: HSTUKVCache,
    exact_rows: torch.Tensor,
) -> HSTUKVCache:
    if (
        compiled.k.shape != exact.k.shape
        or compiled.v.shape != exact.v.shape
        or exact_rows.shape != (compiled.k.shape[1],)
        or exact_rows.device != compiled.k.device
    ):
        raise ValueError("recursive D1 mixed cache differs")
    mask = exact_rows[None, :, None, None]
    return HSTUKVCache(
        k=torch.where(mask, exact.k, compiled.k).contiguous(),
        v=torch.where(mask, exact.v, compiled.v).contiguous(),
        seq_len=compiled.seq_len,
    )


def token_balanced_renewal(
    records: Sequence[tuple[int, int, str]],
    *,
    colors: int | None,
    edge_ordinal: int,
    salt: str,
) -> tuple[set[int], dict[str, object]]:
    if (
        not records
        or edge_ordinal < 0
        or not salt
        or colors is not None
        and colors < 2
    ):
        raise ValueError("recursive D1 renewal request differs")
    record_ids = [int(value[0]) for value in records]
    if (
        len(record_ids) != len(set(record_ids))
        or any(int(tokens) < 1 or not str(stratum) for _, tokens, stratum in records)
    ):
        raise ValueError("recursive D1 renewal records differ")
    if colors is None:
        total_tokens = sum(int(value[1]) for value in records)
        return set(), {
            "policy": "no_scheduled_exact_ablation",
            "colors": None,
            "scheduled_color": None,
            "records": len(records),
            "total_valid_tokens": total_tokens,
            "integer_token_cap": 0,
            "scheduled_exact_records": 0,
            "scheduled_exact_valid_tokens": 0,
            "actual_valid_token_fraction": 0.0,
            "within_integer_token_cap": True,
            "quality_labels_read": False,
        }
    strata: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for record_id, tokens, stratum in records:
        strata[str(stratum)].append((int(record_id), int(tokens)))
    assignments: dict[int, int] = {}
    color_tokens = [0] * colors
    color_records = [0] * colors
    stratum_ledgers = []
    for name in sorted(strata):
        local_tokens = [0] * colors
        local_records = [0] * colors
        ordered = sorted(
            strata[name],
            key=lambda value: (
                -value[1],
                hashlib.sha256(
                    f"{salt}:{name}:{value[0]}".encode()
                ).digest(),
                value[0],
            ),
        )
        for record_id, tokens in ordered:
            color = min(
                range(colors),
                key=lambda index: (
                    local_tokens[index],
                    color_tokens[index],
                    local_records[index],
                    index,
                ),
            )
            assignments[record_id] = color
            local_tokens[color] += tokens
            local_records[color] += 1
            color_tokens[color] += tokens
            color_records[color] += 1
        stratum_ledgers.append(
            {
                "stratum": name,
                "records_by_color": local_records,
                "valid_tokens_by_color": local_tokens,
            }
        )
    scheduled_color = edge_ordinal % colors
    exact_ids = {
        record_id
        for record_id, color in assignments.items()
        if color == scheduled_color
    }
    total_tokens = sum(int(value[1]) for value in records)
    exact_tokens = sum(
        int(tokens)
        for record_id, tokens, _ in records
        if int(record_id) in exact_ids
    )
    maximum_record_tokens = max(int(value[1]) for value in records)
    integer_cap = math.ceil(total_tokens / colors) + maximum_record_tokens - 1
    ledger = {
        "policy": "stable_hash_stratified_greedy_token_balanced_colors",
        "colors": colors,
        "scheduled_color": scheduled_color,
        "records": len(records),
        "total_valid_tokens": total_tokens,
        "integer_token_cap": integer_cap,
        "scheduled_exact_records": len(exact_ids),
        "scheduled_exact_valid_tokens": exact_tokens,
        "actual_record_fraction": len(exact_ids) / len(records),
        "actual_valid_token_fraction": exact_tokens / total_tokens,
        "within_integer_token_cap": exact_tokens <= integer_cap,
        "maximum_nominal_fraction_rounding_slack": (
            maximum_record_tokens / total_tokens
        ),
        "records_by_color": color_records,
        "valid_tokens_by_color": color_tokens,
        "selection_salt": salt,
        "strata": stratum_ledgers,
        "quality_labels_read": False,
        "recommendation_metrics_read": False,
    }
    return exact_ids, ledger


def update_lineage(
    state: RecursiveBatchState,
    *,
    target_version: int,
    exact_record_ids: set[int],
    action_for_nonexact: str,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    if target_version < 1 or action_for_nonexact not in {"compiled", "reuse"}:
        raise ValueError("recursive D1 lineage request differs")
    depths = state.depths.clone()
    last_exact = state.last_exact_versions.clone()
    rows = []
    for index, raw_record_id in enumerate(state.record_ids.tolist()):
        record_id = int(raw_record_id)
        if record_id < 0:
            continue
        input_depth = int(depths[index])
        if record_id in exact_record_ids:
            action = "exact"
            output_depth = 0
            depths[index] = 0
            last_exact[index] = target_version
        else:
            action = action_for_nonexact
            output_depth = input_depth + 1
            depths[index] = output_depth
        rows.append(
            {
                "record_id": record_id,
                "action": action,
                "input_depth": input_depth,
                "output_depth": output_depth,
                "last_exact_version": int(last_exact[index]),
            }
        )
    return depths, last_exact, rows


def _rank_world(process_group=None) -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return (
            dist.get_rank(group=process_group),
            dist.get_world_size(group=process_group),
        )
    return 0, 1


def _all_reduce(value: torch.Tensor, process_group=None) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, group=process_group)
    return value


def select_depth_balanced_tokens(
    record_id_batches: Sequence[torch.Tensor],
    length_batches: Sequence[torch.Tensor],
    depth_batches: Sequence[torch.Tensor],
    *,
    maximum_global_tokens: int,
    seed: int,
    process_group=None,
) -> tuple[torch.Tensor, dict[str, object]]:
    if (
        not record_id_batches
        or len(record_id_batches) != len(length_batches)
        or len(record_id_batches) != len(depth_batches)
        or maximum_global_tokens < 2
        or seed < 0
    ):
        raise ValueError("recursive D1 token sampling request differs")
    rank, world_size = _rank_world(process_group)
    descriptors = []
    local_index = 0
    for record_ids, lengths, depths in zip(
        record_id_batches,
        length_batches,
        depth_batches,
        strict=True,
    ):
        if (
            record_ids.ndim != 1
            or lengths.shape != record_ids.shape
            or depths.shape != record_ids.shape
        ):
            raise ValueError("recursive D1 token sampling batches differ")
        for record_id, length, depth in zip(
            record_ids.tolist(),
            lengths.tolist(),
            depths.tolist(),
            strict=True,
        ):
            if int(record_id) < 0:
                continue
            for position in range(int(length)):
                digest = hashlib.sha256(
                    f"{seed}:{int(depth)}:{int(record_id)}:{position}".encode()
                ).hexdigest()
                descriptors.append(
                    (
                        int(depth),
                        digest,
                        rank,
                        local_index,
                        int(record_id),
                        position,
                    )
                )
                local_index += 1
    gathered: list[object] = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(gathered, descriptors, group=process_group)
    else:
        gathered[0] = descriptors
    selected_payload: list[object] = [None]
    if rank == 0:
        flattened = [value for values in gathered for value in values]
        by_depth: dict[int, list[tuple[object, ...]]] = defaultdict(list)
        for value in flattened:
            by_depth[int(value[0])].append(value)
        limit = min(maximum_global_tokens, len(flattened))
        allocations = {depth: 0 for depth in by_depth}
        remaining = limit
        active = set(by_depth)
        while remaining and active:
            for depth in sorted(tuple(active)):
                if remaining == 0:
                    break
                if allocations[depth] >= len(by_depth[depth]):
                    active.remove(depth)
                    continue
                allocations[depth] += 1
                remaining -= 1
        selected = []
        for depth in sorted(by_depth):
            ordered = sorted(
                by_depth[depth],
                key=lambda value: (value[1], value[2], value[3]),
            )
            selected.extend(ordered[: allocations[depth]])
        selected_payload[0] = {
            "pairs": [(int(value[2]), int(value[3])) for value in selected],
            "global_available_tokens": len(flattened),
            "global_selected_tokens": len(selected),
            "selected_tokens_by_depth": {
                str(depth): allocations[depth] for depth in sorted(allocations)
            },
        }
    if world_size > 1:
        dist.broadcast_object_list(selected_payload, src=0, group=process_group)
    payload = selected_payload[0]
    local = sorted(
        int(index)
        for owner, index in payload["pairs"]
        if int(owner) == rank
    )
    return torch.tensor(local, dtype=torch.int64), {
        **{key: value for key, value in payload.items() if key != "pairs"},
        "local_available_tokens": len(descriptors),
        "local_selected_tokens": len(local),
        "maximum_global_tokens": maximum_global_tokens,
        "selection_seed": seed,
        "depth_balanced": True,
    }


def _flatten_layer(
    caches: Sequence[HSTUKVCache],
    lengths: Sequence[torch.Tensor],
    layer: int,
) -> torch.Tensor:
    values = []
    for cache, batch_lengths in zip(caches, lengths, strict=True):
        if batch_lengths.shape != (cache.k.shape[1],):
            raise ValueError("recursive D1 flattened cache lengths differ")
        joined = torch.cat((cache.k[layer], cache.v[layer]), dim=-1)
        valid = (
            torch.arange(cache.seq_len)[None, :]
            < batch_lengths.to(device="cpu")[:, None]
        )
        values.append(joined[valid])
    return torch.cat(values)


def _distributed_mean(
    value: torch.Tensor,
    *,
    process_group=None,
) -> tuple[torch.Tensor, int]:
    count = torch.tensor(
        [value.shape[0]], device=value.device, dtype=torch.float64
    )
    total = value.double().sum(dim=0)
    _all_reduce(count, process_group)
    _all_reduce(total, process_group)
    global_count = int(count.item())
    if global_count < 2:
        raise ValueError("recursive D1 fit requires two global rows")
    return (total / count).float(), global_count


def _randomized_ridge_layer(
    centered_features: torch.Tensor,
    centered_targets: torch.Tensor,
    *,
    feature_mean: torch.Tensor,
    target_mean: torch.Tensor,
    rank: int,
    ridge: float,
    seed: int,
    maximum_jitter_multiplier: float,
    process_group=None,
) -> tuple[LowRankLayerAdapter, dict[str, object]]:
    if (
        centered_features.ndim != 2
        or centered_targets.ndim != 2
        or centered_features.shape[0] != centered_targets.shape[0]
        or centered_features.shape[0] < 1
        or feature_mean.shape != (centered_features.shape[1],)
        or target_mean.shape != (centered_targets.shape[1],)
        or not 1 <= rank <= min(
            centered_features.shape[1], centered_targets.shape[1]
        )
        or ridge <= 0
        or maximum_jitter_multiplier < 1.0
    ):
        raise ValueError("recursive D1 randomized ridge inputs differ")
    x = centered_features.float()
    y = centered_targets.float()
    count = torch.tensor([x.shape[0]], device=x.device, dtype=torch.float64)
    _all_reduce(count, process_group)
    global_count = int(count.item())
    gram = x.T @ x
    _all_reduce(gram, process_group)
    gram = gram / global_count
    symmetry_error = float((gram - gram.T).abs().max())
    gram = 0.5 * (gram + gram.T)
    scale = gram.diagonal().mean().clamp_min(torch.finfo(gram.dtype).eps)
    identity = torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    base_regularization = ridge * scale
    maximum_jitter = maximum_jitter_multiplier * base_regularization
    jitter = torch.zeros((), device=gram.device, dtype=gram.dtype)
    cholesky = None
    equilibration = None
    cholesky_attempts = 0
    for attempt in range(8):
        cholesky_attempts = attempt + 1
        regularized = gram + (base_regularization + jitter) * identity
        candidate_equilibration = regularized.diagonal().clamp_min(
            torch.finfo(gram.dtype).eps * scale
        ).sqrt()
        equilibrated = (
            regularized
            / candidate_equilibration[:, None]
            / candidate_equilibration[None, :]
        )
        candidate, info = torch.linalg.cholesky_ex(
            equilibrated,
            check_errors=False,
        )
        if int(info.max()) == 0:
            cholesky = candidate
            equilibration = candidate_equilibration
            break
        proposed = (
            torch.finfo(gram.dtype).eps
            * 32.0
            * scale
            * (10.0**attempt)
        )
        jitter = torch.minimum(maximum_jitter, proposed)
    if cholesky is None or equilibration is None:
        raise RuntimeError(
            "recursive D1 ridge solve remains non-positive-definite"
        )

    def solve(rhs: torch.Tensor) -> torch.Tensor:
        scaled = rhs / equilibration[:, None]
        return torch.cholesky_solve(
            scaled, cholesky
        ) / equilibration[:, None]
    sketch_width = min(
        rank + 8,
        centered_features.shape[1],
        centered_targets.shape[1],
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    omega = torch.randn(
        centered_targets.shape[1],
        sketch_width,
        generator=generator,
        dtype=torch.float32,
    ).to(x.device)
    cross_omega = x.T @ (y @ omega)
    _all_reduce(cross_omega, process_group)
    cross_omega = cross_omega / global_count
    range_basis = solve(cross_omega)
    basis = torch.linalg.qr(range_basis, mode="reduced").Q
    inverse_basis = solve(basis)
    small_transpose = y.T @ (x @ inverse_basis)
    _all_reduce(small_transpose, process_group)
    small = (small_transpose / global_count).T
    left_small, singular, right = torch.linalg.svd(
        small, full_matrices=False
    )
    left = basis @ left_small[:, :rank] * singular[:rank]
    right = right[:rank]
    prediction = x @ left @ right
    losses = torch.tensor(
        [
            float(y.square().sum()),
            float((prediction - y).square().sum()),
            float(y.numel()),
        ],
        dtype=torch.float64,
        device=x.device,
    )
    _all_reduce(losses, process_group)
    baseline_mse = float(losses[0] / losses[2])
    fitted_mse = float(losses[1] / losses[2])
    adapter = LowRankLayerAdapter(
        feature_mean=feature_mean.detach().cpu(),
        residual_mean=target_mean.detach().cpu(),
        left=left.detach().cpu(),
        right=right.detach().cpu(),
    )
    return adapter, {
        "global_objective_rows": global_count,
        "ridge_scale": float(scale),
        "requested_diagonal_regularization": float(base_regularization),
        "numerical_jitter": float(jitter),
        "maximum_jitter_multiplier": maximum_jitter_multiplier,
        "effective_diagonal_regularization": float(
            base_regularization + jitter
        ),
        "cholesky_attempts": cholesky_attempts,
        "diagonal_equilibration_minimum": float(equilibration.min()),
        "diagonal_equilibration_maximum": float(equilibration.max()),
        "gram_symmetry_max_abs_before_projection": symmetry_error,
        "randomized_svd_oversample": sketch_width - rank,
        "baseline_centered_mse": baseline_mse,
        "fitted_centered_mse": fitted_mse,
        "centered_mse_recovery": (
            None
            if baseline_mse == 0.0
            else 1.0 - fitted_mse / baseline_mse
        ),
    }


@torch.no_grad()
def fit_rollout_aware_direct_program(
    base_program: DirectOldKVProgram,
    exact_source_caches: Sequence[HSTUKVCache],
    deployed_source_caches: Sequence[HSTUKVCache],
    exact_target_caches: Sequence[HSTUKVCache],
    length_batches: Sequence[torch.Tensor],
    selected_local_indices: torch.Tensor,
    *,
    mode: str,
    rank: int,
    ridge: float,
    seed: int,
    device: torch.device,
    maximum_jitter_multiplier: float = 10.0,
    process_group=None,
) -> tuple[DirectOldKVProgram, DirectOldKVProgram, dict[str, object]]:
    if (
        mode not in {"rollout_only", "ract_kv"}
        or not exact_source_caches
        or len(exact_source_caches) != len(deployed_source_caches)
        or len(exact_source_caches) != len(exact_target_caches)
        or len(exact_source_caches) != len(length_batches)
        or selected_local_indices.ndim != 1
        or selected_local_indices.numel() < 1
        or base_program.num_layers != exact_source_caches[0].k.shape[0]
    ):
        raise ValueError("recursive D1 direct fit request differs")
    float_weights = []
    float_biases = []
    half_weights = []
    half_biases = []
    layer_metrics = []
    for layer in range(base_program.num_layers):
        exact_source = _flatten_layer(
            exact_source_caches, length_batches, layer
        ).index_select(0, selected_local_indices)
        deployed_source = _flatten_layer(
            deployed_source_caches, length_batches, layer
        ).index_select(0, selected_local_indices)
        exact_target = _flatten_layer(
            exact_target_caches, length_batches, layer
        ).index_select(0, selected_local_indices)
        source = exact_source.to(device=device, dtype=torch.float32)
        deployed = deployed_source.to(device=device, dtype=torch.float32)
        target = exact_target.to(device=device, dtype=torch.float32)
        base_weight = base_program.weights[layer].to(
            device=device, dtype=torch.float32
        )
        base_bias = base_program.biases[layer].to(
            device=device, dtype=torch.float32
        )
        deployed_residual = target - (deployed @ base_weight + base_bias)
        if mode == "rollout_only":
            absolute_features = deployed
            absolute_targets = deployed_residual
        else:
            source_residual = target - (source @ base_weight + base_bias)
            absolute_features = torch.cat((deployed, source))
            absolute_targets = torch.cat(
                (deployed_residual, source_residual)
            )
        feature_mean, absolute_count = _distributed_mean(
            absolute_features, process_group=process_group
        )
        target_mean, target_count = _distributed_mean(
            absolute_targets, process_group=process_group
        )
        if absolute_count != target_count:
            raise RuntimeError("recursive D1 paired fit counts differ")
        centered_features = absolute_features - feature_mean
        centered_targets = absolute_targets - target_mean
        if mode == "ract_kv":
            difference_features = deployed - source
            difference_targets = -(
                (deployed @ base_weight + base_bias)
                - (source @ base_weight + base_bias)
            )
            centered_features = torch.cat(
                (centered_features, difference_features)
            )
            centered_targets = torch.cat(
                (centered_targets, difference_targets)
            )
        adapter, metrics = _randomized_ridge_layer(
            centered_features,
            centered_targets,
            feature_mean=feature_mean,
            target_mean=target_mean,
            rank=rank,
            ridge=ridge,
            seed=seed + layer * 104729,
            maximum_jitter_multiplier=maximum_jitter_multiplier,
            process_group=process_group,
        )
        left = adapter.left.to(device)
        right = adapter.right.to(device)
        correction = left @ right
        weight = base_weight + correction
        bias = (
            base_bias
            + adapter.residual_mean.to(device)
            - adapter.feature_mean.to(device) @ correction
        )
        if not bool(torch.isfinite(weight).all()) or not bool(
            torch.isfinite(bias).all()
        ):
            raise RuntimeError("recursive D1 fitted program is nonfinite")
        float_weights.append(weight.cpu().contiguous())
        float_biases.append(bias.cpu().contiguous())
        half_weights.append(weight.half().cpu().contiguous())
        half_biases.append(bias.half().cpu().contiguous())
        layer_metrics.append(
            {
                "layer": layer,
                "absolute_global_rows": absolute_count,
                "objective_terms": (
                    ["deployed_to_target"]
                    if mode == "rollout_only"
                    else [
                        "deployed_to_target",
                        "exact_source_anchor",
                        "observed_error_contraction",
                    ]
                ),
                **metrics,
            }
        )
    float_program = DirectOldKVProgram(
        source_version=base_program.source_version,
        target_version=base_program.target_version,
        weights=torch.stack(float_weights).contiguous(),
        biases=torch.stack(float_biases).contiguous(),
    )
    half_program = DirectOldKVProgram(
        source_version=base_program.source_version,
        target_version=base_program.target_version,
        weights=torch.stack(half_weights).contiguous(),
        biases=torch.stack(half_biases).contiguous(),
    )
    return float_program, half_program, {
        "fit_mode": mode,
        "rank": rank,
        "ridge": ridge,
        "layers": layer_metrics,
        "labels_used": False,
        "recommendation_metrics_used": False,
        "qualification_errors_used": False,
        "semantic_operator": "one_shared_direct_old_kv_affine_per_edge",
        "float_program_numel": (
            float_program.weights.numel() + float_program.biases.numel()
        ),
        "deployed_dtype": "torch.float16",
    }


@torch.no_grad()
def rollout_stability_certificate(
    float_program: DirectOldKVProgram,
    half_program: DirectOldKVProgram,
    exact_source_caches: Sequence[HSTUKVCache],
    deployed_source_caches: Sequence[HSTUKVCache],
    exact_target_caches: Sequence[HSTUKVCache],
    length_batches: Sequence[torch.Tensor],
    depth_batches: Sequence[torch.Tensor],
    *,
    target_ratio: float,
    hard_ratio: float,
    device: torch.device,
    process_group=None,
) -> dict[str, object]:
    if (
        not exact_source_caches
        or len(exact_source_caches) != len(deployed_source_caches)
        or len(exact_source_caches) != len(exact_target_caches)
        or len(exact_source_caches) != len(length_batches)
        or len(exact_source_caches) != len(depth_batches)
        or not 0 < target_ratio <= hard_ratio
        or float_program.source_version != half_program.source_version
        or float_program.target_version != half_program.target_version
    ):
        raise ValueError("recursive D1 certificate request differs")
    local_max_depth = max(
        int(depths.max()) if depths.numel() else 0 for depths in depth_batches
    )
    max_depth_tensor = torch.tensor(
        [local_max_depth], dtype=torch.int64, device=device
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(
            max_depth_tensor, op=dist.ReduceOp.MAX, group=process_group
        )
    maximum_depth = int(max_depth_tensor.item())
    sums = torch.zeros(
        float_program.num_layers,
        maximum_depth + 1,
        5,
        dtype=torch.float64,
        device=device,
    )
    float_device = float_program.to(device, dtype=torch.float32)
    half_device = half_program.to(device, dtype=torch.float16)
    for exact_source_cpu, deployed_cpu, target_cpu, lengths_cpu, depths_cpu in zip(
        exact_source_caches,
        deployed_source_caches,
        exact_target_caches,
        length_batches,
        depth_batches,
        strict=True,
    ):
        exact_source = exact_source_cpu.to(device)
        deployed = deployed_cpu.to(device)
        target = target_cpu.to(device)
        float_deployed = apply_direct_oldkv(float_device, deployed)
        float_source = apply_direct_oldkv(float_device, exact_source)
        half_deployed = apply_direct_oldkv(half_device, deployed)
        float_endpoint = HSTUKVCache(
            k=float_deployed.k.half().float(),
            v=float_deployed.v.half().float(),
            seq_len=float_deployed.seq_len,
        )
        half_endpoint = HSTUKVCache(
            k=half_deployed.k.float(),
            v=half_deployed.v.float(),
            seq_len=half_deployed.seq_len,
        )
        valid = (
            torch.arange(deployed.seq_len, device=device)[None, :]
            < lengths_cpu.to(device)[:, None]
        )
        depths = depths_cpu.to(device)
        deltas = (
            (
                torch.cat((deployed.k, deployed.v), dim=-1).float()
                - torch.cat((exact_source.k, exact_source.v), dim=-1).float()
            ),
            (
                torch.cat(
                    (float_deployed.k, float_deployed.v), dim=-1
                )
                - torch.cat((float_source.k, float_source.v), dim=-1)
            ),
            (
                torch.cat((float_source.k, float_source.v), dim=-1)
                - torch.cat((target.k, target.v), dim=-1).float()
            ),
            (
                torch.cat((half_endpoint.k, half_endpoint.v), dim=-1)
                - torch.cat((float_endpoint.k, float_endpoint.v), dim=-1)
            ),
            (
                torch.cat((deployed.k, deployed.v), dim=-1).float()
                - torch.cat((target.k, target.v), dim=-1).float()
            ),
        )
        squared = [value.square().sum(dim=-1).double() for value in deltas]
        for depth in range(maximum_depth + 1):
            mask = valid & (depths[:, None] == depth)
            if bool(mask.any()):
                for metric, value in enumerate(squared):
                    sums[:, depth, metric] += (value * mask[None]).sum(
                        dim=(1, 2)
                    )
    _all_reduce(sums, process_group)
    rows = []
    hard_failure = False
    target_pass = True
    for layer in range(float_program.num_layers):
        for depth in range(maximum_depth + 1):
            incoming, output_delta, residual, quantization, stale = (
                float(value) for value in sums[layer, depth]
            )
            if max(incoming, output_delta, residual, quantization, stale) == 0.0:
                continue
            gain = (
                None
                if incoming == 0.0
                else math.sqrt(output_delta / incoming)
            )
            bound = (
                math.sqrt(output_delta)
                + math.sqrt(residual)
                + math.sqrt(quantization)
            )
            ratio = (
                0.0
                if stale == 0.0 and bound == 0.0
                else math.inf
                if stale == 0.0
                else bound / math.sqrt(stale)
            )
            if not math.isfinite(ratio) or ratio > hard_ratio:
                hard_failure = True
            if not math.isfinite(ratio) or ratio > target_ratio:
                target_pass = False
            rows.append(
                {
                    "layer": layer,
                    "incoming_depth": depth,
                    "rollout_subspace_gain": gain,
                    "incoming_error_l2": math.sqrt(incoming),
                    "mapped_source_difference_l2": math.sqrt(output_delta),
                    "one_edge_exact_source_residual_l2": math.sqrt(residual),
                    "fp16_program_residual_l2": math.sqrt(quantization),
                    "stale_reuse_error_l2": math.sqrt(stale),
                    "recurrence_bound_over_stale_reuse_error": ratio,
                }
            )
    return {
        "status": "hard_failure" if hard_failure else "pass",
        "target_ratio": target_ratio,
        "hard_ratio": hard_ratio,
        "target_pass": target_pass,
        "hard_failure": hard_failure,
        "maximum_incoming_depth": maximum_depth,
        "full_affine_frobenius_gain_by_layer": [
            float(value.float().norm()) for value in float_program.weights
        ],
        "full_affine_gain_kind": "frobenius_conservative_upper_bound",
        "rows": rows,
        "labels_used": False,
        "recommendation_metrics_used": False,
    }


def action_plan_document(
    *,
    method: str,
    source_version: int,
    target_version: int,
    prefix_tokens: int,
    program_sha256: str | None,
    renewal: dict[str, object],
    fallback_all_exact: bool,
    rows: Sequence[dict[str, object]],
    input_lineage_sha256: str,
    output_cache_state_sha256: str,
) -> dict[str, object]:
    if (
        method not in RECURSIVE_METHODS
        or target_version != source_version + 1
        or prefix_tokens < 1
        or not rows
        or not input_lineage_sha256
        or not output_cache_state_sha256
    ):
        raise ValueError("recursive D1 ActionPlan request differs")
    ordered = sorted(rows, key=lambda value: int(value["record_id"]))
    if len(ordered) != len({int(value["record_id"]) for value in ordered}):
        raise ValueError("recursive D1 ActionPlan records overlap")
    records_sha = canonical_sha256({"records": ordered})
    return {
        "protocol": RECURSIVE_ACTION_PLAN_PROTOCOL,
        "scientific_result": False,
        "formal_result": False,
        "method": method,
        "source_version": source_version,
        "target_version": target_version,
        "single_current_serving_model": True,
        "prefix_tokens_per_record": prefix_tokens,
        "program_sha256": program_sha256,
        "renewal": renewal,
        "fallback_all_exact": fallback_all_exact,
        "input_lineage_sha256": input_lineage_sha256,
        "output_cache_state_sha256": output_cache_state_sha256,
        "records": ordered,
        "record_count": len(ordered),
        "records_sha256": records_sha,
        "output_lineage_sha256": canonical_sha256(
            {
                "source_version": source_version,
                "target_version": target_version,
                "records_sha256": records_sha,
                "program_sha256": program_sha256,
                "output_cache_state_sha256": output_cache_state_sha256,
            }
        ),
    }
