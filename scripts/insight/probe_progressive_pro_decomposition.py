#!/usr/bin/env python3
"""Raw-first, label-free error decomposition for progressive PRO."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "insight"))

from evaluate_candidate_shared_exposed_raw import exposed_groups  # noqa: E402
from evaluate_yambda500m_foundation_raw import (  # noqa: E402
    DAY,
    assign_cache,
    load_histories,
    load_model,
    select_cache,
    stacked_cache,
)
from hstu_kvcache.models import HSTUKVCache  # noqa: E402
from hstu_kvcache.models.state_transition import append_with_rolling_cap  # noqa: E402
from one_release_refinement import parameter_cast_maps  # noqa: E402
from pro_lazy_reader import (  # noqa: E402
    PROProbeComponents,
    generate_lazy_pro_probe_components,
)
from probe_recommendation_state_structure import (  # noqa: E402
    KNOWN_ITEMS,
    candidate_panel,
    checkpoint,
    histories_at_cutover,
)
from progressive_pro import (  # noqa: E402
    ProgressivePROSidecar,
    build_progressive_parent_conditioned_carriers,
    combine_two_probe_components,
    fixed_probe_items,
    global_coverage_corrections,
    progressive_corrections,
)
from reader_compatibility_correction import (  # noqa: E402
    intervene_reader_correction,
    trace_reader_correction,
)


CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_progressive_pro_decomposition_v1.yaml"
POPULATION = ROOT / "results/yambda500m_small_seed17/insight_recommendation_state_structure_v1/population.parquet"
REQUESTS = ROOT / "data/manifests/yambda500m_small_hstu_native_rolling_matrix_fast_v3/requests_fidelity.parquet"
DEFAULT_ROOT = ROOT / "results/yambda500m_small_seed17/insight_progressive_pro_v1/decomposition_v1"
EDGES = (
    ("v0_to_v1", 231, 0, 1),
    ("v1_to_v2", 245, 1, 2),
    ("v2_to_v3", 259, 2, 3),
    ("v3_to_v4", 273, 3, 4),
    ("v4_to_v5", 287, 4, 5),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_contract(path: Path) -> tuple[dict[str, Any], str]:
    contract = yaml.safe_load(path.read_text())
    if contract["scope"]["request_labels"] != "prohibited":
        raise RuntimeError("decomposition contract must prohibit request labels")
    for name, record in contract["frozen_inputs"].items():
        if name == "checkpoints":
            continue
        target = ROOT / record["path"]
        if sha256(target) != record["sha256"]:
            raise RuntimeError(f"frozen input differs: {name}")
    for version in range(6):
        record = contract["frozen_inputs"]["checkpoints"][f"v{version}"]
        target = ROOT / record["path"]
        if target != checkpoint(version) or sha256(target) != record["sha256"]:
            raise RuntimeError(f"frozen v{version} checkpoint differs")
    return contract, sha256(path)


def _prefix_tensors(
    history, uids: list[int], cutover: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    timestamps, items, behaviors, deltas, _ = histories_at_cutover(
        history, np.asarray(uids, dtype=np.int64), cutover
    )
    return (
        torch.as_tensor(items, dtype=torch.long, device=device),
        torch.as_tensor(behaviors, dtype=torch.long, device=device),
        torch.as_tensor(deltas, dtype=torch.float32, device=device),
        timestamps,
    )


def _select_tuple(
    values: tuple[torch.Tensor, ...], indices: list[int]
) -> tuple[torch.Tensor, ...]:
    device = values[0].device
    selected = torch.as_tensor(indices, dtype=torch.long, device=device)
    return tuple(value.index_select(0, selected) for value in values)


def _select_sidecar(
    sidecar: ProgressivePROSidecar, indices: list[int]
) -> ProgressivePROSidecar:
    selected = torch.as_tensor(
        indices, dtype=torch.long, device=sidecar.old_amplitudes.device
    )
    return ProgressivePROSidecar(
        directions=tuple(value.index_select(0, selected) for value in sidecar.directions),
        old_amplitudes=sidecar.old_amplitudes.index_select(0, selected),
        recent_amplitudes=sidecar.recent_amplitudes.index_select(0, selected),
        probe_direction_cosines=sidecar.probe_direction_cosines.index_select(0, selected),
        probe_norm_ratios=sidecar.probe_norm_ratios.index_select(0, selected),
    )


def _flatten(value: torch.Tensor) -> torch.Tensor:
    return value.float().flatten(1)


def _component_cosines(
    first: PROProbeComponents, second: PROProbeComponents
) -> torch.Tensor:
    output = []
    for old_one, old_two, recent_one, recent_two in zip(
        first.old_corrections,
        second.old_corrections,
        first.recent_corrections,
        second.recent_corrections,
        strict=True,
    ):
        old = 0.5 * (_flatten(old_one) + _flatten(old_two))
        recent = 0.5 * (_flatten(recent_one) + _flatten(recent_two))
        output.append(F.cosine_similarity(old, recent, dim=1))
    return torch.stack(output, dim=1)


def _layer_rows(
    *,
    edge: str,
    phase: str,
    identifiers: list[str],
    uids: list[int],
    request_positions: list[str],
    evictions: list[int],
    exact: tuple[torch.Tensor, ...],
    predictions: dict[str, tuple[torch.Tensor, ...]],
    probe_cosines: torch.Tensor,
    probe_norm_ratios: torch.Tensor,
    component_cosines: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer, exact_value in enumerate(exact):
        target = _flatten(exact_value)
        target_norm = target.norm(dim=1).clamp_min(1e-12)
        for method, values in predictions.items():
            estimate = _flatten(values[layer])
            estimate_norm = estimate.norm(dim=1).clamp_min(1e-12)
            cosine = F.cosine_similarity(estimate, target, dim=1)
            relative = (estimate - target).norm(dim=1) / target_norm
            direction = estimate / estimate_norm[:, None]
            projected_amplitude = (target * direction).sum(dim=1)
            oracle = direction * projected_amplitude[:, None]
            oracle_relative = (oracle - target).norm(dim=1) / target_norm
            amplitude_gain = 1.0 - oracle_relative / relative.clamp_min(1e-12)
            for index, identifier in enumerate(identifiers):
                rows.append(
                    {
                        "edge": edge,
                        "phase": phase,
                        "identifier": identifier,
                        "uid": int(uids[index]),
                        "request_position": request_positions[index],
                        "rolling_evictions": int(evictions[index]),
                        "method": method,
                        "layer": layer,
                        "direction_cosine_to_exact_shared_av": float(cosine[index]),
                        "estimated_to_exact_norm_ratio": float(
                            estimate_norm[index] / target_norm[index]
                        ),
                        "relative_l2_to_exact_shared_av": float(relative[index]),
                        "oracle_projection_amplitude_relative_l2": float(
                            oracle_relative[index]
                        ),
                        "oracle_projection_relative_l2_reduction": float(
                            amplitude_gain[index]
                        ),
                        "fixed_probe_direction_cosine": float(
                            probe_cosines[index, layer]
                        ),
                        "fixed_probe_norm_ratio": float(
                            probe_norm_ratios[index, layer]
                        ),
                        "old_to_recent_component_direction_cosine": float(
                            component_cosines[index, layer]
                        ),
                        "labels_read": False,
                    }
                )
    return rows


def _score_rows(
    *,
    edge: str,
    phase: str,
    identifiers: list[str],
    uids: list[int],
    request_positions: list[str],
    evictions: list[int],
    trace,
    predicted_scores: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    exact = trace.exact_scores.float()
    exact_probability = torch.sigmoid(exact)
    shared = trace.stage_scores["av_aggregation"].float()
    reuse = trace.reuse_scores.float()
    reuse_probability_gap = torch.abs(torch.sigmoid(reuse) - exact_probability).mean(dim=1)
    paths = {"reuse": reuse, "exact_shared_av": shared, **predicted_scores}
    rows = []
    for method, scores in paths.items():
        probability_gap = torch.abs(torch.sigmoid(scores.float()) - exact_probability).mean(dim=1)
        for index, identifier in enumerate(identifiers):
            rows.append(
                {
                    "edge": edge,
                    "phase": phase,
                    "identifier": identifier,
                    "uid": int(uids[index]),
                    "request_position": request_positions[index],
                    "rolling_evictions": int(evictions[index]),
                    "method": method,
                    "candidate_width": int(scores.shape[1]),
                    "mean_abs_logit_gap_to_current_exact": float(
                        torch.abs(scores[index] - exact[index]).mean()
                    ),
                    "mean_abs_probability_gap_to_current_exact": float(
                        probability_gap[index]
                    ),
                    "probability_gap_recovery_over_reuse": float(
                        0.0
                        if method == "reuse"
                        else 1.0
                        - probability_gap[index]
                        / reuse_probability_gap[index].clamp_min(1e-12)
                    ),
                    "mean_abs_logit_gap_to_exact_shared_av": float(
                        torch.abs(scores[index] - shared[index]).mean()
                    ),
                    "labels_read": False,
                }
            )
    return rows


@torch.inference_mode()
def _make_sidecar(
    *, parent_cache, current, maps, items, behaviors, deltas, carrier_count: int = 32
) -> tuple[PROProbeComponents, PROProbeComponents, ProgressivePROSidecar, torch.Tensor]:
    carriers, layout = build_progressive_parent_conditioned_carriers(
        parent_cache=parent_cache,
        current=current,
        item_ids=items,
        behaviors=behaviors,
        time_deltas=deltas,
        repair_width=128,
        carrier_count=carrier_count,
    )
    probes = fixed_probe_items(items)
    first = generate_lazy_pro_probe_components(
        current,
        parent_cache,
        carriers,
        maps,
        probes[:, 0],
        old_positions=layout.old_positions,
    )
    second = generate_lazy_pro_probe_components(
        current,
        parent_cache,
        carriers,
        maps,
        probes[:, 1],
        old_positions=layout.old_positions,
    )
    return first, second, combine_two_probe_components(first, second), _component_cosines(first, second)


@torch.inference_mode()
def evaluate_cutover(
    *,
    edge: str,
    parent,
    current,
    history,
    uids: list[int],
    panels: np.ndarray,
    cutover: int,
    batch_size: int,
    maps: tuple[torch.Tensor, ...],
) -> tuple[list[dict], list[dict], list[dict]]:
    device = next(current.parameters()).device
    layer_rows: list[dict] = []
    score_rows: list[dict] = []
    correctness: list[dict] = []
    for start in range(0, len(uids), batch_size):
        batch_uids = uids[start : start + batch_size]
        items, behaviors, deltas, _ = _prefix_tensors(
            history, batch_uids, cutover, device
        )
        exact_cache = current.compute_kv(items, behaviors, deltas)
        reuse_cache = parent.compute_kv(items, behaviors, deltas)
        first, second, sidecar, component_cosines = _make_sidecar(
            parent_cache=reuse_cache,
            current=current,
            maps=maps,
            items=items,
            behaviors=behaviors,
            deltas=deltas,
        )
        candidates = torch.as_tensor(
            panels[start : start + len(batch_uids)], dtype=torch.long, device=device
        )
        query_deltas = torch.zeros(len(batch_uids), device=device)
        trace = trace_reader_correction(
            current, exact_cache, reuse_cache, candidates, query_deltas
        )
        zeros = torch.zeros(len(batch_uids), device=device)
        dual = progressive_corrections(sidecar, zeros)
        predictions = {
            "single_latest_probe": first.corrections,
            "dual_probe": dual,
        }
        identifiers = [f"{uid}:cutover" for uid in batch_uids]
        layer_rows.extend(
            _layer_rows(
                edge=edge,
                phase="cutover",
                identifiers=identifiers,
                uids=batch_uids,
                request_positions=["cutover"] * len(batch_uids),
                evictions=[0] * len(batch_uids),
                exact=trace.corrections["av_aggregation"],
                predictions=predictions,
                probe_cosines=sidecar.probe_direction_cosines,
                probe_norm_ratios=sidecar.probe_norm_ratios,
                component_cosines=component_cosines,
            )
        )
        predicted_scores = {}
        for method, correction in predictions.items():
            predicted_scores[method], _ = intervene_reader_correction(
                current,
                reuse_cache,
                candidates,
                query_deltas,
                stage="av_aggregation",
                corrections=correction,
            )
        score_rows.extend(
            _score_rows(
                edge=edge,
                phase="cutover",
                identifiers=identifiers,
                uids=batch_uids,
                request_positions=["cutover"] * len(batch_uids),
                evictions=[0] * len(batch_uids),
                trace=trace,
                predicted_scores=predicted_scores,
            )
        )
        correctness.append({"edge": edge, "phase": "cutover", **trace.correctness})
    return layer_rows, score_rows, correctness


@torch.inference_mode()
def evaluate_rolling_batch(
    *,
    edge: str,
    parent,
    current,
    history,
    uids: list[int],
    groups_by_user: dict[int, list[dict[str, Any]]],
    cutover: int,
    maps: tuple[torch.Tensor, ...],
) -> tuple[list[dict], list[dict], list[dict]]:
    device = next(current.parameters()).device
    items, behaviors, deltas, prefix_timestamps = _prefix_tensors(
        history, uids, cutover, device
    )
    exact_cache = current.compute_kv(items, behaviors, deltas)
    reuse_cache = parent.compute_kv(items, behaviors, deltas)
    first, second, sidecar, component_cosines = _make_sidecar(
        parent_cache=reuse_cache,
        current=current,
        maps=maps,
        items=items,
        behaviors=behaviors,
        deltas=deltas,
    )
    del second
    last_times = prefix_timestamps[:, -1].astype(np.int64, copy=True)
    append_counts = np.zeros(len(uids), dtype=np.int64)
    evictions = np.zeros(len(uids), dtype=np.int64)
    actions: list[list[tuple[str, int, Any, str]]] = []
    for index, uid in enumerate(uids):
        groups = groups_by_user[uid]
        targets = {
            int(groups[0]["query_timestamp"]): (groups[0], "first_eligible"),
            int(groups[-1]["query_timestamp"]): (groups[-1], "last_eligible"),
        }
        timestamps, event_items, event_behaviors = history.rows[uid]
        start = int(np.searchsorted(timestamps, cutover, side="left"))
        stop = int(np.searchsorted(timestamps, max(targets), side="right"))
        post_events: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for position in range(start, stop):
            post_events[int(timestamps[position])].append(
                (
                    int(timestamps[position]),
                    int(event_items[position]),
                    int(event_behaviors[position]),
                )
            )
        timeline: list[tuple[str, int, Any, str]] = []
        for timestamp in sorted(set(targets) | set(post_events)):
            if timestamp in targets:
                group, request_position = targets[timestamp]
                timeline.append(("query", timestamp, group, request_position))
            for event in sorted(
                post_events.get(timestamp, ()), key=lambda value: (value[1], value[2])
            ):
                timeline.append(("append", timestamp, event, ""))
        actions.append(timeline)

    positions = np.zeros(len(uids), dtype=np.int64)
    layer_rows: list[dict] = []
    score_rows: list[dict] = []
    correctness: list[dict] = []
    while True:
        active = [i for i in range(len(uids)) if positions[i] < len(actions[i])]
        if not active:
            break
        append_indices = [
            i for i in active if actions[i][positions[i]][0] == "append"
        ]
        if append_indices:
            events = [actions[i][positions[i]][2] for i in append_indices]
            event_items = torch.tensor(
                [[event[1]] for event in events], dtype=torch.long, device=device
            )
            event_behaviors = torch.tensor(
                [[event[2]] for event in events], dtype=torch.long, device=device
            )
            event_deltas = torch.tensor(
                [
                    [float(max(0, min(7 * DAY, event[0] - last_times[index])))]
                    for index, event in zip(append_indices, events, strict=True)
                ],
                dtype=torch.float32,
                device=device,
            )
            selected = stacked_cache(
                [
                    select_cache(exact_cache, append_indices),
                    select_cache(reuse_cache, append_indices),
                ]
            )
            updated = append_with_rolling_cap(
                current,
                selected,
                event_items.repeat(2, 1),
                event_behaviors.repeat(2, 1),
                event_deltas.repeat(2, 1),
                512,
            )
            count = len(append_indices)
            assign_cache(
                exact_cache,
                append_indices,
                HSTUKVCache(updated.k[:, :count], updated.v[:, :count], 512),
            )
            assign_cache(
                reuse_cache,
                append_indices,
                HSTUKVCache(updated.k[:, count:], updated.v[:, count:], 512),
            )
            for index, event in zip(append_indices, events, strict=True):
                last_times[index] = event[0]
                append_counts[index] += 1
                evictions[index] += 1
                positions[index] += 1

        remaining = [i for i in range(len(uids)) if positions[i] < len(actions[i])]
        query_indices = [
            i for i in remaining if actions[i][positions[i]][0] == "query"
        ]
        if not query_indices:
            continue
        entries = []
        for index in query_indices:
            _, query_time, group, request_position = actions[index][positions[index]]
            entries.append((index, query_time, group, request_position))
            positions[index] += 1
        for maximum in (16, 8, 4, 2):
            chosen = [entry for entry in entries if entry[2]["max_width"] == maximum]
            if not chosen:
                continue
            indices = [entry[0] for entry in chosen]
            groups = [entry[2] for entry in chosen]
            candidates = torch.as_tensor(
                np.stack([group["items"] for group in groups]),
                dtype=torch.long,
                device=device,
            )
            query_deltas = torch.tensor(
                [float(entry[1] - last_times[entry[0]]) for entry in chosen],
                dtype=torch.float32,
                device=device,
            )
            selected_exact = select_cache(exact_cache, indices)
            selected_reuse = select_cache(reuse_cache, indices)
            trace = trace_reader_correction(
                current,
                selected_exact,
                selected_reuse,
                candidates,
                query_deltas,
            )
            eviction_tensor = torch.as_tensor(
                evictions[indices], dtype=torch.float32, device=device
            )
            single = global_coverage_corrections(
                _select_tuple(first.corrections, indices), eviction_tensor
            )
            chosen_sidecar = _select_sidecar(sidecar, indices)
            dual_cutover = progressive_corrections(
                chosen_sidecar, torch.zeros_like(eviction_tensor)
            )
            dual_global = global_coverage_corrections(
                dual_cutover, eviction_tensor
            )
            dual_segment = progressive_corrections(chosen_sidecar, eviction_tensor)
            predictions = {
                "single_probe_global_decay": single,
                "dual_probe_global_decay": dual_global,
                "dual_probe_segment_decay": dual_segment,
            }
            identifiers = [
                f"{entry[2]['uid']}:{entry[1]}" for entry in chosen
            ]
            selected_component_cosines = component_cosines.index_select(
                0, torch.as_tensor(indices, dtype=torch.long, device=device)
            )
            selected_uids = [uids[index] for index in indices]
            selected_positions = [entry[3] for entry in chosen]
            selected_evictions = [int(evictions[index]) for index in indices]
            layer_rows.extend(
                _layer_rows(
                    edge=edge,
                    phase="rolling",
                    identifiers=identifiers,
                    uids=selected_uids,
                    request_positions=selected_positions,
                    evictions=selected_evictions,
                    exact=trace.corrections["av_aggregation"],
                    predictions=predictions,
                    probe_cosines=chosen_sidecar.probe_direction_cosines,
                    probe_norm_ratios=chosen_sidecar.probe_norm_ratios,
                    component_cosines=selected_component_cosines,
                )
            )
            predicted_scores = {}
            for method, correction in predictions.items():
                predicted_scores[method], _ = intervene_reader_correction(
                    current,
                    selected_reuse,
                    candidates,
                    query_deltas,
                    stage="av_aggregation",
                    corrections=correction,
                )
            score_rows.extend(
                _score_rows(
                    edge=edge,
                    phase="rolling",
                    identifiers=identifiers,
                    uids=selected_uids,
                    request_positions=selected_positions,
                    evictions=selected_evictions,
                    trace=trace,
                    predicted_scores=predicted_scores,
                )
            )
            correctness.append({"edge": edge, "phase": "rolling", **trace.correctness})
    return layer_rows, score_rows, correctness


def _raw_seal(directory: Path, contract_hash: str) -> dict[str, Any]:
    files = {}
    for name in ("layer_metrics.parquet", "score_metrics.parquet", "correctness.csv", "sample_manifest.parquet"):
        path = directory / name
        files[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return {
        "format": "evokv_progressive_pro_decomposition_raw_v1",
        "contract_sha256": contract_hash,
        "labels_read": False,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--scope", choices=("canary", "formal"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    output = args.output or DEFAULT_ROOT / args.scope
    if output.exists() or output.with_name(output.name + ".partial").exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if args.batch_size != 8:
        raise ValueError("the prospective decomposition freezes batch size 8")
    contract, contract_hash = verify_contract(args.contract)
    users_per_edge = (
        contract["fixed_samples"]["focused_canary"]["users_per_edge"]
        if args.scope == "canary"
        else contract["fixed_samples"]["cutover"]["users_per_edge"]
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    population = pd.read_parquet(POPULATION).sort_values(["selector_rank", "uid"])
    population_uids = [int(value) for value in population.uid]
    first_model, first_payload = load_model(checkpoint(0), device)
    oov_buckets = int(first_payload["config"]["num_items"]) - KNOWN_ITEMS
    del first_model, first_payload
    if device.type == "cuda":
        torch.cuda.empty_cache()
    history = load_histories(population_uids, oov_buckets=oov_buckets)

    layer_records: list[dict] = []
    score_records: list[dict] = []
    correctness_records: list[dict] = []
    sample_records: list[dict] = []
    started = time.time()
    for edge, cutover_day, parent_version, current_version in EDGES:
        cutover = cutover_day * DAY
        stop = (cutover_day + 14) * DAY
        _, all_items, _, _, _ = histories_at_cutover(
            history, np.asarray(population_uids, dtype=np.int64), cutover
        )
        all_panels, _, panel_audit = candidate_panel(all_items)
        cutover_uids = population_uids[:users_per_edge]
        cutover_panels = all_panels[:users_per_edge]
        real_groups = exposed_groups(REQUESTS, cutover, stop)
        rolling_uids = [
            uid for uid in population_uids if len(real_groups.get(uid, ())) >= 2
        ][:users_per_edge]
        if len(rolling_uids) != users_per_edge:
            raise RuntimeError(f"{edge} has only {len(rolling_uids)} eligible rolling users")
        target_groups = {
            uid: [real_groups[uid][0], real_groups[uid][-1]] for uid in rolling_uids
        }
        for uid in cutover_uids:
            sample_records.append(
                {"edge": edge, "phase": "cutover", "uid": uid, "selected": True}
            )
        for uid in rolling_uids:
            sample_records.append(
                {
                    "edge": edge,
                    "phase": "rolling",
                    "uid": uid,
                    "selected": True,
                    "first_query_timestamp": int(target_groups[uid][0]["query_timestamp"]),
                    "last_query_timestamp": int(target_groups[uid][-1]["query_timestamp"]),
                }
            )
        parent, _ = load_model(checkpoint(parent_version), device)
        current, _ = load_model(checkpoint(current_version), device)
        maps = parameter_cast_maps(parent, current)
        cut_layer, cut_score, cut_correctness = evaluate_cutover(
            edge=edge,
            parent=parent,
            current=current,
            history=history,
            uids=cutover_uids,
            panels=cutover_panels,
            cutover=cutover,
            batch_size=args.batch_size,
            maps=maps,
        )
        layer_records.extend(cut_layer)
        score_records.extend(cut_score)
        correctness_records.extend(cut_correctness)
        for start in range(0, len(rolling_uids), args.batch_size):
            batch_uids = rolling_uids[start : start + args.batch_size]
            roll_layer, roll_score, roll_correctness = evaluate_rolling_batch(
                edge=edge,
                parent=parent,
                current=current,
                history=history,
                uids=batch_uids,
                groups_by_user=target_groups,
                cutover=cutover,
                maps=maps,
            )
            layer_records.extend(roll_layer)
            score_records.extend(roll_score)
            correctness_records.extend(roll_correctness)
        del parent, current
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "phase": "edge_complete",
                    "edge": edge,
                    "cutover_users": len(cutover_uids),
                    "rolling_users": len(rolling_uids),
                    "panel_audit": panel_audit,
                }
            ),
            flush=True,
        )

    layer_frame = pd.DataFrame(layer_records)
    score_frame = pd.DataFrame(score_records)
    correctness_frame = pd.DataFrame(correctness_records)
    sample_frame = pd.DataFrame(sample_records)
    max_correctness = float(
        correctness_frame[
            ["native_exact", "native_reuse", "final_full_delta", "layer_stage_full_delta"]
        ].max().max()
    )
    summary = {
        "status": f"progressive_pro_decomposition_{args.scope}_raw_complete",
        "scope": args.scope,
        "contract_sha256": contract_hash,
        "labels_read": False,
        "edges": 5,
        "cutover_users_per_edge": users_per_edge,
        "rolling_users_per_edge": users_per_edge,
        "rolling_requests_per_user": 2,
        "layer_rows": len(layer_frame),
        "score_rows": len(score_frame),
        "correctness_max_abs_error": max_correctness,
        "elapsed_seconds": time.time() - started,
    }
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True)
    layer_frame.to_parquet(partial / "layer_metrics.parquet", index=False)
    score_frame.to_parquet(partial / "score_metrics.parquet", index=False)
    correctness_frame.to_csv(partial / "correctness.csv", index=False)
    sample_frame.to_parquet(partial / "sample_manifest.parquet", index=False)
    (partial / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    seal = _raw_seal(partial, contract_hash)
    (partial / "raw.seal.json").write_text(json.dumps(seal, indent=2) + "\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
