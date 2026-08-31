#!/usr/bin/env python3
"""Raw-first C32/C48/C64 label-free fidelity frontier for one PRO."""

from __future__ import annotations

import argparse
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
from pro_lazy_cost import progressive_pro_cost  # noqa: E402
from probe_progressive_pro_decomposition import (  # noqa: E402
    _layer_rows,
    _make_sidecar,
    _prefix_tensors,
    _score_rows,
    _select_sidecar,
    sha256,
)
from probe_recommendation_state_structure import (  # noqa: E402
    KNOWN_ITEMS,
    candidate_panel,
    checkpoint,
    histories_at_cutover,
)
from progressive_pro import (  # noqa: E402
    global_coverage_corrections,
    progressive_corrections,
)
from reader_compatibility_correction import (  # noqa: E402
    intervene_reader_correction,
    trace_reader_correction,
)


CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_progressive_pro_frontier_v1.yaml"
POPULATION = ROOT / "results/yambda500m_small_seed17/insight_recommendation_state_structure_v1/population.parquet"
REQUESTS = ROOT / "data/manifests/yambda500m_small_hstu_native_rolling_matrix_fast_v3/requests_fidelity.parquet"
DEFAULT_ROOT = ROOT / "results/yambda500m_small_seed17/insight_progressive_pro_v1/frontier_v1"
EDGES = (
    ("v0_to_v1", 231, 0, 1),
    ("v1_to_v2", 245, 1, 2),
    ("v2_to_v3", 259, 2, 3),
    ("v3_to_v4", 273, 3, 4),
    ("v4_to_v5", 287, 4, 5),
)
CARRIERS = (32, 48, 64)


def verify_contract(path: Path) -> tuple[dict[str, Any], str]:
    contract = yaml.safe_load(path.read_text())
    if contract["scope"]["labels"] != "prohibited":
        raise RuntimeError("frontier contract must prohibit labels")
    for name, record in contract["frozen_inputs"].items():
        if name == "checkpoints":
            continue
        target = ROOT / record["path"]
        if sha256(target) != record["sha256"]:
            raise RuntimeError(f"frozen input differs: {name}")
        required = record.get("required_status")
        if required and json.loads(target.read_text())["status"] != required:
            raise RuntimeError(f"frozen status differs: {name}")
    for version in range(6):
        record = contract["frozen_inputs"]["checkpoints"][f"v{version}"]
        target = ROOT / record["path"]
        if target != checkpoint(version) or sha256(target) != record["sha256"]:
            raise RuntimeError(f"frozen v{version} checkpoint differs")
    for carriers in CARRIERS:
        observed = progressive_pro_cost(carriers)
        frozen = contract["one_unified_mechanism"]["carrier_axis"][f"C{carriers}"]
        if observed["total_flops_per_user"] != frozen["theoretical_flops_per_user"]:
            raise RuntimeError(f"C{carriers} cost differs from the contract")
        if abs(observed["over_full_fraction"] - frozen["fraction_of_Full"]) > 1e-15:
            raise RuntimeError(f"C{carriers} fraction differs from the contract")
    return contract, sha256(path)


def _convergence_rows(
    *,
    edge: str,
    phase: str,
    identifiers: list[str],
    uids: list[int],
    request_positions: list[str],
    corrections: dict[int, tuple[torch.Tensor, ...]],
) -> list[dict[str, Any]]:
    rows = []
    for source in (32, 48):
        for layer, (left, right) in enumerate(
            zip(corrections[source], corrections[64], strict=True)
        ):
            left_flat = left.float().flatten(1)
            right_flat = right.float().flatten(1)
            cosine = F.cosine_similarity(left_flat, right_flat, dim=1)
            ratio = left_flat.norm(dim=1) / right_flat.norm(dim=1).clamp_min(1e-12)
            relative = (left_flat - right_flat).norm(dim=1) / right_flat.norm(dim=1).clamp_min(1e-12)
            for index, identifier in enumerate(identifiers):
                rows.append(
                    {
                        "edge": edge,
                        "phase": phase,
                        "identifier": identifier,
                        "uid": int(uids[index]),
                        "request_position": request_positions[index],
                        "source_carriers": source,
                        "target_carriers": 64,
                        "layer": layer,
                        "direction_cosine": float(cosine[index]),
                        "norm_ratio": float(ratio[index]),
                        "relative_l2": float(relative[index]),
                        "labels_read": False,
                    }
                )
    return rows


@torch.inference_mode()
def evaluate_cutover(
    *, edge, parent, current, maps, history, uids, panels, cutover, batch_size
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    device = next(current.parameters()).device
    layer_rows: list[dict] = []
    score_rows: list[dict] = []
    convergence_rows: list[dict] = []
    correctness: list[dict] = []
    for start in range(0, len(uids), batch_size):
        batch_uids = uids[start : start + batch_size]
        items, behaviors, deltas, _ = _prefix_tensors(history, batch_uids, cutover, device)
        exact_cache = current.compute_kv(items, behaviors, deltas)
        reuse_cache = parent.compute_kv(items, behaviors, deltas)
        candidates = torch.as_tensor(
            panels[start : start + len(batch_uids)], dtype=torch.long, device=device
        )
        query_deltas = torch.zeros(len(batch_uids), device=device)
        trace = trace_reader_correction(current, exact_cache, reuse_cache, candidates, query_deltas)
        identifiers = [f"{uid}:cutover" for uid in batch_uids]
        request_positions = ["cutover"] * len(batch_uids)
        score_rows.extend(
            _score_rows(
                edge=edge,
                phase="cutover",
                identifiers=identifiers,
                uids=batch_uids,
                request_positions=request_positions,
                evictions=[0] * len(batch_uids),
                trace=trace,
                predicted_scores={},
            )
        )
        corrections: dict[int, tuple[torch.Tensor, ...]] = {}
        for carriers in CARRIERS:
            first, second, sidecar, component_cosines = _make_sidecar(
                parent_cache=reuse_cache,
                current=current,
                maps=maps,
                items=items,
                behaviors=behaviors,
                deltas=deltas,
                carrier_count=carriers,
            )
            del first, second
            correction = progressive_corrections(
                sidecar, torch.zeros(len(batch_uids), device=device)
            )
            corrections[carriers] = correction
            method = f"C{carriers}"
            layer_rows.extend(
                _layer_rows(
                    edge=edge,
                    phase="cutover",
                    identifiers=identifiers,
                    uids=batch_uids,
                    request_positions=request_positions,
                    evictions=[0] * len(batch_uids),
                    exact=trace.corrections["av_aggregation"],
                    predictions={method: correction},
                    probe_cosines=sidecar.probe_direction_cosines,
                    probe_norm_ratios=sidecar.probe_norm_ratios,
                    component_cosines=component_cosines,
                )
            )
            scores, _ = intervene_reader_correction(
                current,
                reuse_cache,
                candidates,
                query_deltas,
                stage="av_aggregation",
                corrections=correction,
            )
            score_rows.extend(
                [
                    row
                    for row in _score_rows(
                        edge=edge,
                        phase="cutover",
                        identifiers=identifiers,
                        uids=batch_uids,
                        request_positions=request_positions,
                        evictions=[0] * len(batch_uids),
                        trace=trace,
                        predicted_scores={method: scores},
                    )
                    if row["method"] == method
                ]
            )
        convergence_rows.extend(
            _convergence_rows(
                edge=edge,
                phase="cutover",
                identifiers=identifiers,
                uids=batch_uids,
                request_positions=request_positions,
                corrections=corrections,
            )
        )
        correctness.append({"edge": edge, "phase": "cutover", **trace.correctness})
    return layer_rows, score_rows, convergence_rows, correctness


@torch.inference_mode()
def evaluate_rolling_batch(
    *, edge, parent, current, maps, history, uids, groups_by_user, cutover
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    device = next(current.parameters()).device
    items, behaviors, deltas, prefix_timestamps = _prefix_tensors(history, uids, cutover, device)
    exact_cache = current.compute_kv(items, behaviors, deltas)
    reuse_cache = parent.compute_kv(items, behaviors, deltas)
    sidecars = {}
    component_cosines = {}
    for carriers in CARRIERS:
        first, second, sidecar, component = _make_sidecar(
            parent_cache=reuse_cache,
            current=current,
            maps=maps,
            items=items,
            behaviors=behaviors,
            deltas=deltas,
            carrier_count=carriers,
        )
        del first, second
        sidecars[carriers] = sidecar
        component_cosines[carriers] = component

    last_times = prefix_timestamps[:, -1].astype(np.int64, copy=True)
    evictions = np.zeros(len(uids), dtype=np.int64)
    actions: list[list[tuple[str, int, Any, str]]] = []
    for uid in uids:
        groups = groups_by_user[uid]
        targets = {
            int(groups[0]["query_timestamp"]): (groups[0], "first_eligible"),
            int(groups[-1]["query_timestamp"]): (groups[-1], "last_eligible"),
        }
        timestamps, event_items, event_behaviors = history.rows[uid]
        start = int(np.searchsorted(timestamps, cutover, side="left"))
        stop = int(np.searchsorted(timestamps, max(targets), side="right"))
        events_at: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for position in range(start, stop):
            events_at[int(timestamps[position])].append(
                (int(timestamps[position]), int(event_items[position]), int(event_behaviors[position]))
            )
        timeline: list[tuple[str, int, Any, str]] = []
        for timestamp in sorted(set(targets) | set(events_at)):
            if timestamp in targets:
                group, request_position = targets[timestamp]
                timeline.append(("query", timestamp, group, request_position))
            for event in sorted(events_at.get(timestamp, ()), key=lambda value: (value[1], value[2])):
                timeline.append(("append", timestamp, event, ""))
        actions.append(timeline)

    positions = np.zeros(len(uids), dtype=np.int64)
    layer_rows: list[dict] = []
    score_rows: list[dict] = []
    convergence_rows: list[dict] = []
    correctness: list[dict] = []
    while True:
        active = [index for index in range(len(uids)) if positions[index] < len(actions[index])]
        if not active:
            break
        append_indices = [index for index in active if actions[index][positions[index]][0] == "append"]
        if append_indices:
            events = [actions[index][positions[index]][2] for index in append_indices]
            event_items = torch.tensor([[event[1]] for event in events], dtype=torch.long, device=device)
            event_behaviors = torch.tensor([[event[2]] for event in events], dtype=torch.long, device=device)
            event_deltas = torch.tensor(
                [[float(max(0, min(7 * DAY, event[0] - last_times[index])))] for index, event in zip(append_indices, events, strict=True)],
                dtype=torch.float32,
                device=device,
            )
            selected = stacked_cache([select_cache(exact_cache, append_indices), select_cache(reuse_cache, append_indices)])
            updated = append_with_rolling_cap(
                current,
                selected,
                event_items.repeat(2, 1),
                event_behaviors.repeat(2, 1),
                event_deltas.repeat(2, 1),
                512,
            )
            count = len(append_indices)
            assign_cache(exact_cache, append_indices, HSTUKVCache(updated.k[:, :count], updated.v[:, :count], 512))
            assign_cache(reuse_cache, append_indices, HSTUKVCache(updated.k[:, count:], updated.v[:, count:], 512))
            for index, event in zip(append_indices, events, strict=True):
                last_times[index] = event[0]
                evictions[index] += 1
                positions[index] += 1

        remaining = [index for index in range(len(uids)) if positions[index] < len(actions[index])]
        query_indices = [index for index in remaining if actions[index][positions[index]][0] == "query"]
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
            candidates = torch.as_tensor(np.stack([group["items"] for group in groups]), dtype=torch.long, device=device)
            query_deltas = torch.tensor(
                [float(entry[1] - last_times[entry[0]]) for entry in chosen],
                dtype=torch.float32,
                device=device,
            )
            selected_exact = select_cache(exact_cache, indices)
            selected_reuse = select_cache(reuse_cache, indices)
            trace = trace_reader_correction(current, selected_exact, selected_reuse, candidates, query_deltas)
            identifiers = [f"{entry[2]['uid']}:{entry[1]}" for entry in chosen]
            selected_uids = [uids[index] for index in indices]
            request_positions = [entry[3] for entry in chosen]
            selected_evictions = [int(evictions[index]) for index in indices]
            eviction_tensor = torch.as_tensor(selected_evictions, dtype=torch.float32, device=device)
            score_rows.extend(
                _score_rows(
                    edge=edge,
                    phase="rolling",
                    identifiers=identifiers,
                    uids=selected_uids,
                    request_positions=request_positions,
                    evictions=selected_evictions,
                    trace=trace,
                    predicted_scores={},
                )
            )
            corrections = {}
            for carriers in CARRIERS:
                selected_sidecar = _select_sidecar(sidecars[carriers], indices)
                correction = global_coverage_corrections(
                    progressive_corrections(selected_sidecar, torch.zeros_like(eviction_tensor)),
                    eviction_tensor,
                )
                corrections[carriers] = correction
                method = f"C{carriers}"
                selected_component = component_cosines[carriers].index_select(
                    0, torch.as_tensor(indices, dtype=torch.long, device=device)
                )
                layer_rows.extend(
                    _layer_rows(
                        edge=edge,
                        phase="rolling",
                        identifiers=identifiers,
                        uids=selected_uids,
                        request_positions=request_positions,
                        evictions=selected_evictions,
                        exact=trace.corrections["av_aggregation"],
                        predictions={method: correction},
                        probe_cosines=selected_sidecar.probe_direction_cosines,
                        probe_norm_ratios=selected_sidecar.probe_norm_ratios,
                        component_cosines=selected_component,
                    )
                )
                scores, _ = intervene_reader_correction(
                    current,
                    selected_reuse,
                    candidates,
                    query_deltas,
                    stage="av_aggregation",
                    corrections=correction,
                )
                score_rows.extend(
                    [
                        row
                        for row in _score_rows(
                            edge=edge,
                            phase="rolling",
                            identifiers=identifiers,
                            uids=selected_uids,
                            request_positions=request_positions,
                            evictions=selected_evictions,
                            trace=trace,
                            predicted_scores={method: scores},
                        )
                        if row["method"] == method
                    ]
                )
            convergence_rows.extend(
                _convergence_rows(
                    edge=edge,
                    phase="rolling",
                    identifiers=identifiers,
                    uids=selected_uids,
                    request_positions=request_positions,
                    corrections=corrections,
                )
            )
            correctness.append({"edge": edge, "phase": "rolling", **trace.correctness})
    return layer_rows, score_rows, convergence_rows, correctness


def _raw_seal(directory: Path, contract_hash: str) -> dict[str, Any]:
    names = (
        "layer_metrics.parquet",
        "score_metrics.parquet",
        "convergence_metrics.parquet",
        "correctness.csv",
        "sample_manifest.parquet",
        "cost.json",
    )
    return {
        "format": "evokv_progressive_pro_frontier_raw_v1",
        "contract_sha256": contract_hash,
        "labels_read": False,
        "files": {
            name: {"sha256": sha256(directory / name), "bytes": (directory / name).stat().st_size}
            for name in names
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--scope", choices=("canary", "formal"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size != 8:
        raise ValueError("the frontier freezes batch size 8")
    output = args.output or DEFAULT_ROOT / args.scope
    if output.exists() or output.with_name(output.name + ".partial").exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    contract, contract_hash = verify_contract(args.contract)
    count = contract["held_out_label_free_samples"]["canary"]["users_per_edge"] if args.scope == "canary" else contract["held_out_label_free_samples"]["cutover"]["users_per_edge"]
    offset = int(contract["held_out_label_free_samples"]["cutover"]["population_offset"])
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
    convergence_records: list[dict] = []
    correctness_records: list[dict] = []
    sample_records: list[dict] = []
    started = time.time()
    for edge, cutover_day, parent_version, current_version in EDGES:
        cutover = cutover_day * DAY
        stop = (cutover_day + 14) * DAY
        _, all_items, _, _, _ = histories_at_cutover(
            history, np.asarray(population_uids, dtype=np.int64), cutover
        )
        all_panels, _, _ = candidate_panel(all_items)
        cutover_uids = population_uids[offset : offset + count]
        cutover_panels = all_panels[offset : offset + count]
        real_groups = exposed_groups(REQUESTS, cutover, stop)
        eligible = [uid for uid in population_uids if len(real_groups.get(uid, ())) >= 2]
        rolling_uids = eligible[offset : offset + count]
        if len(cutover_uids) != count or len(rolling_uids) != count:
            raise RuntimeError(f"{edge} lacks the held-out frontier population")
        target_groups = {uid: [real_groups[uid][0], real_groups[uid][-1]] for uid in rolling_uids}
        sample_records.extend({"edge": edge, "phase": "cutover", "uid": uid} for uid in cutover_uids)
        sample_records.extend(
            {
                "edge": edge,
                "phase": "rolling",
                "uid": uid,
                "first_query_timestamp": int(target_groups[uid][0]["query_timestamp"]),
                "last_query_timestamp": int(target_groups[uid][-1]["query_timestamp"]),
            }
            for uid in rolling_uids
        )
        parent, _ = load_model(checkpoint(parent_version), device)
        current, _ = load_model(checkpoint(current_version), device)
        maps = parameter_cast_maps(parent, current)
        values = evaluate_cutover(
            edge=edge,
            parent=parent,
            current=current,
            maps=maps,
            history=history,
            uids=cutover_uids,
            panels=cutover_panels,
            cutover=cutover,
            batch_size=args.batch_size,
        )
        layer_records.extend(values[0]); score_records.extend(values[1]); convergence_records.extend(values[2]); correctness_records.extend(values[3])
        for start in range(0, len(rolling_uids), args.batch_size):
            batch_uids = rolling_uids[start : start + args.batch_size]
            values = evaluate_rolling_batch(
                edge=edge,
                parent=parent,
                current=current,
                maps=maps,
                history=history,
                uids=batch_uids,
                groups_by_user=target_groups,
                cutover=cutover,
            )
            layer_records.extend(values[0]); score_records.extend(values[1]); convergence_records.extend(values[2]); correctness_records.extend(values[3])
        del parent, current
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({"phase": "edge_complete", "edge": edge, "users": count}), flush=True)

    layer = pd.DataFrame(layer_records)
    score = pd.DataFrame(score_records)
    convergence = pd.DataFrame(convergence_records)
    correctness = pd.DataFrame(correctness_records)
    samples = pd.DataFrame(sample_records)
    summary = {
        "status": f"progressive_pro_frontier_{args.scope}_raw_complete",
        "scope": args.scope,
        "contract_sha256": contract_hash,
        "labels_read": False,
        "edges": 5,
        "users_per_edge_phase": count,
        "rolling_requests_per_user": 2,
        "carriers": list(CARRIERS),
        "layer_rows": len(layer),
        "score_rows": len(score),
        "convergence_rows": len(convergence),
        "correctness_max_abs_error": float(
            correctness[["native_exact", "native_reuse", "final_full_delta", "layer_stage_full_delta"]].max().max()
        ),
        "elapsed_seconds": time.time() - started,
    }
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True)
    layer.to_parquet(partial / "layer_metrics.parquet", index=False)
    score.to_parquet(partial / "score_metrics.parquet", index=False)
    convergence.to_parquet(partial / "convergence_metrics.parquet", index=False)
    correctness.to_csv(partial / "correctness.csv", index=False)
    samples.to_parquet(partial / "sample_manifest.parquet", index=False)
    (partial / "cost.json").write_text(
        json.dumps([progressive_pro_cost(carriers) for carriers in CARRIERS], indent=2) + "\n"
    )
    (partial / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (partial / "raw.seal.json").write_text(json.dumps(_raw_seal(partial, contract_hash), indent=2) + "\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
