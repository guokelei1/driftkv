#!/usr/bin/env python3
"""Disentangle stale-state repair from prefix bypass under a fixed readout.

This is a structural development audit, not a request-quality evaluation.  It
uses target-free Q_main panels and preserves the temporal delta from a cached
prefix into the first appended token.  The fixed-query endpoint holds the
neutral query item, behaviour and temporal embedding constant across append
counts; only preceding current-model context changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from build_yambda_release_snapshot import prepare_catalog_and_popularity
from eval_yambda_cutover_probe_validity import EDGE
from eval_yambda_dilution_curve_v2 import build_inputs, cache_tail, js_and_overlap
from eval_yambda_multi_panel_risk import PANELS, regret
from train_yambda_theta0_medium import MAX_HISTORY, history_to_arrays, build_foundation_data
from train_yambda_two_edges import compact_history_tensors, load_checkpoint


ROOT = Path("results/data_audit/yambda50m_v2")
RAW = Path("data/raw/yambda/flat/50m/listens.parquet")
KS = (0, 1, 2, 4, 8, 16)
QUERY_DELTA = 0.0


def tensor_slice(
    context: list[tuple[int, int, int]],
    query: tuple[int, int, int],
    item_map: dict[int, int],
    device: torch.device,
    *,
    previous_timestamp: int | None = None,
    fixed_query_delta: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode context + canonical query, preserving the cache boundary delta."""
    items, behaviors, deltas = history_to_arrays(
        context + [query], item_map, previous_timestamp=previous_timestamp
    )
    if fixed_query_delta is not None:
        deltas[-1] = fixed_query_delta
    return (
        torch.from_numpy(items[None, :]).to(device),
        torch.from_numpy(behaviors[None, :]).to(device),
        torch.from_numpy(deltas[None, :]).to(device),
        torch.tensor([len(items)], dtype=torch.long, device=device),
    )


def concat(parts, index: int) -> torch.Tensor:
    return torch.cat([part[index] for part in parts], dim=0)


def pair_metrics(full: np.ndarray, other: np.ndarray) -> dict[str, np.ndarray]:
    values, _ = regret(full, other)
    js, overlap = js_and_overlap(full, other)
    score_rms = np.sqrt(np.square(full - other).mean(axis=(1, 2)))
    return {
        "mean_regret": values.mean(axis=1),
        "cvar90_regret": np.sort(values, axis=1)[:, -4:].mean(axis=1),
        "mean_js": js.mean(axis=1),
        "mean_top10_overlap_loss": overlap.mean(axis=1),
        "score_rms": score_rms,
    }


def summarize(rows: list[dict], prefix: str) -> dict:
    if not rows:
        return {"states": 0, "mean": None, "p50": None, "p95": None, "p99": None, "top10_changed_fraction": None}
    values = np.asarray([row[prefix + "mean_regret"] for row in rows], dtype=float)
    return {
        "states": len(rows),
        "mean": float(values.mean()),
        "p50": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "top10_changed_fraction": float(
            np.mean([row[prefix + "mean_top10_overlap_loss"] > 0 for row in rows])
        ),
    }


def score_paths(
    parent,
    current,
    prefix: list[list[tuple[int, int, int]]],
    suffix: list[list[tuple[int, int, int]]],
    queries: list[tuple[int, int, int]],
    candidates: torch.Tensor,
    item_map: dict[int, int],
    device: torch.device,
    *,
    fixed_query_delta: float | None,
) -> dict[str, np.ndarray]:
    """Score Full, Reuse, Suffix Only and latest-append-masked paths.

    Input rows must have equal prefix length: persistent K/V has no per-row
    valid-length mask after materialisation.
    """
    if len({len(history) for history in prefix}) != 1:
        raise ValueError("variable-length prefix cache batch")
    full_parts = [
        tensor_slice(h + s, q, item_map, device, fixed_query_delta=fixed_query_delta)
        for h, s, q in zip(prefix, suffix, queries)
    ]
    prefix_parts = [compact_history_tensors(h, item_map, device) for h in prefix]
    append_parts = [
        tensor_slice(
            s,
            q,
            item_map,
            device,
            previous_timestamp=h[-1][1],
            fixed_query_delta=fixed_query_delta,
        )
        for h, s, q in zip(prefix, suffix, queries)
    ]
    suffix_only_parts = [
        tensor_slice(s, q, item_map, device, fixed_query_delta=fixed_query_delta)
        for s, q in zip(suffix, queries)
    ]
    masked_parts = [
        tensor_slice(h + s[:-1], q, item_map, device, fixed_query_delta=fixed_query_delta)
        for h, s, q in zip(prefix, suffix, queries)
    ]
    fi, fb, fd, fl = [concat(full_parts, index) for index in range(4)]
    pi, pb, pd, pl = [concat(prefix_parts, index) for index in range(4)]
    ai, ab, ad, _ = [concat(append_parts, index) for index in range(4)]
    si, sb, sd, sl = [concat(suffix_only_parts, index) for index in range(4)]
    mi, mb, md, ml = [concat(masked_parts, index) for index in range(4)]
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        full_hidden, _ = current(fi, fb, fd, lengths=fl)
        full = current.score_candidates(full_hidden, candidates, fl)
        old = parent.compute_kv(pi, pb, pd, pl)
        reuse_hidden, _ = current.forward_with_cache(old, ai, ab, ad)
        reuse = current.score_hidden(reuse_hidden[:, -1, :], candidates)
        suffix_hidden, _ = current(si, sb, sd, lengths=sl)
        suffix_only = current.score_candidates(suffix_hidden, candidates, sl)
        masked_hidden, _ = current(mi, mb, md, lengths=ml)
        latest_masked = current.score_candidates(masked_hidden, candidates, ml)
    return {
        "full": full.float().cpu().numpy().reshape(len(prefix), PANELS, 100),
        "reuse": reuse.float().cpu().numpy().reshape(len(prefix), PANELS, 100),
        "suffix_only": suffix_only.float().cpu().numpy().reshape(len(prefix), PANELS, 100),
        "latest_append_masked": latest_masked.float().cpu().numpy().reshape(len(prefix), PANELS, 100),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge", choices=sorted(EDGE), default=None)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--subset-manifest", type=Path, default=None)
    parser.add_argument("--subset-component", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=20260819)
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    _, popular = prepare_catalog_and_popularity(RAW)
    _, _, item_map, _ = build_foundation_data(RAW, set())
    result = {
        "status": "repair_vs_bypass_fixed_query_development",
        "target_injection": False,
        "panels": "Q_main_rank_decay_v1, all 32 panels",
        "append_counts": list(KS),
        "population": "complete_case_no_physical_cap_eviction",
        "fixed_query": {
            "item_id": 0,
            "behavior": 0,
            "query_timestamp": "sixteenth post-release event + 5 seconds",
            "temporal_input_delta": QUERY_DELTA,
            "note": "The shared fixed delta prevents a query-embedding shift; append context and causal position still vary by k.",
        },
        "paths": {
            "full": "current model, pre-release prefix plus current suffix",
            "reuse": "parent prefix KV plus current suffix append",
            "suffix_only": "current model, current suffix only; alias current_full_with_old_prefix_masked",
            "latest_append_masked": "current Full with the latest current suffix event removed",
        },
        "edges": {},
    }
    edges = {args.edge: EDGE[args.edge]} if args.edge else EDGE
    for edge, (release, _, parent_path, current_path) in edges.items():
        snapshot = pq.read_table(f"data/manifests/yambda50m_v2_release_snapshot_{edge}.parquet").to_pydict()
        states = {int(uid): {key: values[index] for key, values in snapshot.items()} for index, uid in enumerate(snapshot["uid"])}
        panel_map: dict[int, list[dict]] = {}
        for row in pq.read_table(f"data/manifests/yambda50m_v2_qmain32_v2_{edge}.parquet").to_pylist():
            panel_map.setdefault(int(row["uid"]), []).append(row)
        for rows in panel_map.values():
            rows.sort(key=lambda row: row["panel_id"])
        inputs = build_inputs(edge, states, item_map, popular)
        if args.subset_manifest is not None:
            subset = pq.read_table(args.subset_manifest).to_pydict()
            selected = {
                int(uid)
                for uid, component in zip(subset["uid"], subset["component"])
                if args.subset_component is None or component == args.subset_component
            }
            inputs = {uid: value for uid, value in inputs.items() if uid in selected}
        # Keep the physical cache cap out of this first causal comparison.
        eligible = {
            uid: value
            for uid, value in inputs.items()
            if len(value[1]) >= 16 and len(value[0]) + 16 <= MAX_HISTORY
        }
        if args.sample_size is not None and len(eligible) > args.sample_size:
            rng = np.random.default_rng(args.sample_seed)
            selected = set(rng.choice(np.asarray(sorted(eligible)), size=args.sample_size, replace=False).tolist())
            eligible = {uid: value for uid, value in eligible.items() if uid in selected}
        if args.max_users is not None:
            eligible = {uid: eligible[uid] for uid in sorted(eligible)[: args.max_users]}
        parent, _ = load_checkpoint(Path(parent_path), device)
        current, _ = load_checkpoint(Path(current_path), device)
        records: list[dict] = []
        for mode in ("natural_service", "fixed_query_structural"):
            for k in KS:
                grouped: dict[int, list[tuple[int, list, list, list]]] = {}
                for uid, (prefix, all_suffix, pool) in eligible.items():
                    suffix = all_suffix[:k]
                    if mode == "natural_service":
                        timestamp = release if k == 0 else suffix[-1][1]
                        query = (0, timestamp, 0)
                        query_delta = None
                    else:
                        query = (0, all_suffix[15][1] + 5, 0)
                        query_delta = QUERY_DELTA
                    grouped.setdefault(len(prefix), []).append((uid, prefix, suffix, query))
                for group in grouped.values():
                    for start in range(0, len(group), 8):
                        chunk = group[start : start + 8]
                        uids = [row[0] for row in chunk]
                        paths = score_paths(
                            parent,
                            current,
                            [row[1] for row in chunk],
                            [row[2] for row in chunk],
                            [row[3] for row in chunk],
                            torch.tensor(
                                [[item_map[int(item)] for panel in panel_map[uid] for item in panel["candidate_item_ids"]] for uid in uids],
                                dtype=torch.long,
                                device=device,
                            ),
                            item_map,
                            device,
                            fixed_query_delta=query_delta,
                        )
                        comparisons = {
                            "stale_": pair_metrics(paths["full"], paths["reuse"]),
                            "history_": pair_metrics(paths["full"], paths["suffix_only"]),
                            "latest_mask_": pair_metrics(paths["full"], paths["latest_append_masked"]),
                        }
                        for index, uid in enumerate(uids):
                            row = {
                                "edge_id": edge,
                                "uid": uid,
                                "mode": mode,
                                "append_count": k,
                                "effective_prefix_length": len(chunk[index][1]),
                                "current_suffix_tokens": k,
                                "physical_cap_eviction": False,
                            }
                            for prefix_name, values in comparisons.items():
                                row.update({prefix_name + name: float(value[index]) for name, value in values.items()})
                            records.append(row)
        summaries = {}
        for mode in ("natural_service", "fixed_query_structural"):
            summaries[mode] = {
                str(k): {
                    "stale_error": summarize([row for row in records if row["mode"] == mode and row["append_count"] == k], "stale_"),
                    "long_history_contribution": summarize([row for row in records if row["mode"] == mode and row["append_count"] == k], "history_"),
                    "latest_append_contribution": summarize([row for row in records if row["mode"] == mode and row["append_count"] == k], "latest_mask_"),
                }
                for k in KS
            }
        result["edges"][edge] = {
            "eligible_states": len(eligible),
            "subset_manifest": None if args.subset_manifest is None else str(args.subset_manifest),
            "subset_component": args.subset_component,
            "sample_size_requested": args.sample_size,
            "sample_seed": args.sample_seed if args.sample_size is not None else None,
            "summaries": summaries,
        }
        pq.write_table(pa.Table.from_pylist(records), ROOT / f"repair_bypass_fixed_query_v1{args.output_suffix}_{edge}.parquet", compression="zstd")
    json_suffix = f"{args.output_suffix}_{args.edge}" if args.edge else args.output_suffix
    (ROOT / f"repair_bypass_fixed_query_v1{json_suffix}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
