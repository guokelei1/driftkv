#!/usr/bin/env python3
"""Registered behavior-triggered, causal next-item recommendation workloads.

This is a different task from the original feedback-manifest AUC. Each round
scores the selected next observed known item and deterministic uniform negative
candidates. History excludes every behavior at the query timestamp. Trigger
settings are fixed before any new candidate quality is measured.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from hstu_kvcache.methods.cachetide.cost import (
    CONFIG, PANEL, REFERENCE, SHARED_COST, _hash, price, read, simulate_user,
)
from hstu_kvcache.evaluation.release_context import RELEASES, END

STRIDES = (8, 16, 32)
CANDIDATES = (2, 16)
NAMESPACE = "query_relay_behavior_rounds_v1_seed17"


def trigger_indices(events, *, stride, known_vocab_size, releases=RELEASES, end=END):
    """First behavior per release, then every stride behaviors from that point.

    Release resets define a returning-session probe: the first actual behavior
    triggers the initial recommendation. No query is created for idle users.
    Other settings remain available as sensitivity, not selected by recovery.
    Unknown targets or targets without a strictly earlier behavior are skipped.
    """
    times, items = events[:2]
    result = {}
    for j, (version, at) in enumerate(releases):
        until = releases[j + 1][1] if j + 1 < len(releases) else end
        first = int(np.searchsorted(times, at, side="left"))
        stop = int(np.searchsorted(times, until, side="left"))
        indices = np.arange(first, stop, stride, dtype=np.int64)
        if len(indices):
            known = (items[indices] > 0) & (items[indices] < known_vocab_size)
            previous = np.searchsorted(times, times[indices], side="left") > 0
            indices = indices[known & previous]
        result[version] = indices
    return result


def compact_rounds(events, *, stride, candidates, known_vocab_size):
    times = events[0]
    indices = trigger_indices(events, stride=stride, known_vocab_size=known_vocab_size)
    return [dict(version=version, query_timestamp=int(times[index]), candidates=candidates)
            for version, selected in indices.items() for index in selected]


def candidate_rows(uid, events, *, stride=16, candidates=16, known_vocab_size,
                   releases=RELEASES, end=END):
    """Evaluation rows sharing the old query evaluator's minimal schema."""
    if candidates < 2 or candidates > known_vocab_size - 1:
        raise ValueError("sampled rounds need one positive and distinct negatives")
    times, items = events[:2]
    rows = []
    selected = trigger_indices(events, stride=stride, known_vocab_size=known_vocab_size,
                               releases=releases, end=end)
    for version, indices in selected.items():
        for index in indices:
            target = int(items[index])
            key = f"{NAMESPACE}/{int(uid)}/{version}/{int(index)}"
            seed = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little")
            rng = np.random.default_rng(seed)
            negatives = []
            seen = {target}
            while len(negatives) < candidates - 1:
                candidate = int(rng.integers(1, known_vocab_size))
                if candidate not in seen:
                    negatives.append(candidate)
                    seen.add(candidate)
            round_id = f"{NAMESPACE}:s{stride}:n{candidates}:{uid}:{version}:{index}"
            for offset, item in enumerate([target, *negatives]):
                rows.append(dict(request_id=f"{round_id}:{offset}", round_id=round_id,
                    uid=int(uid), query_timestamp=int(times[index]), item_idx=item,
                    version=version, edge=f"v{int(version[1:]) - 1}_to_{version}",
                    label=int(offset == 0)))
    return rows


def _native_totals(counts, *, hidden_size=192, layers=6, heads=6, cap=1024):
    from hstu_kvcache.evaluation.cost import CostModel, eager_read
    cm = {name: CostModel(hidden_size, layers, heads, name) for name in ("torch", "triton")}
    totals = dict(native_append_flops=0, native_query_flops=0, once_rebuild_flops=0,
                  all_resident_rebuild_flops=0)
    for key, count in counts.items():
        if key.startswith("native_append_shape_"):
            length, width = map(int, key.removeprefix("native_append_shape_").split(":"))
            model = cm["triton" if length >= cap else "torch"]
            totals["native_append_flops"] += count * model.band_append(length, width, window_size=cap)
        elif key.startswith("native_query_shape_"):
            length, width = map(int, key.removeprefix("native_query_shape_").split(":"))
            totals["native_query_flops"] += count * eager_read(cm["torch"], length, queries=width)
        elif key.startswith("once_rebuild_length_"):
            length = int(key.removeprefix("once_rebuild_length_"))
            totals["once_rebuild_flops"] += count * cm["triton" if length >= cap else "torch"].full_cache(length)
    return totals


def _ledger_from_counts(counts, original, *, stride, candidates):
    native = {v: _native_totals(c) for v, c in counts.items()}
    return dict(status="complete_cpu_workload_shapes", users=original["users"],
        requests=sum(c["ranking_queries"] for c in counts.values()),
        recommendation_rounds=sum(c["ranking_rounds"] for c in counts.values()),
        workload=dict(stride=stride, candidates_per_round=candidates,
            trigger="first actual behavior per release, then every stride behaviors; query before behavior",
            positives="observed known next item", negatives="uniform known IDs; exclude target only",
            scope="Synthetic next-item candidate task; no actual feedback-negative or industrial traffic claim"),
        model_config=original["model_config"], by_release={v: dict(c) for v, c in counts.items()},
        native_by_release=native,
        once_rebuild_flops=sum(v["once_rebuild_flops"] for v in native.values()),
        all_resident_eager_rebuild_flops=original["all_resident_eager_rebuild_flops"],
        catalogue_items=original["catalogue_items"],
        prerelease_resident_distinct_items=original["prerelease_resident_distinct_items"],
        observed_distinct_items_whole_ledger=original["observed_distinct_items_whole_ledger"],
        original_feedback_boundary=str(ROOT / "results/query_relay_2026_10/cost/event_ledger.json"))


def build_cost_ledgers(output, *, threads=4, gate_normalization="rms", feature_mode="native"):
    import pyarrow.parquet as pq
    from hstu_kvcache.evaluation.history import load_histories_semijoin

    began = time.perf_counter()
    original = read(ROOT / "results/query_relay_2026_10/cost/event_ledger.json")
    configuration, panel = read(CONFIG), read(PANEL)
    dataset_path = ROOT / configuration["data"]["dataset"]["path"]
    population_path = ROOT / configuration["data"]["population"]["path"]
    dataset = read(dataset_path)
    uids = sorted(pq.read_table(population_path, columns=["uid"])["uid"].to_pylist())
    if len(uids) != 30000 or len(set(uids)) != 30000:
        raise RuntimeError("fixed30k population changed")
    # This lightweight registration is written before reading candidate scores.
    output.mkdir(parents=True, exist_ok=True)
    registered = dict(status="registered_before_quality", primary=dict(stride=16, candidates=16),
        sensitivity=dict(strides=list(STRIDES), candidates_per_round=list(CANDIDATES)),
        selection="All six settings reported. Default16/16 chosen before new quality, representing moderate behavior batching and a 16-item sampled shortlist.",
        builder_provenance="Original builder groups real feedback rows; it has no request-stride parameter. This workload changes the quality task and population activity. Original feedback evaluation remains a boundary.",
        source_hashes={str(path.relative_to(ROOT)): _hash(path) for path in (
            Path(__file__), ROOT / "src/hstu_kvcache/data/yambda/request_manifest.py",
            dataset_path, population_path, CONFIG, PANEL)})
    (output / "registration.json").write_text(json.dumps(registered, indent=2) + "\n")
    print(json.dumps(dict(status="loading_cpu_histories", users=len(uids))), flush=True)
    histories = load_histories_semijoin(uids, dataset_path=dataset_path,
        known_vocab_size=dataset["foundation_items"],
        oov_buckets=panel["model_config"]["num_items"] - dataset["foundation_items"],
        start_timestamp=RELEASES[0][1], end_timestamp=END, max_history=1024, threads=threads)
    totals = {(stride, candidates): {v: Counter() for v, _ in RELEASES}
              for stride in STRIDES for candidates in CANDIDATES}
    for number, uid in enumerate(uids):
        events = histories.rows.get(uid, (np.array([], dtype=np.int64), np.array([], dtype=np.int64)))
        for stride in STRIDES:
            indices = trigger_indices(events, stride=stride, known_vocab_size=dataset["foundation_items"])
            # Writer bands and all recommendation lengths are independent of
            # candidate count; count them once, scale candidate-specific parts.
            compact = [dict(version=v, query_timestamp=int(events[0][i]), candidates=1)
                       for v, selected in indices.items() for i in selected]
            base = simulate_user(events, compact, RELEASES, END)
            for candidates in CANDIDATES:
                for version, counts in base.items():
                    c = Counter(counts)
                    c["ranking_rounds"] = len(indices[version])
                    for key in ("ranking_queries", "ranking_dense_pairs", "ranking_prefix_pairs",
                                "ranking_nonzero_prefix_queries"):
                        c[key] *= candidates
                    # Same-timestamp rounds can be grouped by the native scorer;
                    # replace its one-candidate batch histogram exactly.
                    for key in tuple(c):
                        if key.startswith("native_query_shape_"):
                            del c[key]
                    by_time = Counter(int(events[0][i]) for i in indices[version])
                    times = events[0]
                    for timestamp, rounds in by_time.items():
                        length = min(1024, int(np.searchsorted(times, timestamp, side="left")))
                        queries = rounds * candidates
                        for offset in range(0, queries, 32):
                            width = min(32, queries - offset)
                            c[f"native_query_shape_{length}:{width}"] += 1
                    totals[stride, candidates][version].update(c)
        if (number + 1) % 5000 == 0:
            print(json.dumps(dict(status="counting_registered_workloads", users=number + 1,
                                  elapsed_seconds=time.perf_counter() - began)), flush=True)
    historical = read(SHARED_COST)["shared12_private6"]
    reports = {}
    for (stride, candidates), counts in totals.items():
        key = f"stride{stride}_candidates{candidates}"
        ledger = _ledger_from_counts(counts, original, stride=stride, candidates=candidates)
        for version, c in counts.items():
            if c["appends"] != original["by_release"][version]["appends"]:
                raise RuntimeError("new workload dropped real behavior writes")
        (output / f"{key}_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
        scope_reports = {}
        for rank in (4, 6, 8, 12, 18):
            report = price(ledger, branch="B", xi_width=3, head_rank=rank,
                gate_normalization=gate_normalization,
                feature_mode=feature_mode,
                calibration_flops=historical["shared_calibration_flops"],
                source_reconstruction_flops=historical["source_reconstruction_sensitivity_flops"])
            # Avoid duplicating full counts in every rank sensitivity.
            report.pop("by_release")
            scope_reports[str(rank)] = report
        reports[key] = dict(workload=ledger["workload"], recommendation_rounds=ledger["recommendation_rounds"],
            candidate_requests=ledger["requests"], once_rebuild_flops=ledger["once_rebuild_flops"],
            price_by_head_rank=scope_reports)
    result = dict(status="complete_cpu_prefilter", registration=registered, workloads=reports,
        source_hashes={str(Path(__file__).relative_to(ROOT)): _hash(Path(__file__)),
                      str((ROOT / "src/hstu_kvcache/methods/cachetide/cost.py").relative_to(ROOT)):
                          _hash(ROOT / "src/hstu_kvcache/methods/cachetide/cost.py")},
        elapsed_seconds=time.perf_counter() - began,
        notes=["All true behavior appends retained. Writer matrix shapes change when recommendation triggers split append bands.",
               "Rebuild denominator counted afresh at each user's first real recommendation round per release.",
               "Original feedback AUC and new uniform-negative next-item AUC are different tasks; no recovery comparison across tasks is valid.",
               "Historical calibration is a cost sensitivity. Reduced read ranks are unimplemented architecture estimates, not quality evidence."])
    (output / "cost_sensitivity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: dict(rounds=value["recommendation_rounds"], candidates=value["candidate_requests"],
        denominator_tf=value["once_rebuild_flops"] / 1e12,
        cost_percent_by_rank={r: p["complete_percent_of_once_rebuild"]
                             for r, p in value["price_by_head_rank"].items()})
        for key, value in reports.items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/query_relay_2026_10/workloads")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reprice", action="store_true", help="reuse retained CPU event shapes")
    parser.add_argument("--gate-normalization", choices=("raw", "rms"), default="rms")
    parser.add_argument("--feature-mode", choices=("native", "origin_encoder"), default="native")
    args = parser.parse_args()
    if args.reprice:
        path = args.output / "cost_sensitivity.json"
        summary = read(path)
        historical = read(SHARED_COST)["shared12_private6"]
        for key, workload in summary["workloads"].items():
            ledger = read(args.output / f"{key}_ledger.json")
            for rank in workload["price_by_head_rank"]:
                report = price(ledger, branch="B", xi_width=3, head_rank=int(rank),
                    gate_normalization=args.gate_normalization,
                    feature_mode=args.feature_mode,
                    calibration_flops=historical["shared_calibration_flops"],
                    source_reconstruction_flops=historical["source_reconstruction_sensitivity_flops"])
                report.pop("by_release")
                workload["price_by_head_rank"][rank] = report
            broadcast = price(ledger, branch="B", xi_width=3, head_rank=6,
                gate_normalization=args.gate_normalization,
                feature_mode=args.feature_mode,
                preparation_mode="compile_once_broadcast", replicas=4,
                calibration_flops=historical["shared_calibration_flops"],
                source_reconstruction_flops=historical["source_reconstruction_sensitivity_flops"])
            broadcast.pop("by_release")
            workload["once_compile_broadcast_sensitivity_rank6"] = broadcast
        summary["pricing_source_sha256"] = _hash(ROOT / "src/hstu_kvcache/methods/cachetide/cost.py")
        path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(dict(status="repriced_existing_workload_shapes", path=str(path))), flush=True)
    else:
        build_cost_ledgers(args.output, threads=args.threads,
                           gate_normalization=args.gate_normalization,
                           feature_mode=args.feature_mode)


if __name__ == "__main__":
    main()
