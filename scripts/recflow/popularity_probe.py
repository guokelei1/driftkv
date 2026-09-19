#!/usr/bin/env python3
"""Development-only temporal count baseline, with one unit per known request."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hstu_kvcache.recflow.data import PreparedRecFlow  # noqa: E402
from hstu_kvcache.recflow.metrics import (  # noqa: E402
    aggregate_metrics,
    request_metrics,
    sample_candidates,
)


def main():
    started = time.monotonic()
    output = ROOT / "results/recflow/development/popularity_temporal_k1m"
    if (output / "summary.json").exists():
        raise SystemExit("Existing evidence found; inspect before replacing.")
    data = PreparedRecFlow(ROOT / "data/processed/recflow_v1")
    cohort = data.cohort(limit=512)
    catalog = data.catalog
    selected = np.lexsort((catalog["raw_item_ids"], -catalog["development_exposure_count"]))[:1_000_000]
    selected.sort()
    raw_ids = catalog["raw_item_ids"][selected]
    catalog_set = set(map(int, raw_ids))
    mapping = np.full(data.catalog_size + 2, -1, dtype=np.int32)
    mapping[selected + 1] = np.arange(len(selected))
    sampling_weights = catalog["development_effective_count"][selected]

    def count_requests(first, last):
        indices = data.request_indices(first, last, uids=cohort, limit=150_000)
        counts = np.zeros(len(selected), dtype=np.float64)
        known_requests = 0
        known_targets = 0
        for index in indices:
            _, items = data.targets(index)
            local = mapping[items]
            local = local[local >= 0]
            if len(local):
                counts[local] += 1.0 / len(local)
                known_requests += 1
                known_targets += len(local)
        return counts, {
            "day_start": first, "day_end": last, "request_limit": 150_000,
            "selected_positive_requests": len(indices), "known_positive_requests": known_requests,
            "known_positive_targets": known_targets,
            "request_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        }

    initial, initial_budget = count_requests(1, 18)
    recent, recent_budget = count_requests(19, 21)
    scores = {"initial": initial, "cumulative": initial + recent, "recent_only": recent}
    rankings = {name: raw_ids[np.lexsort((raw_ids, -score))[:100]].tolist() for name, score in scores.items()}
    print(json.dumps({"initial": initial_budget, "update": recent_budget}), flush=True)

    def evaluate(indices, sampled):
        metric_rows = {name: {"free_top100": []} for name in scores}
        uids = data.requests["uid"][indices]
        for number, index in enumerate(indices):
            positives, _ = data.targets(index)
            for name, ranking in rankings.items():
                metric_rows[name]["free_top100"].append(request_metrics(ranking, positives, catalog_set))
            if sampled:
                for strategy in ("uniform", "popularity"):
                    panel = sample_candidates(raw_ids, positives, 1000, 17,
                        int(data.requests[index]["request_id"]),
                        sampling_weights if strategy == "popularity" else None)
                    local = np.searchsorted(raw_ids, panel)
                    for name, score in scores.items():
                        order = np.lexsort((panel, -score[local]))
                        metric_rows[name].setdefault(f"{strategy}_1000", []).append(
                            request_metrics(panel[order], positives, catalog_set))
            if (number + 1) % 128 == 0 and sampled:
                print(json.dumps({"sampled_requests": number + 1, "seconds": time.monotonic() - started}), flush=True)
        return {
            "requests": len(indices), "request_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
            "strategies": {name: {panel: aggregate_metrics(rows, uids) for panel, rows in panels.items()}
                           for name, panels in metric_rows.items()},
        }

    panel = data.request_indices(22, 22, uids=cohort, limit=512)
    panel_result = evaluate(panel, sampled=True)
    all_result = evaluate(data.request_indices(22, 22, uids=cohort), sampled=False)
    result = {
        "scope": "CPU development-only temporal learnability diagnostic; no neural training, admission or final-user evaluation.",
        "catalog_items": len(raw_ids), "cohort_users": len(cohort),
        "cohort_sha256": hashlib.sha256(cohort.tobytes()).hexdigest(),
        "catalog_sha256": hashlib.sha256(raw_ids.tobytes()).hexdigest(),
        "catalog_selection": "Top 1M initial D1-18 development exposure counts; raw video ID breaks ties.",
        "count_objective": "Each request with known positive items contributes one unit divided equally among those distinct known positives. Unknown positives remain in evaluation denominators.",
        "ranking": "Count descending; raw video ID ascending for label-free tie breaking. Top100 generated from the full fixed 1M-item pilot catalog.",
        "sampled_panels": "512 fixed D22 requests, all known positives plus 1000 uniform or initial-development-effective-count+1 distractors; candidate seed17, shared across strategies. These are sampled ranking diagnostics, not full-catalog retrieval.",
        "budget_note": "Counts use every known positive in each selected request exactly once, equal to the expectation of one uniformly drawn known positive per request. Unlike minibatch neural training, there are no SGD steps, optimizer state, repeated draws, learned history, or beam approximation; initial/cumulative/recent have different temporal data budgets. This diagnoses available temporal signal only.",
        "initial_budget": initial_budget, "update_budget": recent_budget,
        "fixed_512_request_panel": panel_result, "all_day22_positive_requests": all_result,
        "top100_raw_ids": rankings,
        "elapsed_seconds": time.monotonic() - started,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"seconds": result["elapsed_seconds"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
