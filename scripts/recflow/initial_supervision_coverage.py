#!/usr/bin/env python3
"""Count initial positive-pool coverage of fixed RecFlow future labels on CPU."""

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


def digest(values):
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    run = ROOT / "results/recflow/development/window_6l_full_seed17"
    output = run / "initial_supervision_coverage.json"
    if output.exists():
        raise SystemExit("Preserve existing coverage evidence; inspect before replacing.")
    a_summary = run / "A/summary.json"
    saved = json.loads(a_summary.read_text())
    config = saved["configuration"]
    data = PreparedRecFlow(config["data"])
    catalog = data.catalog
    selected = np.lexsort((catalog["raw_item_ids"], -catalog["development_exposure_count"]))[:1_000_000]
    selected.sort()
    known = np.zeros(data.catalog_size + 2, dtype=bool)
    known[selected + 1] = True
    assert digest(catalog["raw_item_ids"][selected]) == config["catalog_sha256"]
    cohort512, cohort2048 = data.cohort(limit=512), data.cohort(limit=2048)
    assert np.array_equal(cohort512, cohort2048[:512])
    assert digest(cohort512) == config["cohort_sha256"]
    capacity_path = ROOT / "results/recflow/development/training_capacity_seed17/summary.json"
    capacity = json.loads(capacity_path.read_text())

    def target_items(indices):
        request = data.requests[indices]
        lengths = request["positive_stop"] - request["positive_start"]
        flat = np.repeat(request["positive_start"] - np.r_[0, np.cumsum(lengths[:-1])], lengths)
        flat += np.arange(lengths.sum())
        items = data.positives["item"][flat]
        return items, items[known[items]]

    initial, positive_sets = {}, {}
    for count, cohort in ((512, cohort512), (2048, cohort2048)):
        indices = data.request_indices(1, 18, uids=cohort)
        all_items, known_items = target_items(indices)
        unique = np.unique(known_items)
        raw_unique = catalog["raw_item_ids"][unique - 1]
        positive_sets[count] = unique
        reference = capacity["cohorts"][str(count)]["initial_unique_known_positive_videos"]
        assert len(unique) == reference["count"] and digest(raw_unique) == reference["raw_ids_sha256"]
        initial[str(count)] = {
            "cohort_users": count, "cohort_sha256": digest(cohort), "training_days": [1, 18],
            "positive_requests": len(indices), "request_indices_sha256": digest(indices),
            "all_positive_occurrences": len(all_items), "known_positive_occurrences": len(known_items),
            "unique_known_positive_videos": len(unique), "unique_raw_ids_sha256": digest(raw_unique),
            "matches_prior_capacity_audit": True,
        }
    future = {}
    for day in (19, 20, 21):
        indices = data.request_indices(day, day, uids=cohort512)
        all_items, known_items = target_items(indices)
        unique = np.unique(known_items)
        coverage = {}
        for count, possible in positive_sets.items():
            seen_occurrences = int(np.isin(known_items, possible).sum())
            seen_unique = int(np.isin(unique, possible).sum())
            coverage[str(count)] = {
                "known_positive_occurrences_seen_in_initial_pool": seen_occurrences,
                "fraction_known_positive_occurrences_seen": seen_occurrences / len(known_items),
                "unique_known_videos_seen_in_initial_pool": seen_unique,
                "fraction_unique_known_videos_seen": seen_unique / len(unique),
            }
        future[str(day)] = {
            "positive_requests": len(indices), "active_users": len(np.unique(data.requests["uid"][indices])),
            "request_indices_sha256": digest(indices), "all_positive_occurrences": len(all_items),
            "known_positive_occurrences": len(known_items), "unique_known_positive_videos": len(unique),
            "coverage_by_initial_training_cohort": coverage,
        }
    historical_path = ROOT / "results/recflow/development/supervision_coverage/summary.json"
    sources = [Path(__file__).resolve(), a_summary, capacity_path, historical_path,
        Path(config["data"]) / "manifest.json", ROOT / "src/hstu_kvcache/recflow/data.py"]
    report = {
        "scope": "CPU-only explanatory development data audit. No training, new request selection, checkpoint scoring, or final-role outcomes.",
        "catalog_items": len(selected), "catalog_sha256": config["catalog_sha256"],
        "evaluation_cohort_users": 512, "evaluation_cohort_sha256": digest(cohort512),
        "cohort_rule": "Development-role users with >=1024 initial D1-18 real exposures, sorted by stable_hash(uid,1818); the fixed512 is the exact prefix of fixed2048. No future activity, labels or model outcomes select either cohort.",
        "definition": "Use every distinct-per-request positive video from the complete initial D1-18 positive pool. Restrict numerator and denominator to the same fixed1M catalog. Occurrence coverage counts repeat video labels across requests; unique coverage counts each future day's distinct known video once. Every known initial positive belongs to an eligible fitting request.",
        "initial_pools": initial, "future_fixed512_days": future,
        "actual_three_epoch_sampling": {
            "coverage_available_from_existing_audit": False,
            "reason": "The completed A retains per-epoch target hashes but neither drawn target arrays nor unique-target counts. The existing supervision_coverage audit belongs to an earlier partial-pass pilot (58,784 A requests), so it cannot provide coverage for this completed three-epoch A.",
            "epochs": [{key: entry["training"][key] for key in ("epoch", "rng_seed", "processed_requests", "sampled_targets_sha256")} for entry in saved["epochs"]],
            "sampling_reconstructed": False,
        },
        "interpretation": "These are upper bounds on direct positive-label coverage available from the eligible initial pool, not measured coverage of the actual three sampled epochs. Initial absence does not mean an item embedding received no gradient: history inputs and within-category softmax can still update it. Greater fixed2048 pool coverage is a possible mechanism to investigate, not evidence that expansion improves future recommendation or update stability.",
        "checks": {"fixed512_exact_prefix_of_fixed2048": True, "catalog_matches_completed_A": True,
            "initial_unique_video_counts_and_hashes_match_prior_capacity_audit": True},
        "source_sha256": {str(path.relative_to(ROOT)): file_digest(path) for path in sources},
        "elapsed_seconds": time.monotonic() - started,
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "initial_pools": initial, "future_fixed512_days": future}), flush=True)


if __name__ == "__main__":
    main()
