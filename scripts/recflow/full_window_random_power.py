#!/usr/bin/env python3
"""CPU-only random-ranking resolution audit; no model scoring or panel change."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from development_probe import ROOT, PreparedRecFlow, ProbeData, save_json
from random_baseline import draw_random_panel


def digest(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def panel_counts(data, indices, y, m):
    requests = data.prepared.requests[indices]
    uids = np.unique(requests["uid"])
    return dict(requests=len(indices), known_positive_requests=int((m > 0).sum()),
        all_oov_positive_requests=int((m == 0).sum()), positive_targets=int(y.sum()),
        known_positive_targets=int(m.sum()), oov_positive_targets=int((y - m).sum()),
        target_coverage=float(m.sum() / y.sum()), users_with_positive_requests=len(uids),
        uid_min=int(uids.min()), uid_max=int(uids.max()), unique_uids_sha256=digest(uids),
        request_indices_sha256=digest(indices), request_uids_sha256=digest(requests["uid"]),
        positive_counts_sha256=digest(y), known_positive_counts_sha256=digest(m),
        roles=np.unique(requests["role"]).astype(int).tolist())


def null_summary(y, m, catalog_size, indices, draws, seed, output, label):
    n = np.full(len(indices), catalog_size, dtype=np.int64)
    signature = hashlib.sha256(np.column_stack([y, m, n]).tobytes() + indices.tobytes()).hexdigest()
    derived_seed = int.from_bytes(hashlib.sha256(f"{seed}:{signature}".encode()).digest()[:8], "little")
    trials, analytic = draw_random_panel(y, m, n, draws, derived_seed)
    values = trials["ndcg@50"]
    expected = analytic["ndcg@50"]
    quantiles = np.quantile(values, [.025, .975, .99])
    discounts = 1.0 / np.log2(np.arange(2, 52))
    ideal = np.cumsum(discounts)[np.minimum(y, 50) - 1]
    # Exact finite-population variance of m positive positions among N items;
    # independent random permutations per request, with all-OOV rows retained.
    variance_per_request = (m * (catalog_size - m) / (catalog_size * (catalog_size - 1))
        * (np.square(discounts).sum() - discounts.sum() ** 2 / catalog_size) / np.square(ideal))
    exact_std = float(np.sqrt(variance_per_request.sum()) / len(indices))
    log_no_hits = 0.0
    for known, requests in zip(*np.unique(m, return_counts=True), strict=True):
        ranks = np.arange(known)
        log_no_hits += int(requests) * np.log1p(-50.0 / (catalog_size - ranks)).sum()
    trial_file = f"{label}_ndcg50_trials.npz"
    np.savez_compressed(output / trial_file, ndcg50=values)
    return dict(metric="ndcg@50", catalog_items=catalog_size, draws=draws,
        derived_seed=derived_seed, panel_signature=signature, trial_file=trial_file,
        analytic_expectation=expected, analytic_random_policy_std=exact_std,
        expected_total_positive_top50_hits=float(50 * m.sum() / catalog_size),
        analytic_probability_zero_top50_hits=float(np.exp(log_no_hits)),
        null_mean=float(values.mean()), null_std=float(values.std(ddof=1)),
        null_p2_5=float(quantiles[0]), null_p97_5=float(quantiles[1]), null_p99=float(quantiles[2]),
        null_p99_over_analytic_expectation=float(quantiles[2] / expected),
        null_std_over_analytic_expectation=exact_std / expected,
        monte_carlo_mean_standard_error=float(values.std(ddof=1) / np.sqrt(draws)),
        monte_carlo_mean_minus_analytic=float(values.mean() - expected),
        hypothetical_5_7x_expected=dict(score=5.7 * expected,
            exceeds_null_p99=bool(5.7 * expected > quantiles[2]),
            empirical_random_policy_tail_probability=float((values >= 5.7 * expected).mean()),
            interpretation="Fixed hypothetical multiplier for resolution comparison, not a measured full-window model score or estimated model-test power."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/recflow_v1")
    parser.add_argument("--panels", type=Path,
        default=ROOT / "results/recflow/development/seed17_complete_epoch_windows/window_panels.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    data = ProbeData(PreparedRecFlow(args.data), 1_000_000, 512, 1024)
    manifest = json.loads(args.panels.with_name("summary.json").read_text())
    assert digest(data.uids) == manifest["cohort_sha256"]
    assert digest(data.raw_ids) == manifest["catalog_sha256"]
    assert hashlib.sha256(args.panels.read_bytes()).hexdigest() == manifest["arrays_file_sha256"]
    frozen = dict(np.load(args.panels))
    windows, arrays = {}, {}
    for first, last in ((19, 21), (22, 24), (25, 27)):
        window = f"{first}_{last}"
        indices = data.indices(first, last, limit=None)
        requests = data.prepared.requests[indices]
        assert np.all(requests["role"] == 0) and np.isin(requests["uid"], data.uids).all()
        # Read relevance counts only for this fixed development cohort.
        counts = np.asarray([(len(raw), len(known)) for raw, known in
                             (data.targets(index) for index in indices)], dtype=np.int64)
        y, m = counts.T
        assert np.all(y > 0)
        assert len(indices) == manifest["evaluation"][window]["all_positive_requests"]["requests"]
        positions = {int(index): row for row, index in enumerate(indices)}
        panel = np.asarray(frozen[f"eval_{window}"], dtype=np.int64)
        selected = np.asarray([positions[int(index)] for index in panel])
        full_null = null_summary(y, m, len(data.raw_ids), indices, args.draws, args.seed,
                                 args.output, f"full_{window}")
        panel_null = null_summary(y[selected], m[selected], len(data.raw_ids), panel,
                                  args.draws, args.seed, args.output, f"panel_{window}")
        daily = {}
        for day in range(first, last + 1):
            chosen = np.flatnonzero(requests["day"] == day)
            daily[str(day)] = panel_counts(data, indices[chosen], y[chosen], m[chosen])
        windows[window] = dict(days=[first, last],
            full=panel_counts(data, indices, y, m), frozen_panel=panel_counts(data, panel, y[selected], m[selected]),
            daily=daily, full_random=full_null, frozen_panel_random=panel_null,
            comparison=dict(request_count_multiplier=len(indices) / len(panel),
                analytic_std_full_over_panel=full_null["analytic_random_policy_std"] / panel_null["analytic_random_policy_std"],
                null_p99_full_over_panel=full_null["null_p99"] / panel_null["null_p99"],
                analytic_expectation_full_over_panel=full_null["analytic_expectation"] / panel_null["analytic_expectation"]))
        for name, values in (("indices", indices), ("uids", requests["uid"]), ("days", requests["day"]),
                             ("positives", y), ("known_positives", m)):
            arrays[f"{window}_{name}"] = values
        print(json.dumps(dict(window=window, full_requests=len(indices),
            full_null99=full_null["null_p99"], panel_null99=panel_null["null_p99"],
            seconds=time.monotonic() - started)), flush=True)
    np.savez_compressed(args.output / "development_window_counts.npz", **arrays)
    sources = [Path(__file__), ROOT / "scripts/recflow/random_baseline.py",
               ROOT / "scripts/recflow/development_probe.py", ROOT / "src/hstu_kvcache/recflow/data.py",
               ROOT / "src/hstu_kvcache/recflow/metrics.py"]
    save_json(args.output / "summary.json", dict(
        scope="CPU random-policy resolution audit only; no GPU, model/checkpoint scoring, training, new current panel, admission/final users, or metric/pool changes.",
        cohort_users=len(data.uids), cohort_sha256=digest(data.uids), catalog_items=len(data.raw_ids),
        catalog_sha256=digest(data.raw_ids), panels=str(args.panels),
        panels_sha256=manifest["arrays_file_sha256"],
        data_manifest_sha256=hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest(),
        seed=args.seed, draws=args.draws, metric="ndcg@50", windows=windows,
        fixed_scope="Same initial-history eligible512 development UID cohort and initial development-exposure top1M catalog; all positive requests in each fixed three-day window, including all-OOV requests; no later-activity filtering.",
        null_definition="Independent uniform random permutations of all1M video IDs per request. Exact without-replacement positive-rank simulation reuses random_baseline.draw_random_panel; raw positive counts including OOV define IDCG. Analytic variance is finite-population exact under this random policy.",
        weighting="Each available positive request has equal weight. The3072 panel has1024 requests per day; the full-window day weights follow daily request volume, so their expectations need not coincide exactly.",
        interpretation="Null variation measures a random policy on a fixed panel, not model/generalization/training-seed uncertainty. A narrower null threshold can motivate a prospectively fixed, matched full-window check; no model score is extrapolated from3072 requests and no model success is claimed. Actual statistical power requires an alternative model-score distribution, unavailable in this counts-only audit. Existing primary panel and all failures remain unchanged.",
        monte_carlo_precision="With5000 trials the estimated99th percentile describes a tail of about50 simulations; retain the analytic expectation/std and raw trials rather than treating the quantile as exact.",
        source_sha256={str(path.resolve().relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        elapsed_seconds=time.monotonic() - started))


if __name__ == "__main__":
    main()
