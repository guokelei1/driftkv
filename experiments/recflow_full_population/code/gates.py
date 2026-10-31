"""Prerelease native learning gates. Cache-gap sign never gates the version chain."""
from __future__ import annotations

import csv
from pathlib import Path

from .protocol import paired_uid_interval


def read_rows(directory):
    rows = {}
    for path in sorted(Path(directory).glob("rank_*.csv")):
        with path.open() as handle:
            for row in csv.DictReader(handle):
                key=tuple(int(row[k]) for k in ["uid","day","rid"])
                if key in rows:
                    raise ValueError("Duplicate validation request")
                rows[key]=row
    return rows


def early_learning(initial, trained):
    old, new = read_rows(initial), read_rows(trained)
    if not old or old.keys()!=new.keys():
        raise ValueError("Validation controls do not share requests")
    keys=[key for key in sorted(old) if old[key]["current_observed_bce"]]
    if any(not new[key]["current_observed_bce"] for key in keys):
        raise ValueError("Validation feedback availability changed")
    paired=paired_uid_interval([k[0] for k in keys],
        [float(old[k]["current_observed_bce"]) for k in keys],
        [float(new[k]["current_observed_bce"]) for k in keys])
    return dict(passed=paired["ci95"][0]>0, criterion="initial_minus_trained_observed_BCE_UID95_lower_gt_zero",
                paired_bce_improvement=paired, initial=str(initial), trained=str(trained), uses_cache_gap=False)


def native_learning(summary):
    comparisons=summary.get("paired_current_minus",{})
    passed=all(comparisons.get(name,{}).get("ci95",[-1,-1])[0]>0 for name in ["random","popularity"])
    return dict(passed=passed, criterion="native_NDCG10_minus_both_fixed_baselines_UID95_lower_gt_zero",
                comparisons={k:comparisons.get(k) for k in ["random","popularity"]},uses_cache_gap=False)


def choose_foundation(candidates):
    if not candidates:
        raise ValueError("No completed alignment epoch")
    return max(candidates,key=lambda c:(c["summary"]["metrics"]["current"]["ndcg@10"],-c["epoch"]))
