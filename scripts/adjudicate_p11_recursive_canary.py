#!/usr/bin/env python3
"""Seal P11.1 batched-vs-single-user recursive-lineage equivalence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

import train_p7_theta0 as p7


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/p11_1_recursive_population_contract_v1.yaml"
OLD = ROOT / "results/p11/p11_0_version_debt_canary_v1/state_metrics.parquet"
NEW = ROOT / "results/p11/p11_1_recursive_population_raw/canary32/m1_seed17/state_metrics.parquet"
MANIFEST = NEW.parent / "raw_manifest.json"
OUTPUT = ROOT / "results/p11/p11_1_recursive_canary_equivalence_v1.json"
MAPPING = {
    "one_hop": "one_hop",
    "direct_age2": "direct_age2",
    "recursive_mixed": "recursive_noop",
}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    contract = yaml.safe_load(CONTRACT.read_text())
    old_rows = {(int(row["uid"]), row["action"]): row for row in pq.read_table(OLD).to_pylist()}
    new_rows = {(int(row["uid"]), row["action"]): row for row in pq.read_table(NEW).to_pylist()}
    comparisons = []
    for old_action, new_action in MAPPING.items():
        uids = sorted(uid for uid, action in old_rows if action == old_action)
        mse = [abs(old_rows[(uid, old_action)]["mse"] - new_rows[(uid, new_action)]["mse"]) for uid in uids]
        js = [abs(old_rows[(uid, old_action)]["bernoulli_js"] - new_rows[(uid, new_action)]["bernoulli_js"]) for uid in uids]
        comparisons.append({
            "single_user_action": old_action, "batched_action": new_action,
            "uids": len(uids), "max_abs_MSE_difference": max(mse),
            "max_abs_JS_difference": max(js),
        })
    action_rows = pq.read_table(NEW).to_pylist()
    mean_js = {
        action: float(np.mean([row["bernoulli_js"] for row in action_rows if row["action"] == action]))
        for action in sorted({row["action"] for row in action_rows})
    }
    noop = mean_js["recursive_noop"]
    recovery = {
        action: float(1.0 - value / noop)
        for action, value in mean_js.items() if action.startswith("recursive_")
    }
    max_mse = max(row["max_abs_MSE_difference"] for row in comparisons)
    max_js = max(row["max_abs_JS_difference"] for row in comparisons)
    passed = (
        max_mse <= float(contract["gates"]["canary_per_uid_MSE_tolerance"])
        and max_js <= float(contract["gates"]["canary_per_uid_JS_tolerance"])
        and json.loads(MANIFEST.read_text())["max_exact_abs_logit"]
        <= float(contract["gates"]["exact_all_max_absolute_logit"])
    )
    payload = {
        "status": "passed_batched_recursive_equivalence" if passed else "failed",
        "comparisons": comparisons, "mean_JS": mean_js,
        "recovery_fraction_vs_recursive_noop": recovery,
        "contract_sha256": p7.sha256_file(CONTRACT), "single_user_raw_sha256": p7.sha256_file(OLD),
        "batched_raw_sha256": p7.sha256_file(NEW), "batched_manifest_sha256": p7.sha256_file(MANIFEST),
        "full_population_authorized": passed, "quality_labels_read": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise RuntimeError("P11.1 canary equivalence failed")


if __name__ == "__main__":
    main()
