#!/usr/bin/env python3
"""Apply the frozen numeric-floor blocking gate to sealed R0 raw scores."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import eval_p7_h_raw as metrics
import numpy as np
import pyarrow.parquet as pq
import train_p7_theta0 as p7

ROOT = Path(__file__).resolve().parents[1]
SEAL = ROOT / "results/p8/r0_control/raw_score_seal_v1.json"
OUTPUT = ROOT / "results/p8/r0_control/adjudication_v1.json"
JS_FLOOR = 1e-8


def bootstrap_upper(values: dict[int, list[float]], namespace: str) -> tuple[float, float]:
    user_points = np.asarray([np.mean(rows) for rows in values.values()], dtype=np.float64)
    seed = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    points = np.empty(2000, dtype=np.float64)
    for index in range(len(points)):
        points[index] = np.mean(rng.choice(user_points, size=len(user_points), replace=True))
    return float(user_points.mean()), float(np.percentile(points, 97.5))


def evaluate_file(path: Path, workload: str, namespace: str) -> dict:
    table = pq.read_table(path).to_pydict()
    requests: dict[str, list[int]] = defaultdict(list)
    for index, request_id in enumerate(table["request_id"]):
        requests[str(request_id)].append(index)
    by_user: dict[int, list[float]] = defaultdict(list)
    max_logit_delta = 0.0
    all_js = []
    for indices in requests.values():
        full = np.asarray([table["current_full512_logit"][index] for index in indices], dtype=np.float64)
        reuse = np.asarray([table["reuse_parent_kv_logit"][index] for index in indices], dtype=np.float64)
        js = metrics.identity_metrics(full, reuse, workload)["output_js_divergence"]
        uid = int(table["uid"][indices[0]])
        by_user[uid].append(js)
        all_js.append(js)
        max_logit_delta = max(max_logit_delta, float(np.max(np.abs(full - reuse))))
    point, upper = bootstrap_upper(by_user, namespace)
    return {
        "requests": len(requests), "users": len(by_user), "JS_mean_equal_user": point,
        "JS_user_bootstrap_CI95_upper": upper, "JS_P95_request": float(np.percentile(all_js, 95)),
        "JS_P99_request": float(np.percentile(all_js, 99)), "max_abs_logit_delta": max_logit_delta,
        "passes_floor": upper <= JS_FLOOR,
    }


def main() -> None:
    seal = json.loads(SEAL.read_text())
    if seal["metrics_computed"] is not False or seal["runs"] != 6:
        raise RuntimeError("R0 raw seal is incomplete")
    results = []
    for artifact in seal["artifacts"]:
        if not artifact["view"].startswith("fidelity"):
            continue
        result = evaluate_file(
            ROOT / artifact["path"], artifact["workload"],
            f"p8-r0:{artifact['model']}:{artifact['seed']}:{artifact['workload']}:{artifact['view']}",
        )
        results.append({
            "model": artifact["model"], "seed": artifact["seed"],
            "workload": artifact["workload"], "view": artifact["view"], **result,
        })
    passed = all(row["passes_floor"] for row in results)
    payload = {
        "status": "R0_blocking_control_passed" if passed else "R0_blocking_control_failed",
        "contract_hash": seal["contract_hash"], "raw_seal_hash": p7.sha256_file(SEAL),
        "JS_floor": JS_FLOOR, "all_six_cache_paths_identical": True,
        "results": results, "R1_R2_authorized_by_blocking_gate": passed,
        "tomography_or_controller_authorized": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "conditions": len(results)}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
