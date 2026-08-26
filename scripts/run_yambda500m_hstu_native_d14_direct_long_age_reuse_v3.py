#!/usr/bin/env python3
"""Complete the D=14/E=14 v2-current, v0-cache direct Reuse cell."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import yaml

from run_yambda500m_hstu_native_d14_direct_long_age_reuse import (
    evaluate,
    report_row,
    sha256,
)
from run_yambda500m_hstu_native_d14_onehop_reuse import MANIFEST, MATRIX, ROOT, checkpoint


CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_d14_direct_long_age_reuse_v3.yaml"
OUTPUT = MATRIX / "d14_direct_long_age_reuse_v3"
CUTOVER_DAY = 245
END_DAY = 259
ADJACENT_REPORT = (
    MATRIX / "d14_onehop_reuse_diagnostic_v1/eval_14d/v1_to_v2/adjudication.json"
)


def verify_frozen_inputs(contract: dict) -> None:
    frozen = contract["frozen_inputs"]
    paths = {
        ROOT / frozen["executor"]: frozen["executor_sha256"],
        ROOT / frozen["adjudicator"]: frozen["adjudicator_sha256"],
        MANIFEST / "manifest.json": frozen["matrix_manifest_sha256"],
        MANIFEST / "requests_fidelity.parquet": frozen["requests_fidelity_sha256"],
        MANIFEST / "requests_quality.parquet": frozen["requests_quality_sha256"],
    }
    for version, record in frozen["checkpoints"].items():
        path = ROOT / record["path"]
        if path != checkpoint(int(version[1:])):
            raise RuntimeError(f"contract checkpoint path disagrees with runner for {version}")
        paths[path] = record["sha256"]
    adjacent = frozen["reused_adjacent_result"]
    paths[ROOT / adjacent["raw"]] = adjacent["raw_sha256"]
    paths[ROOT / adjacent["adjudication"]] = adjacent["adjudication_sha256"]
    for path, expected in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"frozen input hash mismatch: {path}: {observed} != {expected}")

    scope = contract["scope"]
    if scope["direct_long_age_cell"] != {"producer": "v0", "current": "v2"}:
        raise RuntimeError("runner cell disagrees with the frozen contract")
    if scope["evaluation_day_range"] != [CUTOVER_DAY, END_DAY]:
        raise RuntimeError("runner window disagrees with the frozen contract")


def emit() -> None:
    reports = [ADJACENT_REPORT, OUTPUT / "v0_to_v2/adjudication.json"]
    rows = [report_row(report) for report in reports if report.is_file()]
    rows.sort(key=lambda row: int(row["producer"][1:]))
    lines = [
        "# D=14/E=14 direct long-age KV Reuse: current v2",
        "",
        "Both rows use v2 exact rolling cache as Recompute over days [245, 259). The v0 row is direct long-age Reuse; the v1 row reuses the already sealed adjacent one-hop result.",
        "",
        "| Current | KV producer | Version gap | Current - Reuse ROC-AUC (pp) | Current - Reuse PR-AUC (pp) | Reuse - Current event log-loss | User-equal log-loss | JS |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['current']} | {row['producer']} | {row['version_gap']} | "
            f"{row['current_minus_direct_reuse_ROC_AUC_pp']:+.6f} | "
            f"{row['current_minus_direct_reuse_PR_AUC_pp']:+.6f} | "
            f"{row['direct_reuse_minus_current_event_log_loss']:+.6f} | "
            f"{row['direct_reuse_minus_current_user_log_loss']:+.6f} | "
            f"{row['mean_Bernoulli_JS']:.2e} |"
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "direct_long_age_v2_matrix.json").write_text(json.dumps(rows, indent=2) + "\n")
    (OUTPUT / "direct_long_age_v2_matrix.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "v2_direct_long_age_summary_emitted", "rows": len(rows)}))


def environment() -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": "src",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "OMP_NUM_THREADS": "2",
        "PYTHONUNBUFFERED": "1",
    }


def run_canary(contract: dict, env: dict[str, str]) -> None:
    marker = OUTPUT / "canary_complete.json"
    if marker.exists():
        payload = json.loads(marker.read_text())
        if payload.get("contract_sha256") != sha256(CONTRACT):
            raise RuntimeError("existing canary belongs to a different contract")
        print(json.dumps(payload, indent=2))
        return
    with tempfile.TemporaryDirectory(prefix="evokv_d14_v2_long_age_canary_") as temporary:
        destination = Path(temporary) / "v0_to_v2"
        evaluate(
            producer=0,
            current=2,
            cutover=CUTOVER_DAY,
            output=destination,
            env=env,
            max_users=int(contract["canary"]["max_users_per_rank"]),
        )
        report = json.loads((destination / "adjudication.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "four_rank_v2_direct_long_age_canary_passed",
        "contract_sha256": sha256(CONTRACT),
        "edge": report["edge"],
        "evaluation_day_range": report["evaluation_day_range"],
        "requests": report["reuse_minus_recompute"]["paired_harm"]["requests"],
    }
    marker.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def run_formal(env: dict[str, str]) -> None:
    marker = OUTPUT / "canary_complete.json"
    if not marker.is_file():
        raise RuntimeError("formal phase requires the separately launched passing canary")
    canary = json.loads(marker.read_text())
    if canary.get("contract_sha256") != sha256(CONTRACT):
        raise RuntimeError("canary contract hash differs from the current contract")
    destination = OUTPUT / "v0_to_v2"
    report = destination / "adjudication.json"
    if report.exists():
        emit()
        return
    if destination.exists():
        raise RuntimeError(f"partial formal output requires audit: {destination}")
    evaluate(
        producer=0,
        current=2,
        cutover=CUTOVER_DAY,
        output=destination,
        env=env,
    )
    emit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "canary", "formal", "emit"), required=True)
    args = parser.parse_args()
    contract = yaml.safe_load(CONTRACT.read_text())
    verify_frozen_inputs(contract)
    if args.phase == "preflight":
        print(json.dumps({"status": "frozen_inputs_verified", "contract_sha256": sha256(CONTRACT)}, indent=2))
    elif args.phase == "emit":
        emit()
    elif args.phase == "canary":
        run_canary(contract, environment())
    else:
        run_formal(environment())


if __name__ == "__main__":
    main()
