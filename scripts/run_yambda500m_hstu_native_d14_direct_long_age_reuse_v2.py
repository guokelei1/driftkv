#!/usr/bin/env python3
"""Complete the D=14/E=14 direct long-age Reuse row for current v5.

GPU work is deliberately split into explicit ``canary`` and ``formal`` phases.
The runner never advances from the canary to the formal population by itself.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from run_yambda500m_hstu_native_d14_direct_long_age_reuse import report_row, sha256
from run_yambda500m_hstu_native_d14_onehop_reuse import MANIFEST, MATRIX, ROOT, checkpoint


CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_d14_direct_long_age_reuse_v2.yaml"
OUTPUT = MATRIX / "d14_direct_long_age_reuse_v2"
NEW_PRODUCERS = (0, 1, 2, 3)
CUTOVER_DAY = 287
END_DAY = 301
REUSED_V4_TO_V5 = (
    MATRIX / "d14_onehop_reuse_completion_v2/eval_14d/v4_to_v5/adjudication.json"
)


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def evaluate(
    *, producer: int, output: Path, env: dict[str, str], max_users: int = 0
) -> None:
    command = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=4",
        "scripts/evaluate_yambda500m_hstu_native_onehop_reuse_raw.py",
        "--stage",
        f"d14_direct_long_age_v{producer}_to_v5",
        "--edge",
        f"v{producer}_to_v5",
        "--cutover-day",
        str(CUTOVER_DAY),
        "--start-day",
        str(CUTOVER_DAY),
        "--end-day",
        str(END_DAY),
        "--manifest-dir",
        str(MANIFEST),
        "--parent",
        str(checkpoint(producer)),
        "--current",
        str(checkpoint(5)),
        # The existing v4->v5 E=14 result established that this window needs
        # the scalar fallback to keep peak rolling-cache allocation bounded.
        "--force-fallback",
        "--cohort-size",
        "1",
        "--output",
        str(output),
    ]
    if max_users:
        command.extend(["--max-users", str(max_users)])
    run(command, env)
    run(
        [
            sys.executable,
            "scripts/adjudicate_yambda500m_hstu_native_onehop_reuse.py",
            "--raw",
            str(output / "raw.parquet"),
            "--seal",
            str(output / "raw.seal.json"),
            "--labels",
            str(MANIFEST / "requests_quality.parquet"),
            "--output",
            str(output / "adjudication.json"),
        ],
        env,
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
        paths[ROOT / record["path"]] = record["sha256"]
        if ROOT / record["path"] != checkpoint(int(version[1:])):
            raise RuntimeError(f"contract checkpoint path disagrees with runner for {version}")
    for record in frozen["reused_results"].values():
        paths[ROOT / record["raw"]] = record["raw_sha256"]
        paths[ROOT / record["adjudication"]] = record["adjudication_sha256"]
    for path, expected in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"frozen input hash mismatch: {path}: {observed} != {expected}")

    cells = contract["scope"]["direct_long_age_cells"]
    observed_cells = tuple(int(cell["producer"][1:]) for cell in cells)
    if observed_cells != NEW_PRODUCERS or any(cell["current"] != "v5" for cell in cells):
        raise RuntimeError("runner cells disagree with the frozen contract")
    if contract["scope"]["evaluation_day_range"] != [CUTOVER_DAY, END_DAY]:
        raise RuntimeError("runner window disagrees with the frozen contract")


def emit() -> None:
    reports = [REUSED_V4_TO_V5]
    reports.extend(OUTPUT / f"v{producer}_to_v5/adjudication.json" for producer in NEW_PRODUCERS)
    rows = [report_row(report) for report in reports if report.is_file()]
    rows.sort(key=lambda row: int(row["producer"][1:]))
    lines = [
        "# D=14/E=14 direct long-age KV Reuse: current v5",
        "",
        "Every row uses v5 exact rolling cache as Recompute. The named producer materializes the entire pre-cutover prefix at day 287; v5 then reads that producer KV and appends all events in days [287, 301). This is direct long-age Reuse, not recursive lineage.",
        "",
        "| Current | KV producer | Version gap | Current - Direct Reuse ROC-AUC (pp) | Current - Direct Reuse PR-AUC (pp) | Direct Reuse - Current event log-loss | User-equal log-loss | JS |",
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
    (OUTPUT / "direct_long_age_v5_matrix.json").write_text(json.dumps(rows, indent=2) + "\n")
    (OUTPUT / "direct_long_age_v5_matrix.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "v5_direct_long_age_summary_emitted", "rows": len(rows)}))


def run_canary(contract: dict, env: dict[str, str]) -> None:
    marker = OUTPUT / "canary_complete.json"
    if marker.exists():
        payload = json.loads(marker.read_text())
        if payload.get("contract_sha256") != sha256(CONTRACT):
            raise RuntimeError("existing canary belongs to a different contract")
        print(json.dumps(payload, indent=2))
        return
    with tempfile.TemporaryDirectory(prefix="evokv_d14_v5_long_age_canary_") as temporary:
        destination = Path(temporary) / "v0_to_v5"
        evaluate(
            producer=0,
            output=destination,
            env=env,
            max_users=int(contract["canary"]["max_users_per_rank"]),
        )
        canary_report = json.loads((destination / "adjudication.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "four_rank_v5_direct_long_age_canary_passed",
        "contract_sha256": sha256(CONTRACT),
        "edge": canary_report["edge"],
        "evaluation_day_range": canary_report["evaluation_day_range"],
        "requests": canary_report["reuse_minus_recompute"]["paired_harm"]["requests"],
    }
    marker.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def run_formal(producers: tuple[int, ...], env: dict[str, str]) -> None:
    marker = OUTPUT / "canary_complete.json"
    if not marker.is_file():
        raise RuntimeError("formal phase requires the separately launched passing canary")
    canary = json.loads(marker.read_text())
    if canary.get("contract_sha256") != sha256(CONTRACT):
        raise RuntimeError("canary contract hash differs from the current contract")
    for producer in producers:
        output = OUTPUT / f"v{producer}_to_v5"
        report = output / "adjudication.json"
        if report.exists():
            emit()
            continue
        if output.exists():
            raise RuntimeError(f"partial formal output requires audit: {output}")
        evaluate(producer=producer, output=output, env=env)
        emit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "canary", "formal", "emit"), required=True)
    parser.add_argument("--producers", type=int, nargs="+", default=list(NEW_PRODUCERS))
    args = parser.parse_args()
    producers = tuple(args.producers)
    if any(producer not in NEW_PRODUCERS for producer in producers) or len(set(producers)) != len(producers):
        raise ValueError(f"producers must be unique values drawn from {NEW_PRODUCERS}")

    contract = yaml.safe_load(CONTRACT.read_text())
    verify_frozen_inputs(contract)
    if args.phase == "preflight":
        print(json.dumps({"status": "frozen_inputs_verified", "contract_sha256": sha256(CONTRACT)}, indent=2))
        return
    if args.phase == "emit":
        emit()
        return

    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "OMP_NUM_THREADS": "2",
        "PYTHONUNBUFFERED": "1",
    }
    if args.phase == "canary":
        run_canary(contract, env)
    else:
        run_formal(producers, env)


if __name__ == "__main__":
    main()
