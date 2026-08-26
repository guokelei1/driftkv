#!/usr/bin/env python3
"""Run the prospective D14/E14 fixed one-release EvoKV AUC diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_d14_one_release_refinement_auc_v1.yaml"
MATRIX = ROOT / "results/yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v3"
MANIFEST = ROOT / "data/manifests/yambda500m_small_hstu_native_rolling_matrix_fast_v3"
OUTPUT = MATRIX / "d14_one_release_refinement_auc_v1"
GAIN_TABLE = MATRIX / "d14_onehop_reuse_completion_v2/auc_release_gain_coverage_table.json"

PRIOR_RAW = {
    "v0_to_v1": MATRIX / "d14_onehop_reuse_diagnostic_v1/eval_14d/v0_to_v1/raw.parquet",
    "v1_to_v2": MATRIX / "d14_onehop_reuse_diagnostic_v1/eval_14d/v1_to_v2/raw.parquet",
    "v2_to_v3": MATRIX / "d14_onehop_reuse_completion_v2/eval_14d/v2_to_v3/raw.parquet",
    "v3_to_v4": MATRIX / "d14_onehop_reuse_completion_v2/eval_14d/v3_to_v4/raw.parquet",
    "v4_to_v5": MATRIX / "d14_onehop_reuse_completion_v2/eval_14d/v4_to_v5/raw.parquet",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(version: int) -> Path:
    if version == 0:
        return ROOT / "results/yambda500m_small_seed17/hstu_native_release_chain_v1/v0/checkpoint_100.pt"
    return MATRIX / f"train_14d/checkpoints/v{version}/checkpoint_100.pt"


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(value)
    os.replace(temporary, path)


def evaluate(
    *, edge: int, horizon: int, output: Path, env: dict[str, str],
    prior_sha256: str, max_users: int = 0, require_exact_request_set: bool = False,
    cohort_size: int = 32,
) -> None:
    edge_name = f"v{edge - 1}_to_v{edge}"
    cutover = 217 + edge * 14
    command = [
        "torchrun", "--standalone", "--nproc_per_node=4",
        "scripts/evaluate_yambda500m_hstu_native_onehop_reuse_raw.py",
        "--stage", f"d14_one_release_refinement_e{horizon}_edge{edge}",
        "--edge", edge_name,
        "--cutover-day", str(cutover),
        "--start-day", str(cutover),
        "--end-day", str(cutover + horizon),
        "--manifest-dir", str(MANIFEST),
        "--parent", str(checkpoint(edge - 1)),
        "--current", str(checkpoint(edge)),
        "--include-parent-exact",
        "--include-fixed-refinement",
        "--cohort-size", str(cohort_size),
        "--output", str(output),
    ]
    if max_users:
        command.extend(["--max-users", str(max_users)])
    run(command, env)
    adjudicate = [
        sys.executable,
        "scripts/insight/adjudicate_one_release_auc.py",
        "--raw", str(output / "raw.parquet"),
        "--seal", str(output / "raw.seal.json"),
        "--labels", str(MANIFEST / "requests_quality.parquet"),
        "--prior-raw", str(PRIOR_RAW[edge_name]),
        "--prior-raw-sha256", prior_sha256,
        "--full-only-gain-table", str(GAIN_TABLE),
        "--output", str(output / "adjudication.json"),
    ]
    if require_exact_request_set:
        adjudicate.append("--require-exact-request-set")
    run(adjudicate, env)


def reports() -> list[dict]:
    rows = []
    for path in sorted(OUTPUT.glob("eval_14d/v*_to_v*/adjudication.json")):
        report = json.loads(path.read_text())
        requested = report["requested_full_only_reference_ratio"]
        delta = report["rolling_auc_deltas"]
        matched = report["matched_rolling_ratio"]
        rows.append({
            "edge": report["edge"],
            "requests": report["requests"],
            "parent_rolling_AUC": report["absolute_metrics"]["parent_exact_rolling"]["ROC_AUC"],
            "recompute_rolling_AUC": report["absolute_metrics"]["current_exact_rolling"]["ROC_AUC"],
            "reuse_rolling_AUC": report["absolute_metrics"]["one_hop_reuse_rolling"]["ROC_AUC"],
            "our_rolling_AUC": report["absolute_metrics"]["evokv_cast_group_patch_scale_r128_c64_rolling"]["ROC_AUC"],
            "reuse_gain_retained_percent": requested["reuse_gain_retained_percent"],
            "our_gain_retained_percent": requested["our_gain_retained_percent"],
            "our_minus_reuse_ROC_AUC_pp": delta["our_minus_reuse_ROC_AUC_pp"],
            "reuse_harm_recovered_fraction": delta["reuse_harm_recovered_fraction"],
            "matched_rolling_reuse_gain_retained_fraction": matched["reuse_gain_retained_fraction"],
            "matched_rolling_our_gain_retained_fraction": matched["our_gain_retained_fraction"],
        })
    return sorted(rows, key=lambda row: int(row["edge"].split("_to_")[0][1:]))


def render(rows: list[dict]) -> str:
    lines = [
        "# D14/E14 one-release EvoKV rolling AUC",
        "",
        "The requested retained-gain columns preserve the earlier motivation denominator: "
        "D14 Full-only `Current − Parent` AUC. Reuse and Our deltas are measured on the "
        "same full-population rolling requests. Matched rolling ratios remain companions.",
        "Equivalently, `retained(path) = 1 - (AUC_Recompute - AUC_path) / "
        "(AUC_CurrentFull - AUC_ParentFull)`, which is the pre-existing "
        "`(AUC_path - AUC_old) / (AUC_Recompute - AUC_old)` axis.",
        "",
        "Fixed Our plan: parameter-only CAST of the old 384-position prefix, then "
        "GROUP recent 128 evidence into 64 Current PATCH carriers and SCALE each carrier "
        "by represented mass 2. This is one-hop only.",
        "",
        "| Edge | Requests | Parent rolling AUC | Recompute AUC | Reuse AUC | Our AUC | Reuse gain retained | Our gain retained | Our − Reuse AUC (pp) | Reuse harm recovered |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        reuse_retained = row["reuse_gain_retained_percent"]
        our_retained = row["our_gain_retained_percent"]
        recovered = row["reuse_harm_recovered_fraction"]
        lines.append(
            f"| {row['edge'].replace('_to_', ' → ')} | {row['requests']} | "
            f"{row['parent_rolling_AUC']:.6f} | {row['recompute_rolling_AUC']:.6f} | "
            f"{row['reuse_rolling_AUC']:.6f} | {row['our_rolling_AUC']:.6f} | "
            f"{'N/A' if reuse_retained is None else f'{reuse_retained:+.1f}%'} | "
            f"{'N/A' if our_retained is None else f'{our_retained:+.1f}%'} | "
            f"{row['our_minus_reuse_ROC_AUC_pp']:+.6f} | "
            f"{'N/A' if recovered is None else f'{100.0 * recovered:+.1f}%'} |"
        )
    lines.extend([
        "",
        "`Reuse harm recovered = (AUC_Our − AUC_Reuse) / (AUC_Recompute − AUC_Reuse)`. "
        "Values outside [0,100%] are retained rather than clipped.",
        "",
        "All five formal E14 edges have `AUC_Recompute > AUC_Reuse`. The fixed plan "
        "improves Reuse on four edges and fails on v4 → v5. The v3 → v4 retained-gain "
        "ratio is unstable because the Full-only release-gain denominator is only "
        "0.046331 AUC point.",
        "",
    ])
    return "\n".join(lines)


def emit() -> None:
    rows = reports()
    atomic_text(OUTPUT / "auc_summary.json", json.dumps(rows, indent=2) + "\n")
    atomic_text(OUTPUT / "auc_summary.md", render(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--canary-only", action="store_true")
    args = parser.parse_args()
    if any(edge not in range(1, 6) for edge in args.edges):
        raise ValueError("edges must be drawn from 1..5")

    contract = yaml.safe_load(CONTRACT.read_text())
    frozen = contract["frozen_inputs"]
    if sha256(MANIFEST / "manifest.json") != frozen["matrix_manifest_sha256"]:
        raise RuntimeError("manifest differs from the prospective refinement contract")
    if sha256(GAIN_TABLE) != frozen["prior_full_only_gain_table"]["sha256"]:
        raise RuntimeError("Full-only gain table differs from the prospective contract")
    for version in range(6):
        expected = frozen["checkpoints"][f"v{version}"]["sha256"]
        if sha256(checkpoint(version)) != expected:
            raise RuntimeError(f"v{version} checkpoint differs from the prospective contract")
    prior_hashes = {
        edge: frozen["prior_E14_rolling_raw"][edge]["sha256"] for edge in PRIOR_RAW
    }
    for edge, path in PRIOR_RAW.items():
        if sha256(path) != prior_hashes[edge]:
            raise RuntimeError(f"sealed prior raw differs for {edge}")

    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "OMP_NUM_THREADS": "2",
        "PYTHONUNBUFFERED": "1",
    }
    marker = OUTPUT / "preflight_complete.json"
    if not marker.exists():
        with tempfile.TemporaryDirectory(prefix="evokv_one_release_auc_canary_") as temporary:
            evaluate(
                edge=1, horizon=1, output=Path(temporary) / "canary", env=env,
                prior_sha256=prior_hashes["v0_to_v1"], max_users=8,
            )
        atomic_text(marker, json.dumps({
            "status": "four_rank_fixed_refinement_canary_passed",
            "contract_sha256": sha256(CONTRACT),
        }, indent=2) + "\n")
    if args.canary_only:
        return

    for edge in args.edges:
        edge_name = f"v{edge - 1}_to_v{edge}"
        output = OUTPUT / "eval_14d" / edge_name
        report = output / "adjudication.json"
        if report.exists():
            continue
        if output.exists():
            raise RuntimeError(f"partial refinement evaluation requires audit: {output}")
        evaluate(
            edge=edge, horizon=14, output=output, env=env,
            prior_sha256=prior_hashes[edge_name], require_exact_request_set=True,
        )
        emit()
    emit()


if __name__ == "__main__":
    main()
