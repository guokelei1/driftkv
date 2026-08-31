#!/usr/bin/env python3
"""Run the frozen matched-cost evidence-measure basis canary and formal gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_evidence_measure_basis_v1.yaml"
MATRIX = ROOT / "results/yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v3"
MANIFEST = ROOT / "data/manifests/yambda500m_small_hstu_native_rolling_matrix_fast_v3"
OUTPUT = ROOT / "results/yambda500m_small_seed17/insight_evidence_measure_basis_v1"
EDGES = ("v0_to_v1", "v1_to_v2", "v2_to_v3", "v3_to_v4", "v4_to_v5")


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


def design0_raw(edge: str) -> Path:
    return MATRIX / "d14_one_release_refinement_auc_v1/eval_14d" / edge / "raw.parquet"


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(value)
    os.replace(partial, path)


def verify_contract() -> tuple[dict, str]:
    contract = yaml.safe_load(CONTRACT.read_text())
    frozen = contract["frozen_inputs"]
    for name in ("signed_causal_contract", "signed_causal_adjudication", "Design0_contract"):
        item = frozen[name]
        path = ROOT / item["path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"{name} differs from the prospective contract")
    causal = json.loads((ROOT / frozen["signed_causal_adjudication"]["path"]).read_text())
    if causal["status"] != frozen["signed_causal_adjudication"]["required_status"]:
        raise RuntimeError("signed causal gate has not passed")
    for name in ("matrix_manifest", "requests_fidelity", "requests_quality"):
        item = frozen[name]
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"{name} differs from the prospective contract")
    for version in range(6):
        if sha256(checkpoint(version)) != frozen["checkpoints"][f"v{version}"]["sha256"]:
            raise RuntimeError(f"v{version} checkpoint differs from the prospective contract")
    for edge in EDGES:
        if sha256(design0_raw(edge)) != frozen["Design0_formal_raw"][edge]["sha256"]:
            raise RuntimeError(f"{edge} Design-0 raw differs from the prospective contract")
    return contract, sha256(CONTRACT)


def evaluate_edge(
    *, edge_index: int, output: Path, env: dict[str, str], max_users: int,
    formal: bool, design0_hash: str,
) -> None:
    edge = EDGES[edge_index]
    cutover = 231 + 14 * edge_index
    command = [
        "torchrun", "--standalone", "--nproc_per_node=4",
        "scripts/evaluate_yambda500m_hstu_native_onehop_reuse_raw.py",
        "--stage", f"evidence_measure_basis_{'formal' if formal else 'canary'}_edge{edge_index + 1}",
        "--edge", edge,
        "--cutover-day", str(cutover),
        "--start-day", str(cutover),
        "--end-day", str(cutover + 14),
        "--manifest-dir", str(MANIFEST),
        "--parent", str(checkpoint(edge_index)),
        "--current", str(checkpoint(edge_index + 1)),
        "--include-fixed-refinement",
        "--include-evidence-measure-basis",
        "--cohort-size", "32",
        "--output", str(output),
    ]
    if formal:
        command.append("--include-parent-exact")
    if max_users:
        command.extend(["--max-users", str(max_users)])
    run(command, env)

    adjudication = [
        sys.executable,
        "scripts/insight/adjudicate_evidence_measure_basis.py",
        "--raw", str(output / "raw.parquet"),
        "--seal", str(output / "raw.seal.json"),
        "--prior-design0-raw", str(design0_raw(edge)),
        "--prior-design0-sha256", design0_hash,
        "--output", str(output / "adjudication.json"),
    ]
    if formal:
        adjudication.extend([
            "--labels", str(MANIFEST / "requests_quality.parquet"),
            "--require-exact-request-set",
        ])
    run(adjudication, env)
    for transient in (output / ".directory_ready", output / ".raw_complete"):
        transient.unlink(missing_ok=True)


def canary_summary(contract_hash: str) -> dict:
    reports = [
        json.loads((OUTPUT / "canary" / edge / "adjudication.json").read_text())
        for edge in EDGES
    ]
    passed_edges = sum(bool(report["basis_fidelity_not_worse_than_Design0"]) for report in reports)
    summary = {
        "status": "evidence_measure_basis_canary_passed" if passed_edges >= 4 else "evidence_measure_basis_canary_failed",
        "contract_sha256": contract_hash,
        "edges": len(reports),
        "users_sum_across_edges": sum(int(report["users"]) for report in reports),
        "requests_sum_across_edges": sum(int(report["requests"]) for report in reports),
        "basis_fidelity_not_worse_than_Design0_edges": passed_edges,
        "required_edges": 4,
        "labels_read": False,
        "matched_cost": {
            "Design0_joint_KV_CAST_equivalent_positions": 384,
            "basis_joint_KV_CAST_equivalent_positions": 384,
            "Current_PATCH_carriers": 64,
            "raw_repair_positions": 128,
            "materialized_state_positions": 448,
            "nominal_state_positions": 512,
        },
        "formal_quality_launched": False,
        "edge_results": [
            {
                "edge": report["edge"],
                "requests": report["requests"],
                "Design0_mean_abs_logit_gap": report["fidelity"]["evokv_cast_group_patch_scale_r128_c64_rolling"]["mean_abs_logit_gap_to_current"],
                "basis_mean_abs_logit_gap": report["fidelity"]["evokv_cast_measure_current_residual_r128_c64_rolling"]["mean_abs_logit_gap_to_current"],
                "basis_not_worse": report["basis_fidelity_not_worse_than_Design0"],
            }
            for report in reports
        ],
    }
    atomic_text(OUTPUT / "canary/summary.json", json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Matched-cost evidence-measure basis canary",
        "",
        f"Progression gate: **{'PASS' if passed_edges >= 4 else 'FAIL'}** ({passed_edges}/5 edges).",
        "",
        "No label was read. Current, Reuse and Design-0 logits were replayed against the prior sealed full-population raw before comparing the new basis.",
        "",
        "| edge | requests | Design 0 mean abs logit gap | evidence basis gap | basis not worse |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary["edge_results"]:
        lines.append(
            f"| {row['edge']} | {row['requests']} | {row['Design0_mean_abs_logit_gap']:.8g} | "
            f"{row['basis_mean_abs_logit_gap']:.8g} | {row['basis_not_worse']} |"
        )
    lines.extend(["", "The canary changes no model, action set or serving lineage.", ""])
    atomic_text(OUTPUT / "canary/report.md", "\n".join(lines))
    return summary


def formal_summary(contract_hash: str) -> dict:
    reports = [
        json.loads((OUTPUT / "formal/eval_14d" / edge / "adjudication.json").read_text())
        for edge in EDGES
    ]
    auc_pass = sum(report["formal_gate"]["basis_ROC_AUC_not_below_Design0"] for report in reports)
    log_loss_pass = sum(report["formal_gate"]["basis_log_loss_not_above_Design0"] for report in reports)
    passed = auc_pass == 5 and log_loss_pass == 5
    rows = []
    for report in reports:
        metrics = report["absolute_metrics"]
        fidelity = report["fidelity"]
        rows.append({
            "edge": report["edge"],
            "requests": report["requests"],
            "current_ROC_AUC": metrics["current_exact_rolling"]["ROC_AUC"],
            "Design0_ROC_AUC": metrics["evokv_cast_group_patch_scale_r128_c64_rolling"]["ROC_AUC"],
            "basis_ROC_AUC": metrics["evokv_cast_measure_current_residual_r128_c64_rolling"]["ROC_AUC"],
            "Design0_log_loss": metrics["evokv_cast_group_patch_scale_r128_c64_rolling"]["log_loss"],
            "basis_log_loss": metrics["evokv_cast_measure_current_residual_r128_c64_rolling"]["log_loss"],
            "Design0_mean_abs_logit_gap": fidelity["evokv_cast_group_patch_scale_r128_c64_rolling"]["mean_abs_logit_gap_to_current"],
            "basis_mean_abs_logit_gap": fidelity["evokv_cast_measure_current_residual_r128_c64_rolling"]["mean_abs_logit_gap_to_current"],
            "basis_AUC_not_below": report["formal_gate"]["basis_ROC_AUC_not_below_Design0"],
            "basis_log_loss_not_above": report["formal_gate"]["basis_log_loss_not_above_Design0"],
        })
    summary = {
        "status": "evidence_measure_basis_formal_gate_passed" if passed else "evidence_measure_basis_formal_gate_failed",
        "contract_sha256": contract_hash,
        "edges": 5,
        "requests": sum(row["requests"] for row in rows),
        "ROC_AUC_not_below_Design0_edges": auc_pass,
        "log_loss_not_above_Design0_edges": log_loss_pass,
        "required_each": 5,
        "labels_joined_only_after_each_raw_seal": True,
        "matched_cost": {
            "joint_KV_CAST_equivalent_positions": 384,
            "Current_PATCH_carriers": 64,
            "raw_repair_positions": 128,
            "materialized_state_positions": 448,
        },
        "action_admitted": False,
        "edge_results": rows,
    }
    atomic_text(OUTPUT / "formal/summary.json", json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Five-edge matched-cost evidence-measure basis",
        "",
        f"Formal mechanism gate: **{'PASS' if passed else 'FAIL'}**.",
        "",
        "Every edge was generated raw-first and sealed before label join. The basis and Design 0 use identical parameter-map arithmetic, Current PATCH carriers, raw repair scope and state layout.",
        "",
        "| edge | requests | Design 0 AUC | basis AUC | Design 0 log-loss | basis log-loss | Design 0 logit gap | basis logit gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['edge']} | {row['requests']} | {row['Design0_ROC_AUC']:.6f} | "
            f"{row['basis_ROC_AUC']:.6f} | {row['Design0_log_loss']:.6f} | "
            f"{row['basis_log_loss']:.6f} | {row['Design0_mean_abs_logit_gap']:.8g} | "
            f"{row['basis_mean_abs_logit_gap']:.8g} |"
        )
    lines.extend([
        "",
        f"AUC/log-loss gates pass on {auc_pass}/5 and {log_loss_pass}/5 edges, respectively.",
        "",
        "Passing would only make the mechanism eligible for extra-seed and runtime validation; it does not automatically admit a scale action. Failure is retained as a mechanism negative and does not invalidate the signed causal structure result.",
        "",
    ])
    atomic_text(OUTPUT / "formal/report.md", "\n".join(lines))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary-only", action="store_true")
    parser.add_argument("--edges", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    if any(edge not in range(1, 6) for edge in args.edges):
        raise ValueError("edges must be in 1..5")
    contract, contract_hash = verify_contract()
    frozen = contract["frozen_inputs"]
    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "OMP_NUM_THREADS": "2",
        "PYTHONUNBUFFERED": "1",
    }

    for edge_number in range(1, 6):
        edge_index = edge_number - 1
        edge = EDGES[edge_index]
        output = OUTPUT / "canary" / edge
        report = output / "adjudication.json"
        if report.exists():
            continue
        if output.exists():
            raise RuntimeError(f"partial canary requires audit: {output}")
        evaluate_edge(
            edge_index=edge_index,
            output=output,
            env=env,
            max_users=8,
            formal=False,
            design0_hash=frozen["Design0_formal_raw"][edge]["sha256"],
        )
    canary = canary_summary(contract_hash)
    if canary["status"] != "evidence_measure_basis_canary_passed" or args.canary_only:
        return

    for edge_number in args.edges:
        edge_index = edge_number - 1
        edge = EDGES[edge_index]
        output = OUTPUT / "formal/eval_14d" / edge
        report = output / "adjudication.json"
        if report.exists():
            continue
        if output.exists():
            raise RuntimeError(f"partial formal evaluation requires audit: {output}")
        evaluate_edge(
            edge_index=edge_index,
            output=output,
            env=env,
            max_users=0,
            formal=True,
            design0_hash=frozen["Design0_formal_raw"][edge]["sha256"],
        )
        completed = [
            candidate for candidate in EDGES
            if (OUTPUT / "formal/eval_14d" / candidate / "adjudication.json").exists()
        ]
        atomic_text(OUTPUT / "formal/progress.json", json.dumps({
            "completed_edges": completed,
            "contract_sha256": contract_hash,
        }, indent=2) + "\n")
    if all((OUTPUT / "formal/eval_14d" / edge / "adjudication.json").exists() for edge in EDGES):
        formal_summary(contract_hash)


if __name__ == "__main__":
    main()
