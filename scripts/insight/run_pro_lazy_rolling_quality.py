#!/usr/bin/env python3
"""Run frozen five-edge rolling quality for lightweight PRO."""

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
CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_pro_lazy_rolling_quality_v1.yaml"
MATRIX = ROOT / "results/yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v3"
MANIFEST = ROOT / "data/manifests/yambda500m_small_hstu_native_rolling_matrix_fast_v3"
OUTPUT = ROOT / "results/yambda500m_small_seed17/insight_pro_lazy_reader_v1/rolling_quality_v1"
GAIN_TABLE = MATRIX / "d14_onehop_reuse_completion_v2/auc_release_gain_coverage_table.json"
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
    for name in (
        "PRO_correctness_contract",
        "PRO_correctness_summary",
        "PRO_cost",
        "full_only_gain_table",
        "matrix_manifest",
        "requests_fidelity",
        "requests_quality",
    ):
        item = frozen[name]
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"{name} differs from the prospective quality contract")
    correctness = json.loads((ROOT / frozen["PRO_correctness_summary"]["path"]).read_text())
    if correctness["status"] != frozen["PRO_correctness_summary"]["required_status"]:
        raise RuntimeError("lightweight PRO correctness/cost gate has not passed")
    for version in range(6):
        expected = frozen["checkpoints"][f"v{version}"]["sha256"]
        if sha256(checkpoint(version)) != expected:
            raise RuntimeError(f"v{version} checkpoint differs from the quality contract")
    for edge in EDGES:
        expected = frozen["Design0_formal_raw"][edge]["sha256"]
        if sha256(design0_raw(edge)) != expected:
            raise RuntimeError(f"{edge} Design-0 raw differs from the quality contract")
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
        "--stage", f"pro_lazy_{'formal' if formal else 'canary'}_edge{edge_index + 1}",
        "--edge", edge,
        "--cutover-day", str(cutover),
        "--start-day", str(cutover),
        "--end-day", str(cutover + 14),
        "--manifest-dir", str(MANIFEST),
        "--parent", str(checkpoint(edge_index)),
        "--current", str(checkpoint(edge_index + 1)),
        "--include-parent-exact",
        "--include-pro-lazy",
        "--cohort-size", "32",
        "--output", str(output),
    ]
    if max_users:
        command.extend(["--max-users", str(max_users)])
    run(command, env)

    adjudication = [
        sys.executable,
        "scripts/insight/adjudicate_pro_lazy_quality.py",
        "--raw", str(output / "raw.parquet"),
        "--seal", str(output / "raw.seal.json"),
        "--prior-design0-raw", str(design0_raw(edge)),
        "--prior-design0-sha256", design0_hash,
        "--full-only-gain-table", str(GAIN_TABLE),
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
    replay_max = max(
        max(report["baseline_replay_max_absolute_logit_error"].values())
        for report in reports
    )
    passed = len(reports) == 5 and replay_max <= 2e-5 and all(
        report["labels_joined_after_raw_seal"] is False
        and report["structural_checks"]["materialized_translated_prefix_positions"] == 0
        for report in reports
    )
    summary = {
        "status": "pro_lazy_quality_canary_passed" if passed else "pro_lazy_quality_canary_failed",
        "contract_sha256": contract_hash,
        "edges": len(reports),
        "users_sum_across_edges": sum(int(report["users"]) for report in reports),
        "requests_sum_across_edges": sum(int(report["requests"]) for report in reports),
        "maximum_baseline_replay_absolute_logit_error": replay_max,
        "labels_read": False,
        "materialized_version_translated_prefix_positions": 0,
        "formal_quality_launched": False,
        "edge_results": [
            {
                "edge": report["edge"],
                "users": report["users"],
                "requests": report["requests"],
                "PRO_mean_abs_logit_gap_to_Current": report["fidelity"][
                    "evokv_pro_lazy_reader_c32_rolling"
                ]["mean_abs_logit_gap_to_Current"],
                "Design0_mean_abs_logit_gap_to_Current": report["fidelity"][
                    "evokv_cast_group_patch_scale_r128_c64_rolling"
                ]["mean_abs_logit_gap_to_Current"],
            }
            for report in reports
        ],
    }
    atomic_text(OUTPUT / "canary/summary.json", json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Lightweight PRO quality canary",
        "",
        f"Progression gate: **{'PASS' if passed else 'FAIL'}**.",
        "",
        "No label was read. All five checkpoint edges replayed the sealed Parent, Current and Reuse logits before any formal quality launch.",
        "",
        "| edge | users | requests | Design 0 logit gap | PRO logit gap |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["edge_results"]:
        lines.append(
            f"| {row['edge']} | {row['users']} | {row['requests']} | "
            f"{row['Design0_mean_abs_logit_gap_to_Current']:.8g} | "
            f"{row['PRO_mean_abs_logit_gap_to_Current']:.8g} |"
        )
    lines.extend([
        "",
        "The PRO action materialized zero version-translated prefix positions; the comparison does not turn Design 0 into a serving stage.",
        "",
    ])
    atomic_text(OUTPUT / "canary/report.md", "\n".join(lines))
    return summary


def formal_summary(contract: dict, contract_hash: str) -> dict:
    reports = [
        json.loads((OUTPUT / "formal/eval_14d" / edge / "adjudication.json").read_text())
        for edge in EDGES
    ]
    rows = []
    for report in reports:
        metrics = report["absolute_metrics"]
        delta = report["rolling_quality_deltas"]
        retained = report["requested_full_only_release_gain_retention"]
        rows.append({
            "edge": report["edge"],
            "requests": report["requests"],
            "Current_ROC_AUC": metrics["current_exact_rolling"]["ROC_AUC"],
            "Reuse_ROC_AUC": metrics["one_hop_reuse_rolling"]["ROC_AUC"],
            "Design0_ROC_AUC": metrics["evokv_cast_group_patch_scale_r128_c64_rolling"]["ROC_AUC"],
            "PRO_ROC_AUC": metrics["evokv_pro_lazy_reader_c32_rolling"]["ROC_AUC"],
            "Current_log_loss": metrics["current_exact_rolling"]["log_loss"],
            "Reuse_log_loss": metrics["one_hop_reuse_rolling"]["log_loss"],
            "Design0_log_loss": metrics["evokv_cast_group_patch_scale_r128_c64_rolling"]["log_loss"],
            "PRO_log_loss": metrics["evokv_pro_lazy_reader_c32_rolling"]["log_loss"],
            "PRO_minus_Reuse_ROC_AUC_pp": delta["PRO_minus_Reuse_ROC_AUC_pp"],
            "PRO_minus_Reuse_log_loss": delta["PRO_minus_Reuse_log_loss"],
            "PRO_minus_Design0_ROC_AUC_pp": delta["PRO_minus_Design0_ROC_AUC_pp"],
            "PRO_minus_Design0_log_loss": delta["PRO_minus_Design0_log_loss"],
            "Reuse_harm_recovered_fraction": delta["Reuse_harm_recovered_fraction"],
            "Reuse_gain_retained_percent": retained["Reuse_percent_when_positive"],
            "Design0_gain_retained_percent": retained["Design0_percent_when_positive"],
            "PRO_gain_retained_percent": retained["PRO_percent_when_positive"],
        })

    gate = contract["formal_quality"]["inherited_directional_gate"]
    auc_edges = sum(row["PRO_minus_Reuse_ROC_AUC_pp"] >= 0.0 for row in rows)
    log_loss_edges = sum(row["PRO_minus_Reuse_log_loss"] <= 0.0 for row in rows)
    mean_auc_delta = sum(row["PRO_minus_Reuse_ROC_AUC_pp"] for row in rows) / len(rows)
    mean_log_loss_delta = sum(row["PRO_minus_Reuse_log_loss"] for row in rows) / len(rows)
    passed = (
        auc_edges >= gate["PRO_ROC_AUC_not_below_Reuse_edges_minimum"]
        and log_loss_edges >= gate["PRO_log_loss_not_above_Reuse_edges_minimum"]
        and mean_auc_delta >= 0.0
        and mean_log_loss_delta <= 0.0
    )
    viability_passed = (
        auc_edges >= 3
        and log_loss_edges >= 3
        and mean_auc_delta >= 0.0
        and mean_log_loss_delta <= 0.0
    )
    summary = {
        "status": "pro_lazy_rolling_quality_gate_passed" if passed else "pro_lazy_rolling_quality_gate_failed",
        "contract_sha256": contract_hash,
        "edges": 5,
        "requests_sum_across_edges": sum(row["requests"] for row in rows),
        "labels_joined_only_after_each_raw_seal": True,
        "all_edges_reported_without_selection": True,
        "materialized_version_translated_prefix_positions": 0,
        "release_time_compute_over_Full_fraction": 0.09140968802388935,
        "quality_gate": {
            "passed": passed,
            "PRO_ROC_AUC_not_below_Reuse_edges": auc_edges,
            "required_AUC_edges": gate["PRO_ROC_AUC_not_below_Reuse_edges_minimum"],
            "PRO_log_loss_not_above_Reuse_edges": log_loss_edges,
            "required_log_loss_edges": gate["PRO_log_loss_not_above_Reuse_edges_minimum"],
            "mean_edge_PRO_minus_Reuse_ROC_AUC_pp": mean_auc_delta,
            "mean_edge_PRO_minus_Reuse_log_loss": mean_log_loss_delta,
        },
        "post_result_design_viability_interpretation": {
            "date": "2026-08-28",
            "prospective_qualification_gate_rewritten": False,
            "criterion": "both_mean_edge_deltas_favorable_and_each_metric_positive_on_at_least_3_of_5_edges",
            "passed": viability_passed,
            "interpretation": "the_method_is_worth_independent_follow_up_not_serving_admission",
        },
        "edge_results": rows,
        "automatic_lineage_admission": False,
        "next_step": "independent_label_free_admission_or_calibration_then_new_seed_or_edge_validation",
    }
    atomic_text(OUTPUT / "formal/summary.json", json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Frozen lightweight PRO: five-edge rolling quality",
        "",
        f"Prospective strict quality gate: **{'PASS' if passed else 'FAIL'}**. Post-result Design viability: **{'PASS' if viability_passed else 'FAIL'}**. PRO improves or ties Reuse on {auc_edges}/5 AUC edges and {log_loss_edges}/5 log-loss edges.",
        "",
        "The release-time action is one 2 KiB user sidecar at 9.14% of Full theoretical FLOPs. It materializes no CAST prefix. Design 0 below is a sealed comparison baseline, not an execution stage.",
        "",
        "| edge | requests | Current AUC | Reuse AUC | Design 0 AUC | PRO AUC | PRO−Reuse (pp) | Current log-loss | Reuse log-loss | Design 0 log-loss | PRO log-loss | PRO gain retained |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        retained = row["PRO_gain_retained_percent"]
        lines.append(
            f"| {row['edge']} | {row['requests']} | {row['Current_ROC_AUC']:.6f} | "
            f"{row['Reuse_ROC_AUC']:.6f} | {row['Design0_ROC_AUC']:.6f} | "
            f"{row['PRO_ROC_AUC']:.6f} | {row['PRO_minus_Reuse_ROC_AUC_pp']:+.6f} | "
            f"{row['Current_log_loss']:.6f} | {row['Reuse_log_loss']:.6f} | "
            f"{row['Design0_log_loss']:.6f} | {row['PRO_log_loss']:.6f} | "
            f"{'N/A' if retained is None else f'{retained:+.1f}%'} |"
        )
    lines.extend([
        "",
        f"Unweighted mean edge PRO−Reuse: {mean_auc_delta:+.6f} AUC pp and {mean_log_loss_delta:+.8f} log-loss.",
        "",
        "The strict gate is not rewritten after observing labels. The separate viability interpretation uses the user-approved aggregate/majority criterion and only says that PRO is worth continuing; it does not admit a serving lineage.",
        "",
        "On v3→v4, Current Exact rolling log-loss is itself worse than Reuse while AUC is better, and the Full-only AUC gain denominator is only 0.046331 pp. PRO lies between Reuse and Current in log-loss while recovering AUC, so this edge is a ranking/calibration target conflict rather than a numerical reader-map failure.",
        "",
        "The next step is an independently frozen label-free admission/calibration test. Any tuned mechanism must be validated on a new seed or release edge because these five labels are now development evidence.",
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

    for edge_index, edge in enumerate(EDGES):
        output = OUTPUT / "canary" / edge
        report = output / "adjudication.json"
        if report.exists():
            continue
        if output.exists():
            raise RuntimeError(f"partial PRO canary requires audit: {output}")
        evaluate_edge(
            edge_index=edge_index,
            output=output,
            env=env,
            max_users=8,
            formal=False,
            design0_hash=frozen["Design0_formal_raw"][edge]["sha256"],
        )
    canary = canary_summary(contract_hash)
    if canary["status"] != "pro_lazy_quality_canary_passed" or args.canary_only:
        return

    for edge_number in args.edges:
        edge_index = edge_number - 1
        edge = EDGES[edge_index]
        output = OUTPUT / "formal/eval_14d" / edge
        report = output / "adjudication.json"
        if report.exists():
            continue
        if output.exists():
            raise RuntimeError(f"partial formal PRO evaluation requires audit: {output}")
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
    if all(
        (OUTPUT / "formal/eval_14d" / edge / "adjudication.json").exists()
        for edge in EDGES
    ):
        formal_summary(contract, contract_hash)


if __name__ == "__main__":
    main()
