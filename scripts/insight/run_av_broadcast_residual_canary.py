#!/usr/bin/env python3
"""Run and adjudicate the frozen five-edge AV broadcast-residual score canary."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_av_broadcast_residual_v1.yaml"
MANIFEST = ROOT / "data/manifests/yambda500m_small_hstu_native_rolling_matrix_fast_v3"
POPULATION = ROOT / "results/yambda500m_small_seed17/insight_recommendation_state_structure_v1/population.parquet"
MATRIX = ROOT / "results/yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v3"
OUTPUT = ROOT / "results/yambda500m_small_seed17/insight_av_broadcast_residual_v1/canary"
EDGES = ("v0_to_v1", "v1_to_v2", "v2_to_v3", "v3_to_v4", "v4_to_v5")
DESIGN0 = "evokv_cast_group_patch_scale_r128_c64_rolling"
MECHANISM = "compact_probe_AV_broadcast_residual_rolling"


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


def verify_contract() -> tuple[dict, str]:
    contract = yaml.safe_load(CONTRACT.read_text())
    for name, record in contract["frozen_inputs"].items():
        if name in {"checkpoints", "Design0_formal_raw"}:
            continue
        path = ROOT / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"frozen {name} differs")
        required = record.get("required_status")
        if required and json.loads(path.read_text())["status"] != required:
            raise RuntimeError(f"frozen {name} status differs")
    for version in range(6):
        record = contract["frozen_inputs"]["checkpoints"][f"v{version}"]
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"frozen v{version} checkpoint differs")
    for edge in EDGES:
        record = contract["frozen_inputs"]["Design0_formal_raw"][edge]
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"frozen {edge} Design 0 raw differs")
    return contract, sha256(CONTRACT)


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def evaluate(edge_index: int, env: dict[str, str]) -> None:
    edge = EDGES[edge_index]
    target = OUTPUT / edge
    if (target / "raw.seal.json").exists():
        return
    if target.exists():
        raise RuntimeError(f"partial mechanism canary requires audit: {target}")
    cutover = 231 + 14 * edge_index
    run(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node=4",
            "scripts/insight/evaluate_av_broadcast_residual_canary.py",
            "--edge",
            edge,
            "--cutover-day",
            str(cutover),
            "--end-day",
            str(cutover + 14),
            "--manifest-dir",
            str(MANIFEST),
            "--population",
            str(POPULATION),
            "--parent",
            str(checkpoint(edge_index)),
            "--current",
            str(checkpoint(edge_index + 1)),
            "--output",
            str(target),
            "--max-users",
            "8",
        ],
        env,
    )
    for transient in (target / ".directory_ready", target / ".raw_complete"):
        transient.unlink(missing_ok=True)


def adjudicate_edge(edge: str) -> dict:
    target = OUTPUT / edge
    raw = pd.read_parquet(target / "raw.parquet")
    prior = pd.read_parquet(design0_raw(edge))
    selected_ids = set(raw.request_id.astype(str))
    prior = prior[prior.request_id.astype(str).isin(selected_ids)]
    prior_focus = prior[
        prior.path.isin(["current_exact_rolling", "one_hop_reuse_rolling", DESIGN0])
    ][["request_id", "path", "hstu_logit"]].copy()
    replay_focus = raw[
        raw.path.isin(["current_exact_rolling", "one_hop_reuse_rolling", DESIGN0])
    ][["request_id", "path", "hstu_logit"]].copy()
    joined = replay_focus.merge(
        prior_focus,
        on=["request_id", "path"],
        suffixes=("_replay", "_prior"),
        validate="one_to_one",
    )
    if len(joined) != 3 * len(selected_ids):
        raise RuntimeError(f"{edge} prior Design 0 raw request set is incomplete")
    replay_error = float(
        (joined.hstu_logit_replay - joined.hstu_logit_prior).abs().max()
    )
    pivot = raw.pivot(index="request_id", columns="path", values="hstu_logit")
    design0_gap = float((pivot[DESIGN0] - pivot["current_exact_rolling"]).abs().mean())
    mechanism_gap = float(
        (pivot[MECHANISM] - pivot["current_exact_rolling"]).abs().mean()
    )
    seal = json.loads((target / "raw.seal.json").read_text())
    passed = mechanism_gap <= design0_gap
    result = {
        "edge": edge,
        "users": seal["users"],
        "requests": seal["requests"],
        "labels_read": False,
        "prior_replay_max_abs_error": replay_error,
        "probe_replay_max_abs_error": seal["probe_replay_max_abs_error"],
        "Design0_mean_abs_logit_gap": design0_gap,
        "mechanism_mean_abs_logit_gap": mechanism_gap,
        "mechanism_relative_gap_change": mechanism_gap / design0_gap - 1.0,
        "mechanism_not_worse_than_Design0": passed,
        "raw_sha256": seal["raw_sha256"],
    }
    (target / "adjudication.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def emit_summary(contract_hash: str) -> dict:
    results = [adjudicate_edge(edge) for edge in EDGES]
    correctness = max(
        max(row["prior_replay_max_abs_error"], row["probe_replay_max_abs_error"])
        for row in results
    )
    passed_edges = sum(row["mechanism_not_worse_than_Design0"] for row in results)
    passed = correctness <= 2e-5 and passed_edges >= 4
    summary = {
        "status": (
            "av_broadcast_residual_score_canary_passed"
            if passed
            else "av_broadcast_residual_score_canary_failed"
        ),
        "contract_sha256": contract_hash,
        "edges": 5,
        "users_sum_across_edges": sum(row["users"] for row in results),
        "requests_sum_across_edges": sum(row["requests"] for row in results),
        "labels_read": False,
        "correctness_max_abs_error": correctness,
        "mechanism_not_worse_than_Design0_edges": passed_edges,
        "required_edges": 4,
        "formal_quality_launched": False,
        "action_admitted": False,
        "matched_cost": {
            "joint_KV_CAST_positions": 384,
            "Design0_Current_carriers": 64,
            "mechanism_Current_carriers": 32,
            "mechanism_probe_reader_paths": 2,
            "Design0_incremental_attention_pairs_per_layer": 26656,
            "mechanism_incremental_attention_pairs_per_layer": 13840,
            "raw_repair_positions": 128,
            "persistent_sidecar_scalars": 512,
        },
        "edge_results": results,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Compact-probe AV broadcast-residual score canary",
        "",
        f"Progression gate: **{'PASS' if passed else 'FAIL'}** ({passed_edges}/5 edges); labels read: **no**.",
        "",
        "The sidecar is generated once from a fixed latest-history-item probe over a 32-carrier disposable Current source, then coverage-scaled and broadcast at every AV layer. It is not a target-K/V fit or a per-candidate selector.",
        "",
        "| edge | requests | Design 0 logit gap | AV sidecar logit gap | relative change | not worse |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in results:
        lines.append(
            f"| {row['edge']} | {row['requests']} | {row['Design0_mean_abs_logit_gap']:.8g} | "
            f"{row['mechanism_mean_abs_logit_gap']:.8g} | "
            f"{100 * row['mechanism_relative_gap_change']:+.2f}% | "
            f"{row['mechanism_not_worse_than_Design0']} |"
        )
    lines.extend(
        [
            "",
            f"Maximum baseline/probe replay error: {correctness:.8g}.",
            "",
            "Per the prospective contract, this canary does not launch formal quality or admit an action regardless of pass/fail; the result returns to expert discussion.",
            "",
        ]
    )
    (OUTPUT / "report.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    _, contract_hash = verify_contract()
    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "OMP_NUM_THREADS": "2",
        "PYTHONUNBUFFERED": "1",
    }
    for edge_index in range(5):
        evaluate(edge_index, env)
    emit_summary(contract_hash)


if __name__ == "__main__":
    main()
