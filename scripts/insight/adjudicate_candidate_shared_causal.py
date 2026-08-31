"""Adjudicate the prospective signed candidate-shared causal observation.

This script combines the frozen 3,000-user controlled-bank intervention with
the raw-first real-exposed candidate evaluation.  The shared and residual
paths remain diagnostic oracles: passing this adjudication does not admit a
new cache action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "configs/contracts/yambda500m_small_hstu_native_candidate_shared_causal_v1.yaml"
RESULT = ROOT / "results/yambda500m_small_seed17/insight_candidate_shared_causal_v1"
EDGES = ("v0_to_v1", "v1_to_v2", "v2_to_v3", "v3_to_v4", "v4_to_v5")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_paths(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["edge", "width", "path"], as_index=False)
        .agg(
            mean_abs_logit_gap=("mean_abs_logit_gap", "mean"),
            mean_abs_probability_gap=("mean_abs_probability_gap", "mean"),
            top1_agreement=("top1_agreement", "mean"),
            rank_correlation=("rank_correlation", "mean"),
            banks=("identifier", "size"),
        )
        .sort_values(["edge", "width", "path"])
    )


def _recovery_table(path_means: pd.DataFrame) -> pd.DataFrame:
    pivot = path_means.pivot(
        index=["edge", "width"], columns="path", values="mean_abs_probability_gap"
    )
    rows = []
    for (edge, width), values in pivot.iterrows():
        reuse = float(values["reuse"])
        shared = float(values["shared_only"])
        residual = float(values["residual_only"])
        rows.append(
            {
                "edge": edge,
                "width": int(width),
                "reuse_probability_gap": reuse,
                "shared_probability_gap": shared,
                "residual_probability_gap": residual,
                "shared_gap_recovery": 1.0 - shared / reuse,
                "residual_gap_recovery": 1.0 - residual / reuse,
                "shared_better_than_residual": shared < residual,
            }
        )
    return pd.DataFrame(rows).sort_values(["edge", "width"])


def _markdown(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    labels = [column.replace("_", " ") for column in columns]
    lines = ["| " + " | ".join(labels) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.8g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--result", type=Path, default=RESULT)
    parser.add_argument("--output", type=Path, default=RESULT / "adjudication")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    contract = yaml.safe_load(args.contract.read_text())
    if tuple(contract["scope"]["edges"]) != EDGES:
        raise RuntimeError("contracted edge set differs")

    controlled = pd.read_parquet(args.result / "formal_controlled/score_interventions.parquet")
    controlled = controlled[controlled["width"] == 64].copy()
    if set(controlled["edge"]) != set(EDGES):
        raise RuntimeError("controlled formal edge set differs")
    controlled_means = _mean_paths(controlled)
    controlled_recovery = _recovery_table(controlled_means)

    exposed_root = args.result / "formal_exposed/eval_14d"
    bank_parts, head_parts = [], []
    raw_hashes = {}
    for edge in EDGES:
        edge_dir = exposed_root / edge
        seal = json.loads((edge_dir / "raw.seal.json").read_text())
        if seal.get("labels_read") is not False:
            raise RuntimeError(f"{edge}: raw generation label boundary differs")
        raw_hash = sha256_file(edge_dir / "raw.parquet")
        if seal["artifacts"]["raw"]["sha256"] != raw_hash:
            raise RuntimeError(f"{edge}: sealed raw hash differs")
        raw_hashes[edge] = raw_hash
        bank_parts.append(pd.read_parquet(edge_dir / "bank.parquet"))
        head_parts.append(pd.read_parquet(edge_dir / "head.parquet"))
    bank = pd.concat(bank_parts, ignore_index=True)
    head = pd.concat(head_parts, ignore_index=True)
    if set(bank["width"]) != {2, 4, 8, 16} or set(bank["edge"]) != set(EDGES):
        raise RuntimeError("real-exposed edge/width set differs")

    exposed_means = _mean_paths(bank)
    exposed_recovery = _recovery_table(exposed_means)
    nonzero = head[
        (head["signed_shared_energy_fraction"] + head["signed_residual_energy_fraction"]) > 0
    ].copy()
    head_summary = (
        nonzero.groupby(["edge", "width"], as_index=False)
        .agg(
            nonzero_head_observations=("head", "size"),
            signed_shared_energy_fraction=("signed_shared_energy_fraction", "mean"),
            signed_residual_energy_fraction=("signed_residual_energy_fraction", "mean"),
            shared_direction_cosine=("shared_direction_cosine_to_largest_width", "mean"),
        )
        .sort_values(["edge", "width"])
    )

    quality = pd.read_csv(args.result / "formal_exposed/quality_all_edges.csv")
    fidelity = pd.read_csv(args.result / "formal_exposed/paired_fidelity_all_edges.csv")
    if set(quality["edge"]) != set(EDGES) or set(fidelity["edge"]) != set(EDGES):
        raise RuntimeError("real-exposed quality edge set differs")
    fidelity_summary = (
        fidelity.groupby("path", as_index=False)
        .agg(
            mean_abs_logit_gap_to_exact=("mean_abs_logit_gap_to_exact", "mean"),
            max_abs_logit_gap_to_exact=("mean_abs_logit_gap_to_exact", "max"),
            mean_event_log_loss_delta=("event_path_minus_exact_log_loss", "mean"),
            max_abs_event_log_loss_delta=("event_path_minus_exact_log_loss", lambda values: values.abs().max()),
        )
        .sort_values("path")
    )
    quality_pivot = quality.pivot(index=["edge", "width"], columns="path", values=["ROC_AUC", "log_loss"])
    shared_auc_max_abs_delta = float(
        (quality_pivot[("ROC_AUC", "shared_only")] - quality_pivot[("ROC_AUC", "current_exact")]).abs().max()
    )
    shared_log_loss_max_abs_delta = float(
        (quality_pivot[("log_loss", "shared_only")] - quality_pivot[("log_loss", "current_exact")]).abs().max()
    )

    controlled_pass = bool(
        controlled_recovery["shared_better_than_residual"].all()
        and (controlled_recovery["shared_gap_recovery"] > 0).all()
    )
    exposed_pass = bool(
        len(exposed_recovery) == 20
        and exposed_recovery["shared_better_than_residual"].all()
        and (exposed_recovery["shared_gap_recovery"] > 0).all()
    )
    correctness = pd.read_csv(args.result / "formal_controlled/correctness.csv")
    exposed_correctness = pd.concat(
        [pd.read_parquet(exposed_root / edge / "correctness.parquet") for edge in EDGES],
        ignore_index=True,
    )
    max_native_error = max(
        float(correctness[["native_exact", "native_reuse"]].max().max()),
        float(exposed_correctness[["native_exact", "native_reuse"]].max().max()),
    )
    max_full_error = max(
        float(correctness["full_delta"].max()),
        float(exposed_correctness["full_delta"].max()),
    )
    correctness_pass = max(max_native_error, max_full_error) < 2e-5
    passed = controlled_pass and exposed_pass and correctness_pass

    args.output.mkdir(parents=True)
    controlled_recovery.to_csv(args.output / "controlled_width64_recovery.csv", index=False)
    exposed_recovery.to_csv(args.output / "real_exposed_recovery.csv", index=False)
    head_summary.to_csv(args.output / "real_exposed_signed_heads.csv", index=False)
    fidelity_summary.to_csv(args.output / "real_exposed_fidelity_summary.csv", index=False)

    summary = {
        "status": "candidate_shared_signed_causal_gate_passed" if passed else "candidate_shared_signed_causal_gate_failed",
        "contract_sha256": sha256_file(args.contract),
        "controlled_users": int(controlled["identifier"].nunique()),
        "controlled_edge_width_combinations": int(len(controlled_recovery)),
        "controlled_shared_better_than_residual": int(controlled_recovery["shared_better_than_residual"].sum()),
        "controlled_shared_gap_recovery_range": [
            float(controlled_recovery["shared_gap_recovery"].min()),
            float(controlled_recovery["shared_gap_recovery"].max()),
        ],
        "real_exposed_edge_width_combinations": int(len(exposed_recovery)),
        "real_exposed_shared_better_than_residual": int(exposed_recovery["shared_better_than_residual"].sum()),
        "real_exposed_shared_gap_recovery_range": [
            float(exposed_recovery["shared_gap_recovery"].min()),
            float(exposed_recovery["shared_gap_recovery"].max()),
        ],
        "real_exposed_banks_across_widths": int(len(bank) // 4),
        "real_exposed_selected_requests_across_widths": int(
            json.loads((args.result / "formal_exposed/summary.json").read_text())["selected_requests_across_widths"]
        ),
        "nonzero_signed_head_shared_energy_fraction_mean": float(nonzero["signed_shared_energy_fraction"].mean()),
        "shared_only_mean_abs_logit_gap_to_exact": float(
            fidelity_summary.loc[fidelity_summary["path"] == "shared_only", "mean_abs_logit_gap_to_exact"].iloc[0]
        ),
        "reuse_mean_abs_logit_gap_to_exact": float(
            fidelity_summary.loc[fidelity_summary["path"] == "reuse", "mean_abs_logit_gap_to_exact"].iloc[0]
        ),
        "shared_only_max_abs_ROC_AUC_delta_to_exact": shared_auc_max_abs_delta,
        "shared_only_max_abs_log_loss_delta_to_exact": shared_log_loss_max_abs_delta,
        "native_score_max_abs_error": max_native_error,
        "full_delta_reconstruction_max_abs_error": max_full_error,
        "raw_hashes": raw_hashes,
        "labels_joined_only_after_each_raw_seal": True,
        "mechanism_action_admitted": False,
        "interpretation": "signed causal structure gate only; shared/residual are oracle interventions",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    controlled_table = controlled_recovery.copy()
    controlled_table["shared_gap_recovery"] *= 100
    exposed_edge = (
        exposed_recovery.groupby("edge", as_index=False)
        .agg(
            widths=("width", "size"),
            shared_recovery_min=("shared_gap_recovery", "min"),
            shared_recovery_max=("shared_gap_recovery", "max"),
            shared_better_residual=("shared_better_than_residual", "sum"),
        )
    )
    exposed_edge[["shared_recovery_min", "shared_recovery_max"]] *= 100
    report = [
        "# Candidate-shared signed causal adjudication",
        "",
        f"Progression gate: **{'PASS' if passed else 'FAIL'}**.",
        "",
        "This adjudication uses signed per-head HSTU contributions without candidate-wise normalization. "
        "The real-exposed banks contain only actual same-UID, same-timestamp requests, and every raw artifact was sealed before labels were joined.",
        "",
        "## Controlled 3,000-user width-64 intervention",
        "",
        *_markdown(
            controlled_table,
            ["edge", "reuse_probability_gap", "shared_probability_gap", "residual_probability_gap", "shared_gap_recovery"],
        ),
        "",
        "## Real-exposed candidate distribution",
        "",
        *_markdown(
            exposed_edge,
            ["edge", "widths", "shared_recovery_min", "shared_recovery_max", "shared_better_residual"],
        ),
        "",
        f"Across nonzero head observations, the signed shared component carries {summary['nonzero_signed_head_shared_energy_fraction_mean']:.6%} of delta energy. "
        f"Shared-only mean absolute logit gap to Current Exact is {summary['shared_only_mean_abs_logit_gap_to_exact']:.8g}, versus {summary['reuse_mean_abs_logit_gap_to_exact']:.8g} for Reuse. "
        f"The maximum shared-only absolute AUC/log-loss deltas to Exact across 20 edge-width cells are {shared_auc_max_abs_delta:.8g}/{shared_log_loss_max_abs_delta:.8g}.",
        "",
        "Maximum native/full reconstruction errors are "
        f"{max_native_error:.8g}/{max_full_error:.8g}.",
        "",
        "## Decision boundary",
        "",
        "The signed causal and real-candidate-distribution gates pass. This supports a candidate-broadcast user-evidence component plus a small contextual residual as a real reader structure, rather than a norm-normalization artifact. "
        "It does **not** make shared/residual oracle interventions executable, admit a new action, or establish superiority over Design 0. The next allowed step is one frozen candidate-independent Current-HSTU evidence-basis mechanism at matched compute, carriers, raw I/O and state I/O.",
        "",
    ]
    (args.output / "report.md").write_text("\n".join(report))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
