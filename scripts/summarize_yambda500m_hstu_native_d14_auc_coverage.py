#!/usr/bin/env python3
"""Render the existing D=14 Full-only gain and sealed one-hop AUC coverage table.

This deliberately performs no evaluation.  It joins the already sealed D=14
Full-only release matrix with the already sealed one-hop rolling Reuse results.
The two references have different execution semantics, so the erasure ratio is
explicitly labelled as an observational cross-reference rather than rho_erase.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "results/yambda500m_small_seed17/hstu_native_rolling_recipe_matrix_v3"
REUSE = MATRIX / "d14_onehop_reuse_diagnostic_v1"
COMPLETION = MATRIX / "d14_onehop_reuse_completion_v2"
FULL_ONLY_COMPLETION = MATRIX / "d14_full_only_e2_completion_v1" / "train_14d"
HORIZONS = (1, 2, 4, 7, 14)
EDGES = tuple(f"v{index}_to_v{index + 1}" for index in range(5))


def main() -> None:
    release_rows = json.loads((MATRIX / "matrix_result.json").read_text())
    release_gain = {
        (row["edge"], int(row["evaluation_days"])): float(row["roc_auc_delta_pp"])
        for row in release_rows if int(row["training_days"]) == 14 and row["edge"] in EDGES
    }
    for report in FULL_ONLY_COMPLETION.glob("eval_2d/v*_to_v*/adjudication.json"):
        payload = json.loads(report.read_text())
        parent = payload["parent_absolute"]["hstu_native"]
        current = next(iter(payload["candidates"].values()))["absolute"]["hstu_native"]
        release_gain[(report.parent.name, 2)] = (float(current["ROC_AUC"]) - float(parent["ROC_AUC"])) * 100.0
    reuse_loss: dict[tuple[str, int], float] = {}
    for report in (*REUSE.glob("eval_*d/v*_to_v*/adjudication.json"), *COMPLETION.glob("eval_*d/v*_to_v*/adjudication.json")):
        payload = json.loads(report.read_text())
        days = int(payload["evaluation_day_range"][1]) - int(payload["evaluation_day_range"][0])
        reuse_loss[(payload["edge"], days)] = float(
            payload["reuse_minus_recompute"]["current_minus_reuse_ROC_AUC_pp"]
        )
    rows = []
    for edge in EDGES:
        for days in HORIZONS:
            gain = release_gain.get((edge, days))
            loss = reuse_loss.get((edge, days))
            signed_fraction = None if gain is None or loss is None or abs(gain) < 1e-12 else loss / gain
            positive_gain_percent = None if gain is None or loss is None or gain <= 0.0 else 100.0 * loss / gain
            rows.append({
                "edge": edge, "evaluation_days": days,
                "full_only_current_minus_parent_ROC_AUC_pp": gain,
                "rolling_current_minus_reuse_ROC_AUC_pp": loss,
                "cross_reference_signed_fraction": signed_fraction,
                "cross_reference_percent_when_release_gain_positive": positive_gain_percent,
            })
    lines = [
        "# D=14 AUC release gain and One-hop Reuse coverage",
        "",
        "This table uses only already sealed artifacts; it does not run another evaluation.",
        "",
        "- **Old → new gain** is `Current Full − Parent Full` ROC-AUC from the completed D=14 Full-only recipe matrix (pp). Positive means the new model is better.",
        "- **Reuse loss** is `Current Exact Rolling − One-hop Reuse Rolling` ROC-AUC from the sealed cache diagnostic (pp). Positive means Reuse is worse.",
        "- **Reuse / gain** is `Reuse loss / old→new gain`. It is reported as a percentage only for a positive old→new gain. For a non-positive gain it is `N/A`, since there is no positive release benefit to erase.",
        "- These first-pass ratios are **cross-reference observations**, not the strict matched-rolling `rho_erase`: the Full-only gain and rolling Reuse paths have different post-eviction semantics. They are suitable for deciding where to investigate; a formal erasure claim needs the three matched rolling paths.",
        "",
        "`—` means that the requested E had not been measured in the existing sealed artifacts; it is not a zero effect.",
        "",
        "| Edge | E | Old → new gain (pp) | Reuse loss (pp) | Reuse / gain |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        gain = row["full_only_current_minus_parent_ROC_AUC_pp"]
        loss = row["rolling_current_minus_reuse_ROC_AUC_pp"]
        ratio = row["cross_reference_percent_when_release_gain_positive"]
        lines.append(
            f"| {row['edge'].replace('_to_', ' → ')} | {row['evaluation_days']} | "
            f"{'—' if gain is None else f'{gain:+.6f}'} | "
            f"{'—' if loss is None else f'{loss:+.6f}'} | "
            f"{'—' if ratio is None and gain is None or loss is None else ('N/A' if ratio is None else f'{ratio:+.1f}%')} |"
        )
    lines.append("")
    for directory in (REUSE, COMPLETION):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "auc_release_gain_coverage_table.json").write_text(json.dumps(rows, indent=2) + "\n")
        (directory / "auc_release_gain_coverage_table.md").write_text("\n".join(lines))
    print(json.dumps({"rows": len(rows), "output": str(COMPLETION / "auc_release_gain_coverage_table.md")}, indent=2))


if __name__ == "__main__":
    main()
