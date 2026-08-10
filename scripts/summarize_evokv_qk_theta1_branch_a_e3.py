from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.qk_stream_version import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--alignment-config", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    return parser.parse_args()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _preferred_protocol_rows(gate: dict[str, object]) -> list[dict[str, object]]:
    return [
        value
        for value in gate["all_checked"]
        if value["negative_count"] == 99
    ]


def summarize(
    config_path: Path,
    alignment_config_path: Path,
    protocol_config_path: Path,
) -> dict[str, object]:
    config = json.loads(config_path.read_text())
    alignment_config = json.loads(alignment_config_path.read_text())
    protocol_config = json.loads(protocol_config_path.read_text())
    training = json.loads(Path(config["outputs"]["result"]).read_text())
    alignment = json.loads(
        Path(alignment_config["outputs"]["result"]).read_text()
    )
    protocol = json.loads(
        Path(protocol_config["outputs"]["result"]).read_text()
    )
    if (
        training.get("status") != "complete_tuning_measurement"
        or alignment.get("status") != "complete_development_measurement"
        or protocol.get("status") != "complete_development_measurement"
    ):
        raise ValueError("QK branch-A result set is incomplete")
    tuning = training["quality"]["tuning"]
    alignment_evaluation = alignment["quality"]["evaluation"]
    alignment_gate = alignment_evaluation["alignment_gate"]
    protocol_gate = protocol["quality"]["primary_update_local"][
        "stable_gap_gate"
    ]
    rolling_all = alignment_evaluation["summary"]["rolling_next_item"][
        "cohorts"
    ]["all"]
    epoch_stats = training["training"]["epochs"]
    result = {
        "protocol": "evokv_qk_theta1_branch_a_summary_v0",
        "status": "complete_development_measurement",
        "scientific_result": False,
        "formal_result": False,
        "branch": "A",
        "candidate": config["edge"]["candidate_name"],
        "training": {
            "epochs": config["training"]["epochs"],
            "total_steps": training["training"]["total_steps"],
            "total_targets": sum(
                int(value["global_targets"]) for value in epoch_stats
            ),
            "epoch_mean_losses": [
                value["global_mean_loss"] for value in epoch_stats
            ],
            "optimizer_updated_rows": training["training"][
                "optimizer_active_delta"
            ]["global_updated_rows"],
            "runtime_seconds": training["training"]["runtime_seconds"],
        },
        "fit_tuning_full_catalog": {
            "records": tuning["records"],
            "positive_targets": tuning["positive_targets"],
            "reuse": tuning["reuse"],
            "recompute": tuning["recompute"],
            "gaps": tuning["gaps"],
        },
        "update_local_full_catalog_rolling_all": {
            "records": rolling_all["records"],
            "positive_targets": rolling_all["positive_targets"],
            "reuse": rolling_all["reuse"],
            "recompute": rolling_all["recompute"],
            "gaps": rolling_all["gaps"],
        },
        "alignment_gate": alignment_gate,
        "candidate_protocol_gate": {
            **protocol_gate,
            "preferred_99_negative_rows": _preferred_protocol_rows(
                protocol_gate
            ),
        },
        "target_gap_found": bool(
            alignment_gate["admitted"] or protocol_gate["admitted"]
        ),
        "selection_deferred": True,
        "qualification_consumed": False,
        "final_consumed": False,
        "artifacts": {
            "training_config": {
                "path": str(config_path),
                "sha256": file_sha256(config_path),
            },
            "alignment_config": {
                "path": str(alignment_config_path),
                "sha256": file_sha256(alignment_config_path),
            },
            "protocol_config": {
                "path": str(protocol_config_path),
                "sha256": file_sha256(protocol_config_path),
            },
        },
    }
    return result


def _markdown(result: dict[str, object]) -> str:
    rolling = result["update_local_full_catalog_rolling_all"]
    rows = [
        "# QK theta1 Branch A e3 Summary",
        "",
        f"- Target gap found: `{result['target_gap_found']}`",
        f"- Training targets: `{result['training']['total_targets']:,}`",
        f"- Optimizer-updated embedding rows: `{result['training']['optimizer_updated_rows']:,}`",
        "- Qualification/final consumed: `False/False`",
        "",
        "## Update-local full-catalog rolling-next-item, all participants",
        "",
        "| Metric | Reuse | Recompute | Relative gap % | CI positive |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in ("cross_entropy", "ndcg_at_10", "mrr", "hit_rate_at_10"):
        gap = rolling["gaps"][metric]
        rows.append(
            f"| {metric} | {rolling['reuse'][metric]:.8g} | "
            f"{rolling['recompute'][metric]:.8g} | "
            f"{gap['relative_percent']:.6g} | "
            f"{gap['positive_direction_with_ci']} |"
        )
    rows.extend(
        [
            "",
            "## Predeclared gates",
            "",
            f"- Full-catalog alignment: `{result['alignment_gate']['status']}`",
            f"- Candidate protocol sweep: `{result['candidate_protocol_gate']['status']}`",
            "- Candidate selection remains manual after this complete development round.",
        ]
    )
    return "\n".join(rows) + "\n"


def main() -> None:
    args = parse_args()
    result = summarize(
        args.config,
        args.alignment_config,
        args.protocol_config,
    )
    config = json.loads(args.config.read_text())
    _atomic_text(
        Path(config["outputs"]["summary_json"]),
        json.dumps(result, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(
        Path(config["outputs"]["summary_markdown"]),
        _markdown(result),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
