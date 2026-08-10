from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.qk_stream_version import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
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


def summarize(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text())
    outputs = config["outputs"]
    alignment_config_path = Path(outputs["alignment_round_root"]) / "frozen_config.json"
    protocol_config_path = (
        Path(outputs["protocol_sweep_round_root"]) / "frozen_config.json"
    )
    alignment_config = json.loads(alignment_config_path.read_text())
    protocol_config = json.loads(protocol_config_path.read_text())
    training = json.loads(Path(outputs["result"]).read_text())
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
        raise ValueError("QK theta2 candidate result set is incomplete")
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
    core_stable = all(
        rolling_all["gaps"][metric]["positive_direction_with_ci"]
        for metric in ("ndcg_at_10", "mrr")
    )
    alignment_found = bool(alignment_gate["admitted"])
    result = {
        "protocol": "evokv_qk_theta2_route_a_candidate_summary_v0",
        "status": "complete_development_measurement",
        "scientific_result": False,
        "formal_result": False,
        "route": "A",
        "source_version": 1,
        "target_version": 2,
        "training_window": 2,
        "evaluation_window": 3,
        "candidate": config["edge"]["candidate_name"],
        "training": {
            "epochs": config["training"]["epochs"],
            "dense_learning_rate": config["training"][
                "dense_learning_rate"
            ],
            "projection_learning_rate": config["training"][
                "projection_learning_rate"
            ],
            "embedding_learning_rate": config["training"][
                "embedding_learning_rate"
            ],
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
        "selection_signals": {
            "core_ndcg_at_10_and_mrr_ci_positive": core_stable,
            "predeclared_alignment_candidate_found": alignment_found,
            "eligible_for_sweep_ranking": core_stable and alignment_found,
        },
        "selection_deferred": True,
        "qualification_consumed": False,
        "final_consumed": False,
        "checkpoint": training["checkpoint"],
        "artifacts": {
            "training_config": {
                "path": str(config_path),
                "sha256": file_sha256(config_path),
            },
            "training_result": {
                "path": outputs["result"],
                "sha256": file_sha256(Path(outputs["result"])),
            },
            "alignment_result": {
                "path": alignment_config["outputs"]["result"],
                "sha256": file_sha256(
                    Path(alignment_config["outputs"]["result"])
                ),
            },
            "protocol_sweep_result": {
                "path": protocol_config["outputs"]["result"],
                "sha256": file_sha256(
                    Path(protocol_config["outputs"]["result"])
                ),
            },
        },
    }
    return result


def _markdown(result: dict[str, object]) -> str:
    rolling = result["update_local_full_catalog_rolling_all"]
    rows = [
        f"# QK theta2 {result['candidate']} Summary",
        "",
        f"- Sweep eligible: `{result['selection_signals']['eligible_for_sweep_ranking']}`",
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
            "- Candidate selection remains deferred until all three candidates complete.",
        ]
    )
    return "\n".join(rows) + "\n"


def main() -> None:
    args = parse_args()
    result = summarize(args.config)
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
