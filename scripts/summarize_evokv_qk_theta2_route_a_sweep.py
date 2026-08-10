from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.qk_stream_version import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    return parser.parse_args()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def summarize(plan_path: Path) -> dict[str, object]:
    plan = json.loads(plan_path.read_text())
    result_parent = Path(plan["outputs"]["result_parent"])
    rows = []
    for candidate in plan["candidates"]:
        summary_path = result_parent / candidate["round_id"] / "summary.json"
        summary = json.loads(summary_path.read_text())
        if (
            summary.get("protocol")
            != "evokv_qk_theta2_route_a_candidate_summary_v0"
            or summary.get("candidate") != candidate["candidate_name"]
        ):
            raise ValueError("QK theta2 sweep candidate summary differs")
        core = summary["update_local_full_catalog_rolling_all"]
        fit = summary["fit_tuning_full_catalog"]
        admitted = summary["alignment_gate"]["admitted"]
        preferred = (
            max(admitted, key=lambda value: value["ndcg_at_10_relative_percent"])
            if admitted
            else None
        )
        row = {
            "candidate": candidate["candidate_name"],
            "epochs": candidate["epochs"],
            "dense_learning_rate": candidate["dense_learning_rate"],
            "embedding_learning_rate": candidate[
                "embedding_learning_rate"
            ],
            "total_targets": summary["training"]["total_targets"],
            "final_epoch_mean_loss": summary["training"][
                "epoch_mean_losses"
            ][-1],
            "core_recompute": {
                "cross_entropy": core["recompute"]["cross_entropy"],
                "ndcg_at_10": core["recompute"]["ndcg_at_10"],
                "mrr": core["recompute"]["mrr"],
                "hit_rate_at_10": core["recompute"]["hit_rate_at_10"],
            },
            "core_recompute_minus_reuse_relative_percent": {
                metric: core["gaps"][metric]["relative_percent"]
                for metric in (
                    "cross_entropy",
                    "ndcg_at_10",
                    "mrr",
                    "hit_rate_at_10",
                )
            },
            "core_ci_positive": {
                metric: core["gaps"][metric][
                    "positive_direction_with_ci"
                ]
                for metric in ("cross_entropy", "ndcg_at_10", "mrr")
            },
            "fit_recompute": {
                "cross_entropy": fit["recompute"]["cross_entropy"],
                "ndcg_at_10": fit["recompute"]["ndcg_at_10"],
                "mrr": fit["recompute"]["mrr"],
            },
            "preferred_alignment": preferred,
            "eligible_for_sweep_ranking": summary["selection_signals"][
                "eligible_for_sweep_ranking"
            ],
            "summary": {
                "path": str(summary_path),
                "sha256": file_sha256(summary_path),
            },
            "checkpoint": summary["checkpoint"],
        }
        rows.append(row)
    eligible = [row for row in rows if row["eligible_for_sweep_ranking"]]
    ranking = sorted(
        eligible,
        key=lambda row: (
            -row["core_recompute"]["ndcg_at_10"],
            -row["core_recompute"]["mrr"],
            row["epochs"],
            row["embedding_learning_rate"],
        ),
    )
    return {
        "protocol": "evokv_qk_theta2_route_a_sweep_summary_v0",
        "status": "complete_development_measurement",
        "scientific_result": False,
        "formal_result": False,
        "route": "A",
        "source_version": 1,
        "target_version": 2,
        "training_window": 2,
        "evaluation_window": 3,
        "plan": {"path": str(plan_path), "sha256": file_sha256(plan_path)},
        "candidate_count": len(rows),
        "candidates": rows,
        "eligible_candidates": [row["candidate"] for row in eligible],
        "provisional_quality_ranking": [
            row["candidate"] for row in ranking
        ],
        "ranking_semantics": [
            "require core NDCG@10 and MRR positive record-cluster CI",
            "require a predeclared aligned cohort in the 5%-10% range",
            "rank by core Recompute NDCG@10 then MRR then lower training strength",
        ],
        "selection_deferred": True,
        "automatic_checkpoint_retirement": False,
        "qualification_consumed": False,
        "final_consumed": False,
        "next_boundary": "user reviews all candidates before one theta2 is frozen or theta3 is implemented",
    }


def _markdown(result: dict[str, object]) -> str:
    rows = [
        "# QK theta2 Route A Parameter Sweep",
        "",
        "| Candidate | Epochs | Emb LR | Recompute NDCG@10 | NDCG gap % | MRR gap % | Preferred cohort | Eligible |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for value in result["candidates"]:
        preferred = value["preferred_alignment"]
        cohort = "-" if preferred is None else (
            f"{preferred['mode']}/{preferred['cohort']} "
            f"({preferred['ndcg_at_10_relative_percent']:.3f}%)"
        )
        gaps = value["core_recompute_minus_reuse_relative_percent"]
        rows.append(
            f"| {value['candidate']} | {value['epochs']} | "
            f"{value['embedding_learning_rate']:.6g} | "
            f"{value['core_recompute']['ndcg_at_10']:.8g} | "
            f"{gaps['ndcg_at_10']:.4g} | {gaps['mrr']:.4g} | "
            f"{cohort} | {value['eligible_for_sweep_ranking']} |"
        )
    rows.extend(
        [
            "",
            f"- Eligible candidates: `{result['eligible_candidates']}`",
            f"- Provisional quality ranking: `{result['provisional_quality_ranking']}`",
            "- Selection and checkpoint retirement remain deferred for user review.",
            "- Qualification/final consumed: `False/False`.",
        ]
    )
    return "\n".join(rows) + "\n"


def main() -> None:
    args = parse_args()
    result = summarize(args.plan)
    plan = json.loads(args.plan.read_text())
    summary_json = Path(plan["outputs"]["summary_json"])
    summary_markdown = Path(plan["outputs"]["summary_markdown"])
    _atomic_text(
        summary_json, json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    _atomic_text(summary_markdown, _markdown(result))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
