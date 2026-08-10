from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.qk_stream_version import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    return parser.parse_args()


def _best_checked(rows: list[dict[str, object]]) -> dict[str, object] | None:
    eligible = [
        value
        for value in rows
        if value.get("ndcg_at_10_relative_percent") is not None
    ]
    return max(
        eligible,
        key=lambda value: float(value["ndcg_at_10_relative_percent"]),
        default=None,
    )


def _primary_gate(
    rolling_all: dict[str, object],
    relative_range: list[float],
    minimum_targets: int,
) -> dict[str, object]:
    ndcg = rolling_all["gaps"]["ndcg_at_10"]
    mrr = rolling_all["gaps"]["mrr"]
    relative = ndcg.get("relative_percent")
    target_passed = int(rolling_all["positive_targets"]) >= minimum_targets
    range_passed = (
        relative is not None
        and float(relative_range[0]) <= float(relative) <= float(relative_range[1])
    )
    interval_passed = (
        ndcg.get("positive_direction_with_ci") is True
        and mrr.get("positive_direction_with_ci") is True
    )
    return {
        "status": "pass" if target_passed and range_passed and interval_passed else "fail",
        "minimum_targets": minimum_targets,
        "positive_targets": rolling_all["positive_targets"],
        "ndcg_at_10_relative_percent_range": relative_range,
        "ndcg_at_10_relative_percent": relative,
        "ndcg_at_10_positive_ci": ndcg.get("positive_direction_with_ci"),
        "mrr_relative_percent": mrr.get("relative_percent"),
        "mrr_positive_ci": mrr.get("positive_direction_with_ci"),
        "target_count_passed": target_passed,
        "relative_range_passed": range_passed,
        "joint_interval_passed": interval_passed,
    }


def _row(
    name: str,
    origin: str,
    path: Path,
    relative_range: list[float],
    minimum_targets: int,
) -> dict[str, object]:
    result = json.loads(path.read_text())
    evaluation = result["quality"]["evaluation"]
    full = evaluation["summary"]["full_catalog"]
    rolling = full["rolling_next_item"]["cohorts"]["all"]
    boundary = full["boundary_multi_positive"]["cohorts"]["all"]
    gate = evaluation["primary_admission_gate"]
    return {
        "candidate": name,
        "origin": origin,
        "result": {"path": str(path), "sha256": file_sha256(path)},
        "rolling_all": {
            "positive_targets": rolling["positive_targets"],
            "reuse": rolling["reuse"],
            "recompute": rolling["recompute"],
            "gaps": rolling["gaps"],
        },
        "boundary_all": {
            "positive_targets": boundary["positive_targets"],
            "reuse": boundary["reuse"],
            "recompute": boundary["recompute"],
            "gaps": boundary["gaps"],
        },
        "rolling_all_gate": _primary_gate(
            rolling,
            relative_range,
            minimum_targets,
        ),
        "best_predeclared_relation": _best_checked(gate["all_checked"]),
        "relation_admission_status": gate["status"],
        "admitted_relations": gate["admitted"],
    }


def summarize(plan_path: Path) -> dict[str, object]:
    plan = json.loads(plan_path.read_text())
    quality = plan["relevance_quality"]
    relative_range = quality["preferred_relative_gap_percent_range"]
    minimum_targets = int(quality["minimum_cohort_targets"])
    anchor = plan["anchor_candidate"]
    rows = [
        _row(
            anchor["candidate_name"],
            "retained_n32_anchor",
            Path(anchor["relevance_result"]),
            relative_range,
            minimum_targets,
        )
    ]
    round_root = Path(plan["outputs"]["round_root"])
    for candidate in plan["search_training"]["candidates"]:
        name = candidate["candidate_name"]
        path = round_root / "candidates" / name / "relevance" / "result.json"
        if not path.is_file():
            raise FileNotFoundError(f"QK negative-strength result is absent: {path}")
        rows.append(
            _row(
                name,
                candidate["axis"],
                path,
                relative_range,
                minimum_targets,
            )
        )
    preferred = [
        value["candidate"]
        for value in rows
        if value["rolling_all_gate"]["status"] == "pass"
    ]
    return {
        "protocol": "evokv_qk_theta2_negative_strength_summary_v0",
        "status": (
            "preferred_candidate_found"
            if preferred
            else "complete_no_preferred_candidate"
        ),
        "round_id": plan["round_id"],
        "scientific_result": False,
        "formal_result": False,
        "plan": {"path": str(plan_path), "sha256": file_sha256(plan_path)},
        "anchor_candidate": anchor["candidate_name"],
        "candidate_count_including_anchor": len(rows),
        "candidates": rows,
        "preferred_candidates": preferred,
        "selection_deferred": True,
        "qualification_consumed": False,
        "final_consumed": False,
        "interpretation_boundary": (
            "select at most one theta2 only after reviewing the rolling full-catalog joint NDCG/MRR gate; record-cluster intervals remain development diagnostics until seed replication"
        ),
    }


def _markdown(summary: dict[str, object]) -> str:
    lines = [
        "# QK theta2 negative-strength search",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| Candidate | Rolling NDCG gap | NDCG CI+ | Rolling MRR gap | MRR CI+ | HR@10 gap | Boundary NDCG gap | Primary pass |",
        "|---|---:|---|---:|---|---:|---:|---|",
    ]
    for value in summary["candidates"]:
        rolling = value["rolling_all"]["gaps"]
        boundary = value["boundary_all"]["gaps"]
        lines.append(
            "| "
            + " | ".join(
                (
                    value["candidate"],
                    f"{rolling['ndcg_at_10']['relative_percent']:.4f}%",
                    str(rolling["ndcg_at_10"]["positive_direction_with_ci"]),
                    f"{rolling['mrr']['relative_percent']:.4f}%",
                    str(rolling["mrr"]["positive_direction_with_ci"]),
                    f"{rolling['hit_rate_at_10']['relative_percent']:.4f}%",
                    f"{boundary['ndcg_at_10']['relative_percent']:.4f}%",
                    str(value["rolling_all_gate"]["status"] == "pass"),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "The anchor and every search candidate use the same theta1 source, window2 training data, optimizer-participant users, and unseen window3 evaluation protocol.",
            "The primary development gate requires rolling full-catalog NDCG@10 in the frozen 5%-10% range and positive record-cluster intervals for both NDCG@10 and MRR.",
            "No checkpoint is committed automatically, and qualification/final roles are not consumed.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text())
    summary = summarize(args.plan)
    json_path = Path(plan["outputs"]["summary_json"])
    markdown_path = Path(plan["outputs"]["summary_markdown"])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payloads = (
        (json_path, json.dumps(summary, indent=2, sort_keys=True) + "\n"),
        (markdown_path, _markdown(summary)),
    )
    for path, payload in payloads:
        if path.is_file() and path.read_text() != payload:
            raise FileExistsError(f"QK negative-strength summary differs: {path}")
        path.write_text(payload)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "preferred_candidates": summary["preferred_candidates"],
                "summary": {"path": str(json_path), "sha256": file_sha256(json_path)},
                "summary_markdown": {
                    "path": str(markdown_path),
                    "sha256": file_sha256(markdown_path),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
