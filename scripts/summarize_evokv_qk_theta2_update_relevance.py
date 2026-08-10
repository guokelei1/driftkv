from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.qk_stream_version import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--phase", choices=("existing", "complete"), required=True)
    return parser.parse_args()


def _candidate_names(plan: dict[str, object], phase: str) -> list[str]:
    names = [value["candidate_name"] for value in plan["existing_candidates"]]
    if phase == "complete":
        names.extend(
            value["candidate_name"]
            for value in plan["fallback_training"]["candidates"]
        )
    return names


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


def summarize(plan_path: Path, phase: str) -> dict[str, object]:
    plan = json.loads(plan_path.read_text())
    round_root = Path(plan["outputs"]["round_root"])
    candidates = []
    admitted = []
    for name in _candidate_names(plan, phase):
        path = round_root / "candidates" / name / "relevance" / "result.json"
        if not path.is_file():
            raise FileNotFoundError(f"QK update relevance result is absent: {path}")
        result = json.loads(path.read_text())
        evaluation = result["quality"]["evaluation"]
        gate = evaluation["primary_admission_gate"]
        full = evaluation["summary"]["full_catalog"]["rolling_next_item"][
            "cohorts"
        ]
        row = {
            "candidate": name,
            "origin": result["config"]["path"],
            "result": {"path": str(path), "sha256": file_sha256(path)},
            "all_window3": {
                "positive_targets": full["all"]["positive_targets"],
                "recompute_ndcg_at_10": full["all"]["recompute"]["ndcg_at_10"],
                "ndcg_at_10_relative_percent": full["all"]["gaps"]["ndcg_at_10"][
                    "relative_percent"
                ],
                "mrr_relative_percent": full["all"]["gaps"]["mrr"][
                    "relative_percent"
                ],
            },
            "best_predeclared_relation": _best_checked(gate["all_checked"]),
            "primary_admission_status": gate["status"],
            "admitted_relations": gate["admitted"],
        }
        candidates.append(row)
        if gate["admitted"]:
            admitted.append(row)
    if admitted:
        status = "preferred_candidate_found"
    elif phase == "existing":
        status = "no_existing_admission_continue_fallback"
    else:
        status = "complete_no_preferred_candidate"
    return {
        "protocol": "evokv_qk_theta2_update_relevance_summary_v0",
        "status": status,
        "phase": phase,
        "round_id": plan["round_id"],
        "scientific_result": False,
        "formal_result": False,
        "plan": {"path": str(plan_path), "sha256": file_sha256(plan_path)},
        "candidate_count": len(candidates),
        "candidates": candidates,
        "admitted_candidates": [value["candidate"] for value in admitted],
        "selection_deferred": True,
        "qualification_consumed": False,
        "final_consumed": False,
        "interpretation_boundary": (
            "user selects at most one theta2 after reviewing full-catalog relation evidence and supporting candidate protocols"
        ),
    }


def _markdown(summary: dict[str, object]) -> str:
    lines = [
        "# QK theta2 update-relevance round",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| Candidate | All NDCG gap | All MRR gap | Best predeclared relation | Relation NDCG gap | Admitted |",
        "|---|---:|---:|---|---:|---|",
    ]
    for value in summary["candidates"]:
        best = value["best_predeclared_relation"]
        if best is None:
            relation = "-"
            relation_gap = "-"
        else:
            relation = f"{best['mode']}/{best['cohort']}"
            relation_gap = f"{best['ndcg_at_10_relative_percent']:.4f}%"
        lines.append(
            "| "
            + " | ".join(
                (
                    value["candidate"],
                    f"{value['all_window3']['ndcg_at_10_relative_percent']:.4f}%",
                    f"{value['all_window3']['mrr_relative_percent']:.4f}%",
                    relation,
                    relation_gap,
                    str(bool(value["admitted_relations"])),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "All cohorts were frozen from window2 identities, labels, and ordinals before reading any window3 quality value.",
            "Candidate-set protocols are supporting diagnostics; only the full-catalog relation gate can stop fallback training.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = summarize(args.plan, args.phase)
    plan = json.loads(args.plan.read_text())
    round_root = Path(plan["outputs"]["round_root"])
    if args.phase == "existing":
        json_path = round_root / "existing_summary.json"
        markdown_path = round_root / "existing_summary.md"
    else:
        json_path = Path(plan["outputs"]["summary_json"])
        markdown_path = Path(plan["outputs"]["summary_markdown"])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    markdown = _markdown(summary)
    for path, value in ((json_path, payload), (markdown_path, markdown)):
        if path.is_file() and path.read_text() != value:
            raise FileExistsError(f"QK update relevance summary differs: {path}")
        path.write_text(value)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "admitted_candidates": summary["admitted_candidates"],
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
