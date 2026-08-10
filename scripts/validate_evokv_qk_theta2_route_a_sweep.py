from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.qk_stream_version import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    return parser.parse_args()


def validate(plan_path: Path) -> dict[str, object]:
    plan = json.loads(plan_path.read_text())
    summary_path = Path(plan["outputs"]["summary_json"])
    summary = json.loads(summary_path.read_text())
    expected = [
        candidate["candidate_name"] for candidate in plan["candidates"]
    ]
    observed = [value["candidate"] for value in summary.get("candidates", [])]
    if (
        summary.get("protocol")
        != "evokv_qk_theta2_route_a_sweep_summary_v0"
        or summary.get("status") != "complete_development_measurement"
        or summary.get("scientific_result") is not False
        or summary.get("formal_result") is not False
        or summary.get("plan", {}).get("sha256") != file_sha256(plan_path)
        or summary.get("candidate_count") != len(expected)
        or observed != expected
        or summary.get("selection_deferred") is not True
        or summary.get("automatic_checkpoint_retirement") is not False
        or summary.get("qualification_consumed") is not False
        or summary.get("final_consumed") is not False
    ):
        raise ValueError("QK theta2 sweep summary differs")
    eligible = set(summary["eligible_candidates"])
    ranking = summary["provisional_quality_ranking"]
    if (
        not eligible.issubset(expected)
        or set(ranking) != eligible
        or len(ranking) != len(set(ranking))
    ):
        raise ValueError("QK theta2 sweep ranking differs")
    for value in summary["candidates"]:
        candidate_summary = Path(value["summary"]["path"])
        manifest = Path(value["checkpoint"]["path"]) / "manifest.json"
        if (
            file_sha256(candidate_summary) != value["summary"]["sha256"]
            or file_sha256(manifest)
            != value["checkpoint"]["manifest_sha256"]
        ):
            raise ValueError("QK theta2 sweep artifact differs")
    markdown = Path(plan["outputs"]["summary_markdown"])
    if not markdown.is_file() or markdown.stat().st_size < 1:
        raise ValueError("QK theta2 sweep markdown is absent")
    return {
        "status": "pass",
        "summary": {
            "path": str(summary_path),
            "sha256": file_sha256(summary_path),
        },
        "summary_markdown": {
            "path": str(markdown),
            "sha256": file_sha256(markdown),
        },
        "eligible_candidates": summary["eligible_candidates"],
        "provisional_quality_ranking": ranking,
    }


def main() -> None:
    print(
        json.dumps(validate(parse_args().plan), indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
