from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from hstu_kvcache.streaming.qk_stream_version import file_sha256

EXPECTED_CANDIDATES = {
    "theta2_route_a_e3_lr100",
    "theta2_route_a_e4_lr100",
    "theta2_route_a_e3_lr150",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def validate(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text())
    summary_path = Path(config["outputs"]["summary_json"])
    summary = json.loads(summary_path.read_text())
    candidate = config["edge"]["candidate_name"]
    expected_targets = 82_854 * int(config["training"]["epochs"])
    if (
        candidate not in EXPECTED_CANDIDATES
        or config["edge"].get("source_version") != 1
        or config["edge"].get("target_version") != 2
        or config["edge"].get("edge") != 2
        or summary.get("protocol")
        != "evokv_qk_theta2_route_a_candidate_summary_v0"
        or summary.get("status") != "complete_development_measurement"
        or summary.get("scientific_result") is not False
        or summary.get("formal_result") is not False
        or summary.get("candidate") != candidate
        or summary.get("training", {}).get("total_targets")
        != expected_targets
        or summary.get("qualification_consumed") is not False
        or summary.get("final_consumed") is not False
        or summary.get("selection_deferred") is not True
    ):
        raise ValueError("QK theta2 candidate summary differs")
    for family in (
        "fit_tuning_full_catalog",
        "update_local_full_catalog_rolling_all",
    ):
        value = summary[family]
        if value.get("positive_targets", 0) < 1:
            raise ValueError("QK theta2 quality target count differs")
        for method in ("reuse", "recompute"):
            metrics = value[method]
            if any(
                not math.isfinite(float(metrics[name]))
                for name in (
                    "cross_entropy",
                    "ndcg_at_10",
                    "mrr",
                    "hit_rate_at_10",
                )
            ):
                raise ValueError("QK theta2 quality metric differs")
        for metric in ("cross_entropy", "ndcg_at_10", "mrr"):
            interval = value["gaps"][metric][
                "record_cluster_bootstrap_95"
            ]
            if any(
                not math.isfinite(float(interval[key]))
                for key in ("lower", "upper")
            ):
                raise ValueError("QK theta2 quality interval differs")
    if (
        summary["alignment_gate"].get("status")
        not in {"aligned_protocol_found", "no_aligned_protocol"}
        or summary["candidate_protocol_gate"].get("status")
        not in {"admitted_protocol_found", "no_admitted_protocol"}
    ):
        raise ValueError("QK theta2 candidate gate differs")
    checkpoint = summary["checkpoint"]
    manifest = Path(checkpoint["path"]) / "manifest.json"
    if file_sha256(manifest) != checkpoint["manifest_sha256"]:
        raise ValueError("QK theta2 candidate checkpoint differs")
    for artifact in summary["artifacts"].values():
        if file_sha256(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError("QK theta2 candidate artifact differs")
    return {
        "status": "pass",
        "candidate": candidate,
        "summary": {
            "path": str(summary_path),
            "sha256": file_sha256(summary_path),
        },
        "checkpoint_manifest_sha256": file_sha256(manifest),
        "eligible_for_sweep_ranking": summary["selection_signals"][
            "eligible_for_sweep_ranking"
        ],
    }


def main() -> None:
    print(
        json.dumps(
            validate(parse_args().config), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
