from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from hstu_kvcache.streaming.qk_stream_runner import _atomic_json
from hstu_kvcache.streaming.qk_stream_version import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    config_path = parse_args().config
    document = json.loads(config_path.read_text())
    result_path = Path(document["outputs"]["result"])
    result = json.loads(result_path.read_text())
    aggregate = result["aggregate"]
    sanity = aggregate["sanity"]
    expected_records = (
        document["data"]["record_limit_per_rank"] * document["execution"]["world_size"]
    )
    methods = document["interventions"]["methods"]
    endpoints = aggregate["endpoints"]
    if (
        result.get("protocol") != document["protocol"]
        or result.get("status") != "complete_development_measurement"
        or result.get("scientific_result") is not False
        or result.get("formal_result") is not False
        or result["config"]["sha256"] != file_sha256(config_path)
        or any(
            file_sha256(Path(value["path"])) != value["sha256"]
            for value in result["programs"].values()
        )
        or aggregate["records"] != expected_records
        or aggregate["positive_targets"] < expected_records
        or len(endpoints) != len(methods)
        or set(endpoints) != set(methods)
        or not sanity["implementation_passed"]
        or result["execution"]["qualification_consumed"] is not False
        or result["execution"]["final_consumed"] is not False
        or any(
            not math.isfinite(value)
            for endpoint in endpoints.values()
            for value in endpoint.values()
        )
    ):
        raise RuntimeError("QK root-cause sanity validation failed")
    validation = {
        "protocol": document["protocol"],
        "status": "validated",
        "round_id": document["round_id"],
        "config_sha256": file_sha256(config_path),
        "result_sha256": file_sha256(result_path),
        "records": aggregate["records"],
        "positive_targets": aggregate["positive_targets"],
        "implementation_passed": True,
        "qualification_consumed": False,
        "final_consumed": False,
    }
    _atomic_json(Path(document["outputs"]["validation"]), validation)
    comparisons = aggregate["fresh_reference_comparisons"]
    decision = {
        "protocol": "evokv_root_cause_campaign_decision_v0",
        "status": "continue",
        "completed_round": document["round_id"],
        "result_sha256": validation["result_sha256"],
        "reason": "same-version and canonical equivalence pass, and strong cache interventions change hidden states",
        "next_rounds": [
            "round_03_a_qk_attribution",
            "round_03_b_kuairand_natural_day",
        ],
        "diagnostic_snapshot": {
            method: {
                metric: comparisons[method][metric]
                for metric in ("cross_entropy", "ndcg_at_10", "mrr")
            }
            for method in (
                "stale_theta1",
                "zero_prefix",
                "no_prefix",
                "wrong_user_fresh",
                "shuffled_prefix",
                "recent_16",
                "recent_64",
            )
        },
        "scientific_result": False,
        "qualification_consumed": False,
        "final_consumed": False,
    }
    _atomic_json(Path(document["outputs"]["decision"]), decision)
    print(json.dumps({"validation": validation, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
