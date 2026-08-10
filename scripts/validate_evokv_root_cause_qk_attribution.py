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


def _positive_ci(value: dict[str, object], endpoint: str) -> bool:
    return bool(value.get(f"{endpoint}_advantage_positive_with_ci"))


def main() -> None:
    config_path = parse_args().config
    document = json.loads(config_path.read_text())
    result_path = Path(document["outputs"]["result"])
    result = json.loads(result_path.read_text())
    aggregate = result["aggregate"]
    methods = document["interventions"]["methods"]
    expected_records = (
        document["data"]["record_limit_per_rank"] * document["execution"]["world_size"]
    )
    if (
        result.get("protocol") != document["protocol"]
        or result.get("status") != "complete_development_measurement"
        or result.get("scope") != document["scope"]
        or result.get("scientific_result") is not False
        or result.get("formal_result") is not False
        or result["config"]["sha256"] != file_sha256(config_path)
        or any(
            file_sha256(Path(value["path"])) != value["sha256"]
            for value in result["programs"].values()
        )
        or aggregate["records"] != expected_records
        or aggregate["positive_targets"] < expected_records
        or set(aggregate["endpoints"]) != set(methods)
        or not aggregate["sanity"]["implementation_passed"]
        or any(
            not math.isfinite(value)
            for endpoint in aggregate["endpoints"].values()
            for value in endpoint.values()
        )
        or result["execution"]["qualification_consumed"] is not False
        or result["execution"]["final_consumed"] is not False
    ):
        raise RuntimeError("QK root-cause attribution validation failed")
    validation = {
        "protocol": document["protocol"],
        "status": "validated",
        "round_id": document["round_id"],
        "scope": document["scope"],
        "config_sha256": file_sha256(config_path),
        "result_sha256": file_sha256(result_path),
        "records": aggregate["records"],
        "positive_targets": aggregate["positive_targets"],
        "implementation_passed": True,
        "qualification_consumed": False,
        "final_consumed": False,
    }
    _atomic_json(Path(document["outputs"]["validation"]), validation)
    if document["scope"] == "implementation_canary":
        decision = {
            "protocol": "evokv_root_cause_campaign_decision_v0",
            "status": "scale_to_development_attribution",
            "completed_round": document["round_id"],
            "result_sha256": validation["result_sha256"],
            "next_record_limit_per_rank": 512,
            "reason": "recursive lineage, cross-model fresh quality, history controls, and coarse parameter hybrids completed with finite outputs",
            "scientific_result": False,
            "qualification_consumed": False,
            "final_consumed": False,
        }
    else:
        comparisons = aggregate["fresh_theta2_comparisons"]
        update = aggregate["theta1_to_theta2_fresh_update_value"]
        history = {
            name: {
                metric: _positive_ci(comparisons[name][metric], "fresh_theta2")
                for metric in ("cross_entropy", "ndcg_at_10", "mrr")
            }
            for name in ("no_prefix", "wrong_user_fresh", "recent_16", "recent_64")
        }
        stale = {
            metric: _positive_ci(comparisons["stale_theta1"][metric], "fresh_theta2")
            for metric in ("cross_entropy", "ndcg_at_10", "mrr")
        }
        update_value = {
            metric: _positive_ci(update[metric], "theta2_fresh")
            for metric in ("cross_entropy", "ndcg_at_10", "mrr")
        }
        decision = {
            "protocol": "evokv_root_cause_campaign_decision_v0",
            "status": "complete_qk_attribution",
            "completed_round": document["round_id"],
            "result_sha256": validation["result_sha256"],
            "history_value_positive_ci": history,
            "theta1_to_theta2_update_value_positive_ci": update_value,
            "adjacent_stale_tax_positive_ci": stale,
            "next_rounds": ["round_03_b_kuairand_natural_day", "round_03_c_cross_diagnosis"],
            "scientific_result": False,
            "qualification_consumed": False,
            "final_consumed": False,
        }
    _atomic_json(Path(document["outputs"]["decision"]), decision)
    print(json.dumps({"validation": validation, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
