from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from .kuairand_engagement import (
    PROTOCOL,
    _atomic_json,
    _evaluate_edge,
    _load_checkpoint,
    file_sha256,
    load_engagement_config,
    make_model,
)
from .kuairand_multiversion_staleness import _prepare_plan

DEEP_STALENESS_PROTOCOL = "evokv_kuairand_deep_staleness_replication_v0"


def load_deep_staleness_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    data = document.get("data_config", {})
    replications = document.get("replications", [])
    bindings = [data]
    for replication in replications:
        bindings.extend(
            [replication.get("base_round_config", {}), replication.get("chain_round_config", {})]
        )
    if (
        document.get("protocol") != DEEP_STALENESS_PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or [value.get("training_seed") for value in replications] != [61031, 14929]
        or document.get("current_version") != 7
        or document.get("source_version") != 0
        or document.get("evaluation_date_index") != 20
        or document.get("horizons") != [4, 8]
        or document.get("split_fraction") != 0.5
        or any(file_sha256(value.get("path", "")) != value.get("sha256") for value in bindings)
    ):
        raise ValueError("KuaiRand deep-staleness config differs")
    return document


def _seed_pass(values: list[dict[str, Any]]) -> bool:
    return all(
        value["comparisons"]["recompute_over_reuse"]["average_precision"][
            "positive_direction_with_ci"
        ]
        and value["comparisons"]["recompute_over_reuse"]["ndcg_at_10"][
            "positive_direction_with_ci"
        ]
        and value["comparisons"]["recompute_over_reuse"]["ndcg_at_50"][
            "positive_direction_with_ci"
        ]
        and value["comparisons"]["recompute_over_reuse"]["average_precision"][
            "relative_percent"
        ]
        >= 5.0
        for value in values
    )


def run_deep_staleness(config_path: str | Path) -> dict[str, Any]:
    config = load_deep_staleness_config(config_path)
    output = Path(config["output"])
    if output.is_file():
        return json.loads(output.read_text())
    document = load_engagement_config(config["data_config"]["path"])
    device = torch.device("cuda:0")
    replications = []
    started = time.monotonic()
    for replication in config["replications"]:
        plan, metadata, source_date, boundary, update_key, evaluation_key = _prepare_plan(
            document,
            int(config["evaluation_date_index"]),
            float(config["split_fraction"]),
        )
        base_document = json.loads(Path(replication["base_round_config"]["path"]).read_text())
        chain_document = json.loads(Path(replication["chain_round_config"]["path"]).read_text())
        source = make_model(document, plan, device)
        current = make_model(document, plan, device)
        _load_checkpoint(
            source,
            Path(base_document["outputs"]["checkpoint_root"]),
            int(config["source_version"]),
        )
        _load_checkpoint(
            current,
            Path(chain_document["checkpoint_root"]),
            int(config["current_version"]),
        )
        source.eval()
        current.eval()
        values = []
        for horizon in config["horizons"]:
            print(
                f"phase=deep_staleness seed={replication['training_seed']} horizon={horizon}",
                flush=True,
            )
            values.append(
                _evaluate_edge(
                    document,
                    plan,
                    source,
                    current,
                    update_key,
                    evaluation_key,
                    int(config["current_version"]),
                    device,
                    max_exposures=int(horizon),
                )
            )
        replications.append(
            {
                "training_seed": replication["training_seed"],
                "source_version": config["source_version"],
                "current_version": config["current_version"],
                "lag": int(config["current_version"]) - int(config["source_version"]),
                "data": metadata,
                "source_date": source_date,
                "boundary_time_ms": boundary,
                "values": values,
                "passed": _seed_pass(values),
            }
        )
    result = {
        "protocol": DEEP_STALENESS_PROTOCOL,
        "source_protocol": PROTOCOL,
        "status": "complete_development_replication",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "replications": replications,
        "decision": {
            "all_training_seeds_passed": all(value["passed"] for value in replications),
            "passed_training_seeds": [
                value["training_seed"] for value in replications if value["passed"]
            ],
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output, result)
    return result


def validate_deep_staleness(result: dict[str, Any]) -> None:
    if (
        result.get("protocol") != DEEP_STALENESS_PROTOCOL
        or result.get("status") != "complete_development_replication"
        or result.get("scientific_result") is not False
        or [value.get("training_seed") for value in result.get("replications", [])]
        != [61031, 14929]
        or not all(value.get("lag") == 7 for value in result["replications"])
        or not all(len(value.get("values", [])) == 2 for value in result["replications"])
        or not all(
            metric.get("same_model_sanity_passed")
            for value in result["replications"]
            for metric in value["values"]
        )
    ):
        raise ValueError("KuaiRand deep-staleness result differs")
