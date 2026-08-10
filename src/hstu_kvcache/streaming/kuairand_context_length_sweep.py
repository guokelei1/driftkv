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

CONTEXT_SWEEP_PROTOCOL = "evokv_kuairand_context_length_sweep_v0"


def load_context_sweep_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    data = document.get("data_config", {})
    replications = document.get("replications", [])
    bindings = [data]
    for replication in replications:
        bindings.extend(
            [
                replication.get("source_round_config", {}),
                replication.get("chain_round_config", {}),
            ]
        )
    if (
        document.get("protocol") != CONTEXT_SWEEP_PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or [value.get("training_seed") for value in replications] != [61031, 14929]
        or document.get("current_version") != 7
        or document.get("source_version") != 2
        or document.get("evaluation_date_index") != 20
        or document.get("horizons") != [4, 8]
        or document.get("prefix_caps") != [128, 256, 384]
        or document.get("plan_max_original_seq_len") != 512
        or document.get("split_fraction") != 0.5
        or any(file_sha256(value.get("path", "")) != value.get("sha256") for value in bindings)
    ):
        raise ValueError("KuaiRand context-length sweep config differs")
    return document


def run_context_length_sweep(config_path: str | Path) -> dict[str, Any]:
    config = load_context_sweep_config(config_path)
    output = Path(config["output"])
    if output.is_file():
        return json.loads(output.read_text())
    document = load_engagement_config(config["data_config"]["path"])
    document = json.loads(json.dumps(document))
    document["data"]["max_original_seq_len"] = config["plan_max_original_seq_len"]
    device = torch.device("cuda:0")
    replications = []
    started = time.monotonic()
    for replication in config["replications"]:
        plan, metadata, source_date, boundary, update_key, evaluation_key = _prepare_plan(
            document,
            int(config["evaluation_date_index"]),
            float(config["split_fraction"]),
        )
        source_document = json.loads(
            Path(replication["source_round_config"]["path"]).read_text()
        )
        chain_document = json.loads(Path(replication["chain_round_config"]["path"]).read_text())
        source = make_model(document, plan, device)
        current = make_model(document, plan, device)
        _load_checkpoint(
            source,
            Path(source_document["checkpoint_root"]),
            int(config["source_version"]),
        )
        _load_checkpoint(
            current,
            Path(chain_document["checkpoint_root"]),
            int(config["current_version"]),
        )
        source.eval()
        current.eval()
        cells = []
        for prefix_cap in config["prefix_caps"]:
            for horizon in config["horizons"]:
                print(
                    f"phase=context_length seed={replication['training_seed']} "
                    f"prefix={prefix_cap} horizon={horizon}",
                    flush=True,
                )
                value = _evaluate_edge(
                    document,
                    plan,
                    source,
                    current,
                    update_key,
                    evaluation_key,
                    int(config["current_version"]),
                    device,
                    prefix_cap=int(prefix_cap),
                    max_exposures=int(horizon),
                )
                value["requested_prefix_cap"] = prefix_cap
                cells.append(value)
        replications.append(
            {
                "training_seed": replication["training_seed"],
                "source_version": config["source_version"],
                "current_version": config["current_version"],
                "lag": int(config["current_version"]) - int(config["source_version"]),
                "data": metadata,
                "source_date": source_date,
                "boundary_time_ms": boundary,
                "cells": cells,
            }
        )
    candidates = []
    for prefix_cap in config["prefix_caps"]:
        seed_values = []
        for replication in replications:
            values = [
                value
                for value in replication["cells"]
                if value["requested_prefix_cap"] == prefix_cap
            ]
            passed = all(
                value["comparisons"]["recompute_over_reuse"]["average_precision"][
                    "positive_direction_with_ci"
                ]
                and value["comparisons"]["recompute_over_reuse"]["ndcg_at_10"][
                    "positive_direction_with_ci"
                ]
                and value["comparisons"]["recompute_over_reuse"]["average_precision"][
                    "relative_percent"
                ]
                >= 5.0
                for value in values
            )
            seed_values.append({"training_seed": replication["training_seed"], "passed": passed})
        candidates.append(
            {
                "prefix_cap": prefix_cap,
                "seed_values": seed_values,
                "all_training_seeds_passed": all(value["passed"] for value in seed_values),
            }
        )
    result = {
        "protocol": CONTEXT_SWEEP_PROTOCOL,
        "source_protocol": PROTOCOL,
        "status": "complete_development_diagnostic",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "replications": replications,
        "decision": {
            "stable_prefix_caps": [
                value["prefix_cap"] for value in candidates if value["all_training_seeds_passed"]
            ],
            "candidates": candidates,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output, result)
    return result


def validate_context_length_sweep(result: dict[str, Any]) -> None:
    if (
        result.get("protocol") != CONTEXT_SWEEP_PROTOCOL
        or result.get("status") != "complete_development_diagnostic"
        or result.get("scientific_result") is not False
        or [value.get("training_seed") for value in result.get("replications", [])]
        != [61031, 14929]
        or not all(len(value.get("cells", [])) == 6 for value in result["replications"])
        or not all(
            cell.get("same_model_sanity_passed")
            for value in result["replications"]
            for cell in value["cells"]
        )
    ):
        raise ValueError("KuaiRand context-length sweep result differs")
