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
    load_plan,
    make_model,
)
from .kuairand_intraday_update import _split_day

MULTIVERSION_PROTOCOL = "evokv_kuairand_multiversion_staleness_v0"


def load_multiversion_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    data = document.get("data_config", {})
    source = document.get("source_round_config", {})
    chain = document.get("chain_round_config", {})
    if (
        document.get("protocol") != MULTIVERSION_PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("current_version") != 7
        or document.get("source_versions") != [6, 5, 4, 3, 2]
        or document.get("evaluation_date_index") != 20
        or document.get("horizons") != [4, 8, 16]
        or file_sha256(data.get("path", "")) != data.get("sha256")
        or file_sha256(source.get("path", "")) != source.get("sha256")
        or file_sha256(chain.get("path", "")) != chain.get("sha256")
    ):
        raise ValueError("KuaiRand multiversion-staleness config differs")
    return document


def _prepare_plan(document, evaluation_date_index, split_fraction):
    plan, metadata = load_plan(document)
    plan.init_base()
    dates = plan.base_dates + plan.stream_dates
    for date_index in range(14, evaluation_date_index):
        _, update, evaluation = _split_day(
            plan.daily_segments[dates[date_index]], split_fraction
        )
        for frame in (update, evaluation):
            for user, group in frame.groupby("user_idx"):
                plan._append_day_to_history(int(user), group)
    boundary, update, evaluation = _split_day(
        plan.daily_segments[dates[evaluation_date_index]], split_fraction
    )
    update_key = "multiversion_update"
    evaluation_key = "multiversion_evaluation"
    plan.daily_segments[update_key] = update
    plan.daily_segments[evaluation_key] = evaluation
    plan.ingest_day(update_key)
    return plan, metadata, dates[evaluation_date_index], boundary, update_key, evaluation_key


def _source_root(version, source_root, chain_root):
    return source_root if version == 2 else chain_root


def run_multiversion_staleness(config_path: str | Path):
    config = load_multiversion_config(config_path)
    output = Path(config["output"])
    if output.is_file():
        return json.loads(output.read_text())
    document = load_engagement_config(config["data_config"]["path"])
    source_document = json.loads(Path(config["source_round_config"]["path"]).read_text())
    chain_document = json.loads(Path(config["chain_round_config"]["path"]).read_text())
    plan, metadata, source_date, boundary, update_key, evaluation_key = _prepare_plan(
        document,
        int(config["evaluation_date_index"]),
        float(config["split_fraction"]),
    )
    device = torch.device("cuda:0")
    source_root = Path(source_document["checkpoint_root"])
    chain_root = Path(chain_document["checkpoint_root"])
    current = make_model(document, plan, device)
    _load_checkpoint(current, chain_root, int(config["current_version"]))
    current.eval()
    lags = []
    started = time.monotonic()
    for source_version in config["source_versions"]:
        source = make_model(document, plan, device)
        _load_checkpoint(
            source,
            _source_root(int(source_version), source_root, chain_root),
            int(source_version),
        )
        source.eval()
        values = []
        for horizon in config["horizons"]:
            print(
                f"phase=multiversion source={source_version} "
                f"current={config['current_version']} horizon={horizon}",
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
        lags.append(
            {
                "source_version": source_version,
                "lag": int(config["current_version"]) - int(source_version),
                "values": values,
            }
        )
    result = {
        "protocol": MULTIVERSION_PROTOCOL,
        "source_protocol": PROTOCOL,
        "status": "complete_development_diagnostic",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "data_config": config["data_config"],
        "source_round_config": config["source_round_config"],
        "chain_round_config": config["chain_round_config"],
        "data": metadata,
        "source_date": source_date,
        "boundary_time_ms": boundary,
        "current_version": config["current_version"],
        "lags": lags,
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(output, result)
    return result


def validate_multiversion_staleness(result: dict[str, Any]) -> None:
    if (
        result.get("protocol") != MULTIVERSION_PROTOCOL
        or result.get("status") != "complete_development_diagnostic"
        or result.get("scientific_result") is not False
        or [lag.get("source_version") for lag in result.get("lags", [])] != [6, 5, 4, 3, 2]
        or not all(len(lag.get("values", [])) == 3 for lag in result["lags"])
        or not all(
            value.get("same_model_sanity_passed")
            for lag in result["lags"]
            for value in lag["values"]
        )
    ):
        raise ValueError("KuaiRand multiversion-staleness result differs")
