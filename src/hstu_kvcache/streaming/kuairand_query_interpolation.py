from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from .kuairand_query_transition import (
    _atomic_json,
    _evaluate,
    _summary,
    build_workload,
    file_sha256,
    load_config,
    make_model,
)

PROTOCOL = "evokv_kuairand_query_publish_interpolation_v0"


def load_interpolation_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    parent = document.get("parent")
    selection = document.get("selection")
    allowed_alphas = (
        [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        [0.65, 0.7, 0.725, 0.75, 0.775],
        [0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        [0.75, 0.755, 0.76, 0.765, 0.77, 0.775, 0.78, 0.785, 0.79, 0.795, 0.8],
    )
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or not isinstance(parent, dict)
        or not isinstance(selection, dict)
        or document.get("alphas") not in allowed_alphas
        or selection.get("metrics") != ["mrr", "ndcg_at_10", "hit_rate_at_10"]
        or float(selection.get("minimum_relative_percent", 0)) != 5.0
        or float(selection.get("maximum_relative_percent", 0)) != 15.0
        or int(selection.get("minimum_metrics", 0)) != 2
        or not 0.0 < float(parent.get("theta1_source_alpha", 1.0)) <= 1.0
    ):
        raise ValueError("KuaiRand interpolation config differs")
    for field in ("base_config", "theta0", "theta1"):
        artifact = parent.get(field)
        if not isinstance(artifact, dict):
            raise ValueError("KuaiRand interpolation parent is absent")
        artifact_path = Path(artifact.get("path", ""))
        if not artifact_path.is_file() or file_sha256(artifact_path) != artifact.get("sha256"):
            raise ValueError(f"KuaiRand interpolation parent differs: {field}")
    return document


def _interpolate(
    current,
    previous_state: dict[str, torch.Tensor],
    raw_state: dict[str, torch.Tensor],
    alpha: float,
    source_alpha: float = 1.0,
) -> None:
    state = current.state_dict()
    with torch.no_grad():
        for name, target in state.items():
            previous = previous_state[name].to(target.device)
            raw = raw_state[name].to(target.device)
            target.copy_(previous + (raw - previous) * (alpha / source_alpha))


def run_interpolation(config_path: str | Path) -> dict[str, Any]:
    document = load_interpolation_config(config_path)
    output_root = Path(document["outputs"]["root"])
    result_path = output_root / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        validate_interpolation_result(result, document)
        return result
    base_config = load_config(document["parent"]["base_config"]["path"])
    workload = build_workload(base_config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("KuaiRand interpolation requires CUDA")
    previous_payload = torch.load(
        document["parent"]["theta0"]["path"], map_location="cpu", weights_only=True
    )
    raw_payload = torch.load(
        document["parent"]["theta1"]["path"], map_location="cpu", weights_only=True
    )
    previous_state = previous_payload["state_dict"]
    raw_state = raw_payload["state_dict"]
    source_alpha = float(document["parent"].get("theta1_source_alpha", 1.0))
    previous = make_model(
        base_config, int(workload["metadata"]["embedding_rows"]), device
    )
    current = make_model(
        base_config, int(workload["metadata"]["embedding_rows"]), device
    )
    previous.load_state_dict(previous_state)
    previous.eval()
    cells = []
    started = time.monotonic()
    for alpha in document["alphas"]:
        _interpolate(current, previous_state, raw_state, float(alpha), source_alpha)
        current.eval()
        evaluation = _evaluate(previous, current, workload, base_config)
        compact = _summary(evaluation, base_config)
        cell_path = output_root / "cells" / f"alpha_{float(alpha):.3f}.json"
        _atomic_json(
            cell_path,
            {
                "alpha": alpha,
                "summary": compact,
                "records": evaluation["records"],
            },
        )
        cells.append(
            {
                "alpha": alpha,
                "path": str(cell_path),
                "sha256": file_sha256(cell_path),
                "summary": compact,
            }
        )
        stale = compact["comparisons"]["recompute_over_reuse"]
        print(
            f"phase=kuairand_query_interpolation alpha={alpha:.2f} "
            f"mrr={stale['mrr']['relative_percent']:.3f}% "
            f"ndcg10={stale['ndcg_at_10']['relative_percent']:.3f}% "
            f"hr10={stale['hit_rate_at_10']['relative_percent']:.3f}% "
            f"gate={compact['gate']['passed']}",
            flush=True,
        )
    selection = document["selection"]
    admitted = []
    for cell in cells:
        stale = cell["summary"]["comparisons"]["recompute_over_reuse"]
        passing = [
            metric
            for metric in selection["metrics"]
            if stale[metric]["positive_direction_with_ci"]
            and float(selection["minimum_relative_percent"])
            <= stale[metric]["relative_percent"]
            <= float(selection["maximum_relative_percent"])
        ]
        if cell["summary"]["gate"]["passed"] and len(passing) >= int(
            selection["minimum_metrics"]
        ):
            admitted.append({"alpha": cell["alpha"], "passing_metrics": passing})
    selected = admitted[0] if admitted else None
    result = {
        "protocol": PROTOCOL,
        "round_id": document["round_id"],
        "status": "complete",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "parent": document["parent"],
        "workload": workload["metadata"],
        "cells": cells,
        "decision": {
            "admitted": admitted,
            "selected": selected,
            "next": "locked_independent_replication" if selected else "train_update_strength_sweep",
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    validate_interpolation_result(result, document)
    _atomic_json(result_path, result)
    return result


def validate_interpolation_result(result: dict[str, Any], document: dict[str, Any]) -> None:
    cells = result.get("cells")
    if (
        result.get("protocol") != PROTOCOL
        or result.get("round_id") != document["round_id"]
        or result.get("status") != "complete"
        or result.get("scientific_result") is not False
        or result.get("formal_result") is not False
        or not isinstance(cells, list)
        or [value.get("alpha") for value in cells] != document["alphas"]
    ):
        raise ValueError("KuaiRand interpolation result differs")
    for cell in cells:
        path = Path(cell["path"])
        if not path.is_file() or file_sha256(path) != cell["sha256"]:
            raise ValueError("KuaiRand interpolation cell binding differs")
        if not cell.get("summary", {}).get("sanity", {}).get("passed"):
            raise ValueError("KuaiRand interpolation sanity failed")
