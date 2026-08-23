#!/usr/bin/env python3
"""Write raw Full-1024/Recent-32 scores for the frozen 8L pilot H gate.

This is a development-only scale-reproduction gate.  It deliberately reads the
development split and cannot unlock P7 qualification or theta3 data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

import eval_p7_h_raw as p7_h
import scale_8l_common as scale
import train_p7_theta0 as p7
from hstu_kvcache.data.p7_training import load_p7_requests
from hstu_kvcache.models import HSTU, HSTUConfig

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/contracts/scale_8l_h_pilot_v1.yaml"
CHECKPOINT = ROOT / "results/scale_8l_v1/theta0/m0_f_seed17/theta0_selected.pt"
OUTPUT = ROOT / "results/scale_8l_v1/pilot/h_m0_f_seed17"
VIEWS = ("quality", "fidelity")
MICROBATCH = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract() -> dict[str, Any]:
    value = yaml.safe_load(CONTRACT.read_text())
    expected = {
        "scale_contract_sha256": scale.CONTRACT,
        "development_manifest_sha256": scale.P7_MANIFEST / "development/manifest.index.json",
        "frozen_base_bundle_sha256": scale.BASE_ROOT / "bundle_manifest.json",
        "checkpoint_sha256": CHECKPOINT,
        "raw_evaluator_sha256": Path(__file__),
    }
    for key, path in expected.items():
        actual = sha256_file(path)
        if value["sealed_inputs"][key] != actual:
            raise RuntimeError(f"pilot H contract hash mismatch: {key}")
    if value["data_access"]["split"] != "development":
        raise RuntimeError("pilot H contract may only read development")
    if value["data_access"]["qualification_or_theta3"] is not False:
        raise RuntimeError("pilot H contract illegally authorizes sealed data")
    return value


def common_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("request_id", pa.string(), nullable=False),
            pa.field("uid", pa.int64(), nullable=False),
            pa.field("candidate_id", pa.int64(), nullable=False),
            pa.field("base_logit", pa.float32(), nullable=False),
            pa.field("recent32_residual_logit", pa.float32(), nullable=False),
            pa.field("full1024_residual_logit", pa.float32(), nullable=False),
            pa.field("recent32_deployment_logit", pa.float32(), nullable=False),
            pa.field("full1024_deployment_logit", pa.float32(), nullable=False),
            pa.field("query_timestamp", pa.int64(), nullable=False),
            pa.field("history_length", pa.int32(), nullable=False),
            pa.field("request_weight", pa.float64(), nullable=False),
            pa.field("manifest_view", pa.string(), nullable=False),
            pa.field("model_condition", pa.string(), nullable=False),
            pa.field("seed", pa.int32(), nullable=False),
        ]
    )


def quality_schema() -> pa.Schema:
    return pa.schema(
        list(common_schema())
        + [
            pa.field("label", pa.int8(), nullable=False),
            pa.field("is_organic", pa.int8()),
            pa.field("prior_30m_same_item", pa.bool_()),
            pa.field("latest_item", pa.bool_()),
            pa.field("long_gap_at_least_3d", pa.bool_()),
        ]
    )


def load_model(device: torch.device) -> tuple[HSTU, dict[str, Any]]:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    required = {
        "contract": "scale_8l_v1",
        "model_name": "m0_f",
        "seed": 17,
        "history_limit": 1024,
        "qualification_or_theta3_scored": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"checkpoint identity mismatch: {key}")
    if payload["contract_hash"] != sha256_file(scale.CONTRACT):
        raise RuntimeError("checkpoint scale contract hash differs")
    model = HSTU(HSTUConfig(**payload["config"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    del payload["model_state_dict"]
    return model.to(device).eval(), payload


def load_requests(view: str):
    # No QualificationUnlock can be passed on this path by construction.
    return load_p7_requests(
        scale.P7_MANIFEST,
        scale.RAW,
        "development",
        "F",
        manifest_kind=view,
        history_limit=1024,
    )


@torch.no_grad()
def score_batch(model: HSTU, base, requests, device: torch.device):
    full_tensors = p7_h.collate(requests, device, history_tokens=1024)
    recent_tensors = p7_h.collate(requests, device, history_tokens=32)
    full = p7_h.score_path(model, full_tensors, device, workload="F", chunk_size=1)
    recent = p7_h.score_path(model, recent_tensors, device, workload="F", chunk_size=1)
    base_full = base(full_tensors["features"].float()).float()
    base_recent = base(recent_tensors["features"].float()).float()
    base_delta = float((base_full - base_recent).abs().max())
    if base_delta != 0.0:
        raise AssertionError("Frozen Base differs between Full-1024 and Recent-32")
    return base_full, recent, full, base_delta


def columns(requests, base, recent, full, view: str) -> dict[str, list[Any]]:
    schema = quality_schema() if view == "quality" else common_schema()
    output: dict[str, list[Any]] = {name: [] for name in schema.names}
    for index, row in enumerate(requests):
        if len(row.candidate_ids) != 1:
            raise RuntimeError("F pilot expects exactly one candidate per request")
        base_logit = float(base[index, 0])
        recent_logit = float(recent[index, 0])
        full_logit = float(full[index, 0])
        values: dict[str, Any] = {
            "request_id": row.request_id,
            "uid": row.uid,
            "candidate_id": int(row.candidate_ids[0]),
            "base_logit": base_logit,
            "recent32_residual_logit": recent_logit,
            "full1024_residual_logit": full_logit,
            "recent32_deployment_logit": base_logit + recent_logit,
            "full1024_deployment_logit": base_logit + full_logit,
            "query_timestamp": row.query_timestamp,
            "history_length": len(row.history_items),
            "request_weight": row.request_weight,
            "manifest_view": view,
            "model_condition": "m0_f",
            "seed": 17,
        }
        if view == "quality":
            if row.label is None:
                raise RuntimeError("quality row has no feedback label")
            values.update(
                {
                    "label": int(row.label),
                    "is_organic": row.is_organic,
                    "prior_30m_same_item": row.prior_30m_same_item,
                    "latest_item": row.latest_item,
                    "long_gap_at_least_3d": row.query_time_delta >= 3 * 86_400,
                }
            )
        for name in schema.names:
            output[name].append(values[name])
    return output


@torch.no_grad()
def run_canary(device: torch.device) -> None:
    contract = load_contract()
    output = OUTPUT / "development_identity_canary.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    model, _ = load_model(device)
    bases, _ = p7.load_bases(("F",), device)
    requests = scale.deterministic_subset(load_requests("fidelity"), 64, "scale-H-pilot-canary")
    base, recent, full, base_delta = score_batch(model, bases["F"], requests, device)
    _, recent_repeat, full_repeat, _ = score_batch(model, bases["F"], requests, device)
    repeat_delta = max(
        float((recent - recent_repeat).abs().max()),
        float((full - full_repeat).abs().max()),
    )
    values = {
        "status": "passed_development_only_scale_H_identity_canary",
        "requests": len(requests),
        "qualification_or_theta3_read": False,
        "base_full_recent_max_abs_delta": base_delta,
        "repeat_score_max_abs_delta": repeat_delta,
        "all_scores_finite": bool(torch.isfinite(torch.cat((base.flatten(), recent.flatten(), full.flatten()))).all()),
        "contract_sha256": sha256_file(CONTRACT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "numeric_floors": contract["gates"]["numeric_floors"],
    }
    if base_delta != 0.0 or repeat_delta != 0.0 or not values["all_scores_finite"]:
        raise RuntimeError(f"scale H identity canary failed: {values}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(values, indent=2) + "\n")
    print(json.dumps(values, indent=2))


@torch.no_grad()
def run_raw(device: torch.device) -> None:
    load_contract()
    summary_path = OUTPUT / "raw_run_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite {summary_path}")
    canary = OUTPUT / "development_identity_canary.json"
    if not canary.is_file():
        raise RuntimeError("identity canary must pass before raw scoring")
    model, _ = load_model(device)
    bases, _ = p7.load_bases(("F",), device)
    outputs = []
    for view in VIEWS:
        requests = load_requests(view)
        output_path = OUTPUT / f"F_{view}_raw.parquet"
        temporary = output_path.with_suffix(".tmp.parquet")
        if output_path.exists() or temporary.exists():
            raise FileExistsError(f"refusing to overwrite {output_path}")
        schema = quality_schema() if view == "quality" else common_schema()
        writer = pq.ParquetWriter(temporary, schema, compression="zstd")
        base_delta = 0.0
        try:
            for start in range(0, len(requests), MICROBATCH):
                batch = requests[start : start + MICROBATCH]
                base, recent, full, delta = score_batch(model, bases["F"], batch, device)
                base_delta = max(base_delta, delta)
                writer.write_table(
                    pa.Table.from_pydict(
                        columns(batch, base.cpu().numpy(), recent.cpu().numpy(), full.cpu().numpy(), view),
                        schema=schema,
                    )
                )
        finally:
            writer.close()
        os.replace(temporary, output_path)
        names = set(pq.read_schema(output_path).names)
        forbidden = {"label", "is_organic", "prior_30m_same_item", "latest_item", "long_gap_at_least_3d"}
        if view == "fidelity" and names & forbidden:
            raise RuntimeError("target-free fidelity artifact contains quality fields")
        outputs.append(
            {
                "view": view,
                "path": str(output_path.relative_to(ROOT)),
                "sha256": sha256_file(output_path),
                "requests": len(requests),
                "users": len({row.uid for row in requests}),
                "base_full_recent_max_abs_delta": base_delta,
                "schema": sorted(names),
            }
        )
    summary = {
        "status": "scale_H_raw_scores_written_before_metrics",
        "evidence_level": "development_pilot_only",
        "model_condition": "m0_f",
        "seed": 17,
        "comparison": "Base+Full1024_vs_Base+Recent32",
        "qualification_or_theta3_read": False,
        "metrics_computed": False,
        "contract_sha256": sha256_file(CONTRACT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "outputs": outputs,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "outputs": outputs}, indent=2))


def run_seal() -> None:
    load_contract()
    output = OUTPUT / "raw_score_seal.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    summary_path = OUTPUT / "raw_run_summary.json"
    summary = json.loads(summary_path.read_text())
    if summary["metrics_computed"] is not False:
        raise RuntimeError("raw summary claims metrics were already computed")
    artifacts = []
    for row in summary["outputs"]:
        path = ROOT / row["path"]
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"raw artifact hash differs: {path}")
        artifacts.append(row)
    value = {
        "status": "sealed_scale_H_raw_scores_before_metrics",
        "evidence_level": "development_pilot_only",
        "qualification_or_theta3_read": False,
        "contract_sha256": sha256_file(CONTRACT),
        "raw_summary_sha256": sha256_file(summary_path),
        "metrics_computed": False,
        "artifacts": artifacts,
    }
    output.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"status": value["status"], "sha256": sha256_file(output)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("canary", "raw", "seal"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.mode == "seal":
        run_seal()
    elif args.mode == "canary":
        run_canary(torch.device(args.device))
    else:
        run_raw(torch.device(args.device))


if __name__ == "__main__":
    main()
