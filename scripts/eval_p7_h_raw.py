#!/usr/bin/env python3
"""Write frozen P7.8 Full-512/Recent-32 raw scores before metric reveal."""

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

from hstu_kvcache.data.compact_manifest import QualificationUnlock
from hstu_kvcache.data.p7_training import QUERY_TYPES, P7Request, load_p7_requests
from hstu_kvcache.models import HSTU, FrozenLinearBaseRanker, HSTUConfig

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = ROOT / "data/manifests/p7_full_v1"
RAW_LISTENS = ROOT / "data/raw/yambda/flat/50m/listens.parquet"
BASE_ROOT = ROOT / "results/p7/base_fit/frozen_base_bundle_v1"
THETA_ROOT = ROOT / "results/p7/theta0_training/runs"
OUTPUT_ROOT = ROOT / "results/p7/h_qualification/raw"
CONTRACT = ROOT / "configs/contracts/p7_8_h_qualification_contract_v1.yaml"
RUN_PLAN = ROOT / "configs/contracts/p7_8_qualification_run_plan_v1.json"

MODELS = {
    "m0_n": ("N",),
    "m0_r": ("R",),
    "m0_f": ("F",),
    "m1": ("N", "R", "F"),
}
VIEWS = {
    "N": ("quality", "fidelity"),
    "R": ("quality_rankable", "fidelity_all_eligible", "fidelity_rankable_companion"),
    "F": ("quality", "fidelity"),
}
MICROBATCH = {"N": 4, "R": 1, "F": 8}
CHUNK_SIZE = {"N": 25, "R": 16, "F": 1}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float64) - float(np.max(values))
    output = np.exp(shifted)
    return output / output.sum()


def bernoulli_js(first_logit: float, second_logit: float) -> float:
    first = 1.0 / (1.0 + math.exp(-float(first_logit)))
    second = 1.0 / (1.0 + math.exp(-float(second_logit)))
    p = np.asarray([first, 1.0 - first], dtype=np.float64)
    q = np.asarray([second, 1.0 - second], dtype=np.float64)
    middle = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / middle)) + 0.5 * np.sum(q * np.log(q / middle)))


def categorical_js(first: np.ndarray, second: np.ndarray) -> float:
    p = softmax(first)
    q = softmax(second)
    middle = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / middle)) + 0.5 * np.sum(q * np.log(q / middle)))


def identity_metrics(first: np.ndarray, second: np.ndarray, workload: str) -> dict[str, float]:
    delta = second.astype(np.float64) - first.astype(np.float64)
    if workload == "F":
        first_probability = 1.0 / (1.0 + math.exp(-float(first[0])))
        second_probability = 1.0 / (1.0 + math.exp(-float(second[0])))
        return {
            "output_js_divergence": bernoulli_js(float(first[0]), float(second[0])),
            "normalized_score_rms": float(np.sqrt(np.mean(delta**2)) / max(float(np.std(first)), 1e-3)),
            "absolute_probability_difference": abs(first_probability - second_probability),
        }
    centered_delta = delta - delta.mean()
    centered_full = first.astype(np.float64) - float(np.mean(first))
    signs_first = np.sign(first[:, None] - first[None, :])
    signs_second = np.sign(second[:, None] - second[None, :])
    upper = np.triu_indices(len(first), 1)
    inversion = float(np.mean(signs_first[upper] != signs_second[upper])) if len(first) > 1 else 0.0
    top = min(10, len(first))
    first_top = set(np.argsort(-first, kind="stable")[:top].tolist())
    second_top = set(np.argsort(-second, kind="stable")[:top].tolist())
    return {
        "output_js_divergence": categorical_js(first, second),
        "normalized_score_rms": float(
            np.sqrt(np.mean(centered_delta**2))
            / (np.sqrt(np.mean(centered_full**2)) + 1e-6)
        ),
        "pairwise_inversion": inversion,
        "top10_overlap_loss": 1.0 - len(first_top & second_top) / top,
    }


def collate(
    requests: list[P7Request],
    device: torch.device,
    *,
    history_tokens: int,
) -> dict[str, torch.Tensor]:
    batch = len(requests)
    history_width = max(min(len(row.history_items), history_tokens) for row in requests)
    candidate_width = max(len(row.candidate_ids) for row in requests)
    items = np.zeros((batch, history_width), dtype=np.int64)
    behaviors = np.zeros_like(items)
    deltas = np.zeros((batch, history_width), dtype=np.float32)
    lengths = np.zeros(batch, dtype=np.int64)
    candidates = np.zeros((batch, candidate_width), dtype=np.int64)
    candidate_mask = np.zeros((batch, candidate_width), dtype=bool)
    features = np.zeros((batch, candidate_width, 7), dtype=np.float32)
    query_deltas = np.zeros(batch, dtype=np.float32)
    for index, row in enumerate(requests):
        length = min(len(row.history_items), history_tokens)
        candidate_count = len(row.candidate_ids)
        items[index, :length] = row.history_items[-length:]
        behaviors[index, :length] = row.history_behaviors[-length:]
        selected_deltas = row.history_time_deltas[-length:].copy()
        selected_deltas[0] = 0.0
        deltas[index, :length] = selected_deltas
        lengths[index] = length
        candidates[index, :candidate_count] = row.candidate_ids
        candidate_mask[index, :candidate_count] = True
        features[index, :candidate_count] = row.base_features
        query_deltas[index] = row.query_time_delta
    return {
        "items": torch.from_numpy(items).to(device),
        "behaviors": torch.from_numpy(behaviors).to(device),
        "deltas": torch.from_numpy(deltas).to(device),
        "lengths": torch.from_numpy(lengths).to(device),
        "candidates": torch.from_numpy(candidates).to(device),
        "candidate_mask": torch.from_numpy(candidate_mask).to(device),
        "features": torch.from_numpy(features).to(device),
        "query_deltas": torch.from_numpy(query_deltas).to(device),
        "query_types": torch.full(
            (batch,), QUERY_TYPES[requests[0].workload], dtype=torch.long, device=device
        ),
    }


@torch.no_grad()
def score_path(
    model: HSTU,
    tensors: dict[str, torch.Tensor],
    device: torch.device,
    *,
    workload: str,
    chunk_size: int,
) -> torch.Tensor:
    # Qualification uses FP32 so the frozen exact-chunking contract is not
    # confounded by BF16 matrix-shape rounding across candidate chunk sizes.
    chunks = model.score_cc_full_chunked(
        tensors["items"],
        tensors["behaviors"],
        tensors["deltas"],
        tensors["candidates"],
        tensors["query_deltas"],
        chunk_size=chunk_size,
        lengths=tensors["lengths"],
        query_type_ids=tensors["query_types"],
    )
    scores = torch.cat(chunks, dim=1).float()
    if not torch.isfinite(scores[tensors["candidate_mask"]]).all():
        raise FloatingPointError(f"{workload} residual contains non-finite scores")
    return scores


def load_model(model_name: str, seed: int, device: torch.device) -> tuple[HSTU, str]:
    checkpoint = THETA_ROOT / f"{model_name}_seed{seed}/theta0_selected.pt"
    checkpoint_hash = sha256_file(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["model_name"] != model_name or int(payload["seed"]) != seed:
        raise RuntimeError("checkpoint identity differs")
    if payload["qualification_scored"] is not False:
        raise RuntimeError("checkpoint claims qualification access")
    model = HSTU(HSTUConfig(**payload["config"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    del payload
    return model.to(device).eval(), checkpoint_hash


def load_base(workload: str, device: torch.device) -> FrozenLinearBaseRanker:
    path = BASE_ROOT / f"base_{workload.lower()}_v1.json"
    artifact = json.loads(path.read_text())
    scorer = FrozenLinearBaseRanker.from_frozen_artifact(artifact).to(device).eval()
    if list(scorer.parameters()):
        raise RuntimeError("Frozen Base exposes parameters")
    return scorer


def quality_schema() -> pa.Schema:
    fields = list(common_schema())
    fields.extend(
        [
            pa.field("target_index", pa.int32()),
            pa.field("label", pa.int8()),
            pa.field("is_target", pa.bool_()),
            pa.field("is_organic", pa.int8()),
            pa.field("prior_30m_same_item", pa.bool_()),
            pa.field("latest_item", pa.bool_()),
            pa.field("long_gap_at_least_3d", pa.bool_()),
            pa.field("history_position_cohort", pa.string()),
        ]
    )
    return pa.schema(fields)


def common_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("request_id", pa.string(), nullable=False),
            pa.field("uid", pa.int64(), nullable=False),
            pa.field("candidate_id", pa.int64(), nullable=False),
            pa.field("candidate_position", pa.int32(), nullable=False),
            pa.field("base_logit", pa.float32(), nullable=False),
            pa.field("recent32_residual_logit", pa.float32(), nullable=False),
            pa.field("full512_residual_logit", pa.float32(), nullable=False),
            pa.field("recent32_deployment_logit", pa.float32(), nullable=False),
            pa.field("full512_deployment_logit", pa.float32(), nullable=False),
            pa.field("model_condition", pa.string(), nullable=False),
            pa.field("seed", pa.int32(), nullable=False),
            pa.field("workload", pa.string(), nullable=False),
            pa.field("manifest_view", pa.string(), nullable=False),
            pa.field("query_timestamp", pa.int64(), nullable=False),
            pa.field("history_length", pa.int32(), nullable=False),
            pa.field("request_weight", pa.float64(), nullable=False),
        ]
    )


def history_position(row: P7Request) -> str:
    candidate = int(row.candidate_ids[0])
    positions = np.flatnonzero(row.history_items == candidate)
    if len(positions) and positions[-1] >= len(row.history_items) - 32:
        return "recent_seen"
    if len(positions):
        return "old_seen"
    # The frozen item-history-missing base feature distinguishes catalog items
    # unseen in all causal history from items seen only before the retained 512.
    item_history_missing = float(row.base_features[0, 5]) > 0.5
    return "unseen" if item_history_missing else "seen_only_before_512"


def raw_columns(
    requests: list[P7Request],
    base: np.ndarray,
    recent: np.ndarray,
    full: np.ndarray,
    *,
    model_name: str,
    seed: int,
    workload: str,
    view: str,
) -> dict[str, list[Any]]:
    output: dict[str, list[Any]] = {name: [] for name in common_schema().names}
    is_quality = "quality" in view
    if is_quality:
        output.update({name: [] for name in quality_schema().names if name not in output})
    for request_index, row in enumerate(requests):
        count = len(row.candidate_ids)
        position_cohort = history_position(row) if workload == "F" else None
        for candidate_position in range(count):
            base_logit = float(base[request_index, candidate_position])
            recent_logit = float(recent[request_index, candidate_position])
            full_logit = float(full[request_index, candidate_position])
            values = {
                "request_id": row.request_id,
                "uid": row.uid,
                "candidate_id": int(row.candidate_ids[candidate_position]),
                "candidate_position": candidate_position,
                "base_logit": base_logit,
                "recent32_residual_logit": recent_logit,
                "full512_residual_logit": full_logit,
                "recent32_deployment_logit": base_logit + recent_logit,
                "full512_deployment_logit": base_logit + full_logit,
                "model_condition": model_name,
                "seed": seed,
                "workload": workload,
                "manifest_view": view,
                "query_timestamp": row.query_timestamp,
                "history_length": len(row.history_items),
                "request_weight": row.request_weight,
            }
            if is_quality:
                values.update(
                    {
                        "target_index": row.target_index,
                        "label": row.label,
                        "is_target": None
                        if row.target_index is None
                        else candidate_position == row.target_index,
                        "is_organic": row.is_organic,
                        "prior_30m_same_item": row.prior_30m_same_item,
                        "latest_item": row.latest_item,
                        "long_gap_at_least_3d": row.query_time_delta >= 3 * 86_400,
                        "history_position_cohort": position_cohort,
                    }
                )
            for name, value in values.items():
                output[name].append(value)
    return output


@torch.no_grad()
def score_view(
    model: HSTU,
    base_model: FrozenLinearBaseRanker,
    requests: list[P7Request],
    device: torch.device,
    *,
    model_name: str,
    seed: int,
    workload: str,
    view: str,
    output_path: Path,
) -> dict[str, Any]:
    schema = quality_schema() if "quality" in view else common_schema()
    temporary = output_path.with_suffix(".tmp.parquet")
    if output_path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite raw artifact: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(temporary, schema, compression="zstd")
    request_count = candidate_rows = 0
    max_base_path_delta = 0.0
    try:
        microbatch = MICROBATCH[workload]
        for start in range(0, len(requests), microbatch):
            micro = requests[start : start + microbatch]
            full_tensors = collate(micro, device, history_tokens=512)
            recent_tensors = collate(micro, device, history_tokens=32)
            full = score_path(
                model,
                full_tensors,
                device,
                workload=workload,
                chunk_size=CHUNK_SIZE[workload],
            )
            recent = score_path(
                model,
                recent_tensors,
                device,
                workload=workload,
                chunk_size=CHUNK_SIZE[workload],
            )
            base_full = base_model(full_tensors["features"].float()).float()
            base_recent = base_model(recent_tensors["features"].float()).float()
            max_base_path_delta = max(
                max_base_path_delta,
                float((base_full - base_recent).abs().max()),
            )
            if max_base_path_delta != 0.0:
                raise AssertionError("Frozen Base differs between Full and Recent")
            columns = raw_columns(
                micro,
                base_full.cpu().numpy(),
                recent.cpu().numpy(),
                full.cpu().numpy(),
                model_name=model_name,
                seed=seed,
                workload=workload,
                view=view,
            )
            writer.write_table(pa.Table.from_pydict(columns, schema=schema))
            request_count += len(micro)
            candidate_rows += sum(len(row.candidate_ids) for row in micro)
    finally:
        writer.close()
    os.replace(temporary, output_path)
    return {
        "path": str(output_path.relative_to(ROOT)),
        "sha256": sha256_file(output_path),
        "requests": request_count,
        "candidate_rows": candidate_rows,
        "base_full_recent_max_abs_delta": max_base_path_delta,
        "fidelity_schema_has_forbidden_fields": bool(
            "fidelity" in view
            and set(quality_schema().names).difference(common_schema().names)
            & set(pq.read_schema(output_path).names)
        ),
    }


def validate_plan(model_name: str, seed: int) -> tuple[dict[str, Any], QualificationUnlock]:
    plan = json.loads(RUN_PLAN.read_text())
    if plan["status"] != "sealed_before_qualification_unlock":
        raise RuntimeError("qualification run plan is not sealed")
    if plan["evaluation_contract_sha256"] != sha256_file(CONTRACT):
        raise RuntimeError("evaluation contract hash differs")
    if plan["raw_evaluator_sha256"] != sha256_file(Path(__file__)):
        raise RuntimeError("raw evaluator changed after run-plan sealing")
    if plan["frozen_base_bundle_sha256"] != sha256_file(BASE_ROOT / "bundle_manifest.json"):
        raise RuntimeError("Frozen Base changed after run-plan sealing")
    checkpoint_key = f"{model_name}_seed{seed}"
    checkpoint_path = ROOT / plan["checkpoints"][checkpoint_key]["path"]
    if sha256_file(checkpoint_path) != plan["checkpoints"][checkpoint_key]["sha256"]:
        raise RuntimeError("checkpoint changed after run-plan sealing")
    unlock = QualificationUnlock(
        qualification_index_hash=plan["qualification_index_sha256"],
        frozen_base_hashes=(plan["frozen_base_bundle_sha256"],),
        frozen_checkpoint_hashes=tuple(
            row["sha256"] for row in plan["checkpoints"].values()
        ),
        checkpoint_selection_complete=True,
    )
    return plan, unlock


def run_raw(model_name: str, seed: int, device: torch.device) -> None:
    plan, unlock = validate_plan(model_name, seed)
    model, checkpoint_hash = load_model(model_name, seed, device)
    expected = plan["checkpoints"][f"{model_name}_seed{seed}"]["sha256"]
    if checkpoint_hash != expected:
        raise RuntimeError("loaded checkpoint hash differs")
    outputs = []
    for workload in MODELS[model_name]:
        base = load_base(workload, device)
        for view in VIEWS[workload]:
            requests = load_p7_requests(
                MANIFEST_ROOT,
                RAW_LISTENS,
                "qualification",
                workload,
                manifest_kind=view,
                qualification_unlock=unlock,
            )
            output = OUTPUT_ROOT / f"{model_name}_seed{seed}" / f"{workload}_{view}.parquet"
            outputs.append(
                {
                    "workload": workload,
                    "view": view,
                    **score_view(
                        model,
                        base,
                        requests,
                        device,
                        model_name=model_name,
                        seed=seed,
                        workload=workload,
                        view=view,
                        output_path=output,
                    ),
                }
            )
            del requests
    summary = {
        "status": "raw_scores_written_before_metrics",
        "model_condition": model_name,
        "seed": seed,
        "checkpoint_sha256": checkpoint_hash,
        "evaluation_contract_sha256": sha256_file(CONTRACT),
        "run_plan_sha256": sha256_file(RUN_PLAN),
        "qualification_index_sha256": plan["qualification_index_sha256"],
        "metrics_computed": False,
        "outputs": outputs,
    }
    summary_path = OUTPUT_ROOT / f"{model_name}_seed{seed}" / "raw_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "model": model_name, "seed": seed, "files": len(outputs)}, indent=2))


@torch.no_grad()
def run_canary(device: torch.device, output: Path) -> None:
    rows = []
    metric_maxima: dict[str, float] = {}
    for model_name, workload in (("m0_n", "N"), ("m0_r", "R"), ("m0_f", "F")):
        model, checkpoint_hash = load_model(model_name, 17, device)
        requests = load_p7_requests(
            MANIFEST_ROOT,
            RAW_LISTENS,
            "development",
            workload,
        )[:4]
        tensors = collate(requests, device, history_tokens=512)
        first = score_path(
            model, tensors, device, workload=workload, chunk_size=CHUNK_SIZE[workload]
        )
        second = score_path(
            model, tensors, device, workload=workload, chunk_size=CHUNK_SIZE[workload]
        )
        alternate = score_path(
            model,
            tensors,
            device,
            workload=workload,
            chunk_size=max(1, min(tensors["candidates"].shape[1], CHUNK_SIZE[workload] + 1)),
        )
        mask = tensors["candidate_mask"]
        repeat_delta = float((first[mask] - second[mask]).abs().max())
        chunk_delta = float((first[mask] - alternate[mask]).abs().max())
        metric_rows = []
        first_numpy = first.cpu().numpy()
        alternate_numpy = alternate.cpu().numpy()
        for request_index, request in enumerate(requests):
            count = len(request.candidate_ids)
            metrics = identity_metrics(
                first_numpy[request_index, :count],
                alternate_numpy[request_index, :count],
                workload,
            )
            metric_rows.append(metrics)
            for name, value in metrics.items():
                metric_maxima[name] = max(metric_maxima.get(name, 0.0), value)
        rows.append(
            {
                "model": model_name,
                "workload": workload,
                "checkpoint_sha256": checkpoint_hash,
                "requests": len(requests),
                "repeat_score_max_abs_delta": repeat_delta,
                "alternate_chunk_score_max_abs_delta": chunk_delta,
                "alternate_chunk_metric_maxima": {
                    name: max(metrics[name] for metrics in metric_rows)
                    for name in metric_rows[0]
                },
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    maximum = max(max(row["repeat_score_max_abs_delta"], row["alternate_chunk_score_max_abs_delta"]) for row in rows)
    floors = {
        name: max(1e-8 if name == "output_js_divergence" else 1e-7, 10.0 * value)
        for name, value in metric_maxima.items()
    }
    payload = {
        "status": "passed_development_identity_canary_before_qualification_unlock",
        "qualification_read": False,
        "rows": rows,
        "maximum_score_abs_delta": maximum,
        "metric_identity_maxima": metric_maxima,
        "frozen_numeric_floors": floors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite canary: {output}")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("canary", "raw"), required=True)
    parser.add_argument("--model", choices=tuple(MODELS))
    parser.add_argument("--seed", type=int, choices=(17, 37, 71))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--canary-output",
        type=Path,
        default=ROOT / "results/p7/h_qualification/development_identity_canary_v1.json",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.mode == "canary":
        if args.model is not None or args.seed is not None:
            raise ValueError("canary mode does not accept model or seed")
        run_canary(device, args.canary_output)
    else:
        if args.model is None or args.seed is None:
            raise ValueError("raw mode requires model and seed")
        run_raw(args.model, args.seed, device)


if __name__ == "__main__":
    main()
