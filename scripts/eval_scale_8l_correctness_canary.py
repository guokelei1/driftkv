#!/usr/bin/env python3
"""S2 real-data correctness canary for the frozen 8L scale architecture."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

import scale_8l_common as scale
import train_p7_theta0 as p7
from hstu_kvcache.models import (
    append_with_rolling_cap,
    hybrid_tail_refresh,
    project_exact_layer0_segment,
    retain_latest_cache,
    transition_work,
)


def tensors(row, device: torch.device, *, start: int = 0, stop: int | None = None):
    stop = len(row.history_items) if stop is None else stop
    return (
        torch.tensor(row.history_items[start:stop][None, :], dtype=torch.long, device=device),
        torch.tensor(row.history_behaviors[start:stop][None, :], dtype=torch.long, device=device),
        torch.tensor(row.history_time_deltas[start:stop][None, :], dtype=torch.float32, device=device),
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=scale.OUTPUT / "s2_correctness_canary.json")
    args = parser.parse_args()
    contract = scale.contract()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("8L correctness canary must exercise a CUDA execution path")

    rows = scale.load_requests(scale.P7_MANIFEST, "residual_train", "F")
    eligible = [row for row in rows if len(row.history_items) >= 640]
    if not eligible:
        raise RuntimeError("scale loader did not recover any history beyond 512")
    row = scale.deterministic_subset(eligible, 1, "scale-canary")[0]
    model = scale.make_model(17, device).eval()
    bases, _ = p7.load_bases(("F",), device)
    base = bases["F"]

    full_inputs = tensors(row, device)
    full_cache = model.compute_kv(*full_inputs)
    split = min(512, len(row.history_items) - 1)
    prefix = model.compute_kv(*tensors(row, device, stop=split))
    _, appended = model.forward_with_cache(prefix, *tensors(row, device, start=split))
    append_delta = max(
        float((full_cache.k - appended.k).abs().max()),
        float((full_cache.v - appended.v).abs().max()),
    )

    cap = min(640, len(row.history_items))
    initial_width = cap - 2
    capped_start = len(row.history_items) - cap
    initial = model.compute_kv(*tensors(row, device, start=capped_start, stop=capped_start + initial_width))
    suffix = tensors(row, device, start=capped_start + initial_width)
    rolled = append_with_rolling_cap(model, initial, *suffix, max_length=initial_width)
    manual = initial
    for position in range(suffix[0].shape[1]):
        manual = retain_latest_cache(manual, initial_width - 1)
        _, manual = model.forward_with_cache(
            manual,
            suffix[0][:, position : position + 1],
            suffix[1][:, position : position + 1],
            suffix[2][:, position : position + 1],
        )
    rolling_delta = max(
        float((rolled.k - manual.k).abs().max()), float((rolled.v - manual.v).abs().max())
    )

    r0 = copy.deepcopy(model)
    with torch.no_grad():
        for name, parameter in r0.named_parameters():
            if name.startswith(("query_encoder.", "cc_score_head.")):
                parameter.add_(0.001)
    r0_cache = r0.compute_kv(*full_inputs)
    r0_delta = max(
        float((full_cache.k - r0_cache.k).abs().max()), float((full_cache.v - r0_cache.v).abs().max())
    )

    recent = project_exact_layer0_segment(model, full_cache, *full_inputs, "recent_128")
    hybrid = hybrid_tail_refresh(model, full_cache, *full_inputs, 128)
    exact = model.compute_kv(*full_inputs)
    exact_delta = max(float((full_cache.k - exact.k).abs().max()), float((full_cache.v - exact.v).abs().max()))
    work = {
        action: transition_work(action, full_cache, *full_inputs).__dict__
        for action in ("noop", "layer0_recent128", "layer0_middle", "layer0_full", "hybrid_tail128", "exact_all")
    }

    features = torch.tensor(row.base_features[None, :, :], dtype=torch.float32, device=device)
    base_full = base(features).float()
    base_recent = base(features).float()
    base_delta = float((base_full - base_recent).abs().max())

    candidate = torch.tensor(row.candidate_ids[None, :], dtype=torch.long, device=device)
    query_delta = torch.tensor([row.query_time_delta], dtype=torch.float32, device=device)
    query_type = torch.tensor([2], dtype=torch.long, device=device)
    before_k = full_cache.k.clone()
    before_v = full_cache.v.clone()
    score = model.score_cc_reuse(
        full_cache, candidate, query_delta,
        prefix_lengths=torch.tensor([full_cache.seq_len], device=device),
        query_type_ids=query_type,
    )
    query_mutation = max(
        float((before_k - full_cache.k).abs().max()), float((before_v - full_cache.v).abs().max())
    )

    # Batch-vs-serial exact cache check is the grouped-executor primitive.
    pair = scale.deterministic_subset([value for value in eligible if len(value.history_items) == len(row.history_items)], 2, "scale-batch")
    if len(pair) < 2:
        pair = [row, row]
    batch_inputs = tuple(torch.cat([tensors(value, device)[i] for value in pair], dim=0) for i in range(3))
    grouped = model.compute_kv(*batch_inputs)
    grouped_delta = 0.0
    for index, value in enumerate(pair):
        serial = model.compute_kv(*tensors(value, device))
        grouped_delta = max(
            grouped_delta,
            float((grouped.k[:, index : index + 1] - serial.k).abs().max()),
            float((grouped.v[:, index : index + 1] - serial.v).abs().max()),
        )

    tolerance = 5e-5
    checks = {
        "scale_history_exceeds_512": len(row.history_items) > 512,
        "scale_history_within_1024": len(row.history_items) <= 1024,
        "full_equals_prefix_append": append_delta <= tolerance,
        "rolling_helper_equals_manual_evict_before_append": rolling_delta == 0.0,
        "R0_cache_producer_identity": r0_delta == 0.0,
        "Exact_equals_CurrentFull": exact_delta == 0.0,
        "base_identical_across_history_paths": base_delta == 0.0,
        "candidate_query_does_not_mutate_state": query_mutation == 0.0,
        "grouped_equals_serial": grouped_delta <= tolerance,
        "partial_actions_shape_valid": recent.k.shape == hybrid.k.shape == full_cache.k.shape,
        "all_outputs_finite": bool(torch.isfinite(score).all()),
    }
    payload = {
        "status": "passed" if all(checks.values()) else "failed",
        "contract_sha256": scale.sha256_file(scale.CONTRACT),
        "device": args.device,
        "GPU_name": torch.cuda.get_device_name(device),
        "request_id": row.request_id,
        "history_length": len(row.history_items),
        "model": scale.model_metadata(model),
        "checks": checks,
        "max_abs_deltas": {
            "full_vs_prefix_append": append_delta,
            "rolling_vs_manual": rolling_delta,
            "R0_cache": r0_delta,
            "Exact_vs_CurrentFull": exact_delta,
            "base_path": base_delta,
            "query_state_mutation": query_mutation,
            "grouped_vs_serial": grouped_delta,
        },
        "work": work,
        "long_training_launched": False,
        "qualification_or_theta3_accessed": False,
        "GPU_allowlist": contract["execution"]["GPU_allowlist"],
    }
    scale.json_dump(args.output, payload)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "passed":
        raise SystemExit("8L correctness canary failed")


if __name__ == "__main__":
    main()
