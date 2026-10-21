#!/usr/bin/env python3
"""One-user Yambda checkpoint/data/GPU forward smoke; no training or quality claim."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/processed/yambda500m_unified_v1/scales/medium/dataset.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results/unified_training_2026_09/medium/seed17/checkpoints/v2/checkpoint_100.pt")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the installed PyTorch wheel and NVIDIA driver")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This smoke is intended to validate a CUDA GPU")
    torch.cuda.set_device(device)

    from hstu_kvcache.data.yambda_history import load_yambda_histories
    from hstu_kvcache.models import HSTU, HSTUConfig

    dataset = json.loads(args.dataset.read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not str(checkpoint.get("status", "")).startswith("formal_") or checkpoint.get("progress") != 1.0:
        raise RuntimeError("Expected a completed formal Yambda checkpoint")

    users_path = args.dataset.parent / dataset["users_path"]
    users = pq.read_table(users_path, columns=["uid", "n_foundation_train"]).to_pylist()
    uid = next(int(row["uid"]) for row in users if int(row["n_foundation_train"]) >= 512)
    end_time = int(dataset["feedback_access"]["default_evaluation_end_exclusive"])
    history = load_yambda_histories(
        args.dataset, [uid], known_vocab_size=int(checkpoint["known_vocab_size"]),
        oov_buckets=int(checkpoint["oov_buckets"]), end_timestamp=end_time, threads=2,
    )
    timestamps, item_ids, behaviors = history.rows[uid]
    if not len(item_ids):
        raise RuntimeError("Selected Yambda user has no causal history in the evaluation prefix")
    timestamps, item_ids, behaviors = timestamps[-512:], item_ids[-512:], behaviors[-512:]
    deltas = np.zeros(len(timestamps), dtype=np.float32)
    if len(timestamps) > 1:
        deltas[1:] = np.diff(timestamps)

    model = HSTU(HSTUConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval().requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    items = torch.as_tensor(item_ids[None, :], dtype=torch.long, device=device)
    behavior_tensor = torch.as_tensor(behaviors[None, :], dtype=torch.long, device=device)
    delta_tensor = torch.as_tensor(deltas[None, :], dtype=torch.float32, device=device)
    candidate = items[:, -1:]
    query_delta = torch.tensor([float(end_time - int(timestamps[-1]))], device=device)
    query_type = torch.full((1, 1), int(checkpoint["config"]["query_type_id"]), device=device)
    query_action = torch.full((1, 1), int(checkpoint["config"]["query_action_id"]), device=device)

    with torch.inference_mode():
        scores, readouts = model.observe_cc_full(
            items, behavior_tensor, delta_tensor, candidate, query_delta,
            lengths=torch.tensor([len(timestamps)], device=device),
            query_type_ids=query_type, query_action_ids=query_action,
        )
        torch.cuda.synchronize(device)
    if not torch.isfinite(scores).all() or not torch.isfinite(readouts).all():
        raise RuntimeError("Yambda forward produced nonfinite output")
    print(json.dumps({
        "status": "passed_yambda_single_gpu_smoke",
        "host": platform.node(),
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "device": torch.cuda.get_device_name(device),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "checkpoint_version": checkpoint.get("version"),
        "model": f"{len(model.blocks)}L_H{model.cfg.hidden_size}",
        "history_rows": int(len(timestamps)),
        "uid": uid,
        "history_end_timestamp": int(timestamps[-1]),
        "query_timestamp": end_time,
        "candidate_item_idx": int(item_ids[-1]),
        "scores_shape": list(scores.shape),
        "scores": scores.cpu().tolist(),
        "readouts_shape": list(readouts.shape),
        "score_finite": True,
        "readout_finite": True,
        "quality_claim": False,
        "timing_comparison": False,
    }, indent=2))


if __name__ == "__main__":
    main()
