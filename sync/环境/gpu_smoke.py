#!/usr/bin/env python3
"""Synthetic environment check; no dataset, trained weights or quality claim."""

from __future__ import annotations

import json
import platform
import tempfile
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import triton
import triton.language as tl

from hstu_kvcache.models import HSTU, HSTUConfig


@triton.jit
def add_one(source, destination, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(destination + offsets, tl.load(source + offsets) + 1)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.set_num_threads(2)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Exercise the same storage libraries used by the real-history loader.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rows.parquet"
        pq.write_table(pa.table({"uid": [1, 2], "item": [3, 4]}), path)
        with duckdb.connect() as connection:
            rows = connection.execute(
                "SELECT sum(item) FROM read_parquet(?)", [str(path)]
            ).fetchone()[0]
        if rows != 7:
            raise RuntimeError("Parquet/DuckDB round trip failed")

    items = torch.randint(1, 128, (2, 32))
    behaviors = torch.randint(0, 4, (2, 32))
    deltas = torch.ones(2, 32)
    candidates = torch.tensor([[7, 9], [11, 13]])
    query_deltas = torch.ones(2)
    inputs = (items, behaviors, deltas, candidates, query_deltas)
    checks = []
    for variant in ("legacy", "hstu_reference"):
        model = HSTU(HSTUConfig(
            num_items=128, num_behaviors=4, hidden_size=64,
            num_layers=2, num_heads=2, max_seq_len=64,
            input_dropout=0.0, block_variant=variant,
            activation="silu" if variant == "hstu_reference" else "elu_plus1",
            relative_position_bias=variant == "hstu_reference",
        )).eval()
        with torch.no_grad():
            expected, _ = model.observe_cc_full(*inputs)
        for index in range(torch.cuda.device_count()):
            device = torch.device(f"cuda:{index}")
            with torch.cuda.device(device):
                source = torch.arange(32, dtype=torch.float32, device=device)
                destination = torch.empty_like(source)
                add_one[(1,)](source, destination, BLOCK=32)
                torch.testing.assert_close(destination, source + 1)
                model.to(device)
                gpu_inputs = tuple(value.to(device) for value in inputs)
                with torch.no_grad():
                    scores, readouts = model.observe_cc_full(*gpu_inputs)
                    cache = model.compute_kv(*gpu_inputs[:3])
                    reused = model.score_cc_reuse(cache, *gpu_inputs[3:])
                torch.testing.assert_close(scores.cpu(), expected, atol=1e-4, rtol=1e-4)
                torch.testing.assert_close(reused, scores, atol=1e-5, rtol=1e-5)
                if not torch.isfinite(readouts).all():
                    raise RuntimeError("Nonfinite model readout")

                # Check autograd without updating or saving any model weights.
                model.zero_grad(set_to_none=True)
                loss = model.score_cc_full(*gpu_inputs).square().mean()
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise RuntimeError("Missing or nonfinite model gradients")
                torch.cuda.synchronize(device)
                checks.append({
                    "device_index": index,
                    "device": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "block_variant": variant,
                    "max_cpu_gpu_score_error": float((scores.cpu() - expected).abs().max()),
                    "max_full_reuse_score_error": float((scores - reused).abs().max()),
                    "backward_loss": float(loss.detach()),
                    "gradients_finite": True,
                    "triton_jit_passed": True,
                })
                model.cpu()

    print(json.dumps({
        "status": "passed_synthetic_gpu_smoke",
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "triton": triton.__version__,
        "seed": 17,
        "model": "2L_H64; batch=2; history=32; candidates=2; FP32; TF32 disabled",
        "parquet_duckdb_roundtrip": True,
        "checks": checks,
        "synthetic_inputs": True,
        "quality_claim": False,
        "timing_comparison": False,
        "ddp_validated": False,
    }, indent=2))


if __name__ == "__main__":
    main()
