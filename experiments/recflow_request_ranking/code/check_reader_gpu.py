"""Bounded synthetic GPU numerical audit; no fitting, throughput or quality claim."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

from hstu_kvcache.models.hstu import HSTU

from .configuration import DEFAULT_CONFIG, backbone_config, load_proposal
from .objective import ranking_loss
from .shared_reader import score_full


def run(device):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = HSTU(backbone_config(load_proposal(), "pilot", num_items=512)).to(device).eval()
    items = torch.randint(1, 513, (2, 1024), device=device)
    behaviors = torch.randint(1, 5, (2, 1024), device=device)
    deltas = torch.rand(2, 1024, device=device) * 30
    lengths = torch.tensor([1024, 733], device=device)
    candidates = torch.randint(1, 513, (2, 17), device=device)
    times = torch.tensor([2., 5.], device=device)
    labels = torch.full_like(candidates, -1)
    labels[:, :4] = 0
    labels[:, 0] = 1
    labels[1, 2] = 1
    mask = torch.ones_like(labels, dtype=torch.bool)
    rows = []
    for precision, dtype in [("fp32", None), ("bf16", torch.bfloat16)]:
        results = {}
        for name, backend, native in [("native_torch", "torch", True),
                                      ("shared_torch", "torch", False),
                                      ("shared_triton_prefix", "triton", False)]:
            current = copy.deepcopy(model)
            for block in current.blocks:
                block.attn.backend = backend
            context = nullcontext() if dtype is None else torch.autocast("cuda", dtype=dtype)
            started = time.monotonic()
            with context:
                if native:
                    logits = current.score_cc_full(items, behaviors, deltas, candidates, times, lengths=lengths)
                else:
                    logits = score_full(current, items, behaviors, deltas, lengths, candidates, times, chunk_size=8)
                loss, _ = ranking_loss(logits, labels, mask)
            loss.backward()
            torch.cuda.synchronize(device)
            gradients = torch.cat([p.grad.flatten() for p in current.parameters() if p.grad is not None])
            results[name] = (logits.detach().float().cpu(), gradients.detach().float().cpu())
            print(json.dumps({"precision": precision, "path": name,
                              "seconds_diagnostic_only": time.monotonic() - started}), flush=True)
            del current, gradients, logits, loss
        reference_scores, reference_gradient = results["native_torch"]
        # Fixed before execution. BF16 compares arithmetic implementations, not
        # a statistical metric; actual trained-model score ordering is still pending.
        limits = {"score_max_abs": 1e-4, "gradient_relative_l2": 1e-3} if dtype is None else {
            "score_max_abs": .02, "gradient_relative_l2": .08}
        for name in ["shared_torch", "shared_triton_prefix"]:
            scores, gradient = results[name]
            score_error = float((scores - reference_scores).abs().max())
            gradient_error = float((gradient - reference_gradient).norm() / reference_gradient.norm().clamp_min(1e-12))
            finite = bool(torch.isfinite(scores).all() and torch.isfinite(gradient).all())
            rows.append({"precision": precision, "path": name, "reference": "native_torch",
                         "score_max_abs": score_error, "gradient_relative_l2": gradient_error,
                         "finite": finite, "limits": limits,
                         "passed": finite and score_error <= limits["score_max_abs"]
                         and gradient_error <= limits["gradient_relative_l2"]})
    # Three candidate chunks, including a one-candidate final chunk. This
    # checks gradient flow through checkpointed queries back into shared KV.
    wide_candidates = torch.stack([torch.randperm(512, device=device)[:257] + 1 for _ in range(2)])
    checkpoint_results = []
    for checkpointed in [False, True]:
        current = copy.deepcopy(model)
        for block in current.blocks:
            block.attn.backend = "triton"
        scores = score_full(current, items, behaviors, deltas, lengths, wide_candidates, times,
                            chunk_size=128, checkpoint_queries=checkpointed, validate=False)
        scores.square().mean().backward()
        gradient = torch.cat([p.grad.flatten() for p in current.parameters() if p.grad is not None])
        checkpoint_results.append((scores.detach().cpu(), gradient.detach().cpu()))
        del current, scores, gradient
    direct, checked = checkpoint_results
    score_error = float((direct[0] - checked[0]).abs().max())
    gradient_error = float((direct[1] - checked[1]).norm() / direct[1].norm().clamp_min(1e-12))
    rows.append({"precision": "fp32", "path": "checkpointed_257_candidates_chunk128",
                 "reference": "uncheckpointed_shared_triton_prefix", "score_max_abs": score_error,
                 "gradient_relative_l2": gradient_error,
                 "limits": {"score_max_abs": 1e-5, "gradient_relative_l2": 1e-4},
                 "passed": score_error <= 1e-5 and gradient_error <= 1e-4})
    return {"created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "scope": "synthetic_numerical_only_not_training_or_throughput",
            "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
            "torch": torch.__version__, "history_width": 1024, "true_lengths": [1024, 733],
            "layers": 2, "hidden_size": 128, "candidates": 17, "candidate_chunk": 8,
            "rows": rows, "passed": all(row["passed"] for row in rows),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "model_training": False, "quality_result": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Use a new output path; numerical evidence must not be overwritten")
    report = run(torch.device(args.device))
    root = Path(__file__).resolve().parents[3]
    paths = list(Path(__file__).parent.glob("*.py")) + [DEFAULT_CONFIG]
    paths += list((root / "src/hstu_kvcache/models").glob("*.py"))
    report["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
