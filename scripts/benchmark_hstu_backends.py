#!/usr/bin/env python3
"""Bounded numerical/performance comparison; never fit or rewrite checkpoints.

Synthetic six-layer backbone timings exclude data loading, optimizer updates,
and ranking heads. Optional retained-checkpoint probes separately cover a real
RecFlow development batch and existing Yambda Medium weights. These are backend
equivalence checks, not model admission or recommendation quality experiments.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "recflow"))

from hstu_kvcache.models.attention import PointwiseAttention  # noqa: E402
from hstu_kvcache.models.hstu import HSTU, HSTUConfig  # noqa: E402


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def backend(model, name):
    for module in model.modules():
        if isinstance(module, PointwiseAttention):
            module.backend = name


def delta(reference, value):
    reference, value = reference.double(), value.double()
    difference = value - reference
    return dict(max_abs=float(difference.abs().max()),
                relative_l2=float(difference.norm() / reference.norm().clamp_min(1e-30)),
                finite=bool(torch.isfinite(value).all()))


def capture_gradients(model, embedding_rows=None):
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        value = parameter.grad.detach()
        if name.endswith("item_emb.weight") and embedding_rows is not None:
            value = value[embedding_rows]
        result[name] = value.cpu().clone()
    return result


def gradient_delta(reference, value):
    assert reference.keys() == value.keys(), "Backend changed parameter gradient coverage"
    entries = {name: delta(reference[name], value[name]) for name in reference}
    return dict(finite=all(row["finite"] for row in entries.values()),
                worst_relative_l2=max(entries.values(), key=lambda row: row["relative_l2"])["relative_l2"],
                parameters=entries)


def timing(fn, args):
    torch.cuda.synchronize(args.device)
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize(args.device)
    cold_ms = (time.perf_counter() - start) * 1000
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    start = time.perf_counter()
    for _ in range(args.iterations):
        fn()
    torch.cuda.synchronize(args.device)
    return dict(first_call_ms=cold_ms,
                warmed_wall_ms=(time.perf_counter() - start) * 1000 / args.iterations,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                iterations=args.iterations)


def synthetic(args):
    reports = []
    for variant in args.variants:
        cfg = HSTUConfig(num_items=4096, num_behaviors=4, hidden_size=192,
                         num_layers=6, num_heads=6, max_seq_len=max(1027, args.length),
                         input_dropout=0, block_variant=variant,
                         activation="silu" if variant == "hstu_reference" else "elu_plus1",
                         relative_position_bias=variant == "hstu_reference")
        torch.manual_seed(args.seed)
        model = HSTU(cfg).to(args.device).eval()
        x = torch.randn(args.batch_size, args.length, cfg.hidden_size, device=args.device)
        target = torch.randn_like(x)
        lengths = torch.full((len(x),), args.length, device=args.device, dtype=torch.long)
        if len(x) > 1:
            lengths[-1] = max(1, args.length // 2)
        for precision in args.precisions:
            dtype = getattr(torch, precision)
            reference = None
            for name in args.backends:
                backend(model, name)

                def forward(model=model, x=x, lengths=lengths, dtype=dtype):
                    with torch.no_grad(), torch.autocast("cuda", dtype=dtype,
                                                        enabled=dtype != torch.float32):
                        return model.forward_embedded(x, return_kv=True, lengths=lengths)

                def backward(model=model, x=x, lengths=lengths, dtype=dtype, target=target):
                    model.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                        hidden, _ = model.forward_embedded(x, lengths=lengths)
                        loss = (hidden.float() - target).square().mean()
                    loss.backward()
                    return loss

                model.zero_grad(set_to_none=True)
                forward_time = timing(forward, args)
                hidden, cache = forward()
                snapshot = dict(hidden=hidden.cpu(), k=cache.k.cpu(), v=cache.v.cpu())
                del hidden, cache
                backward_time = timing(backward, args)
                snapshot.update(loss=backward().detach().cpu(), gradients=capture_gradients(model))
                if reference is None:
                    reference = snapshot
                parity = {key: delta(reference[key], snapshot[key])
                          for key in ("hidden", "k", "v", "loss")}
                parity["gradients"] = gradient_delta(reference["gradients"], snapshot["gradients"])
                report = dict(scope="synthetic_six_layer_backbone", variant=variant,
                              precision=precision, backend=name, configuration=asdict(cfg),
                              shape=list(x.shape), last_row_length=int(lengths[-1]),
                              forward_kv=forward_time, forward_backward=backward_time, parity=parity)
                reports.append(report)
                print(json.dumps({key: value for key, value in report.items()
                                  if key not in ("configuration", "parity")}), flush=True)
            model.zero_grad(set_to_none=True)
        del model, x, target, reference, snapshot, forward, backward
        gc.collect()
        torch.cuda.empty_cache()
    return reports


def yambda_checkpoint(args):
    path = args.yambda_checkpoint.resolve()
    # Only the explicitly supplied trusted local file is read; theta3 is not discovered.
    saved = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    cfg = HSTUConfig(**saved["config"])
    assert cfg.num_layers == 6
    model = HSTU(cfg)
    model.load_state_dict(saved["model"], strict=True)
    model.to(args.device).eval()
    torch.manual_seed(args.seed)
    width = min(args.length, cfg.max_seq_len - 3)
    items = torch.randint(1, cfg.num_items, (2, width), device=args.device)
    behaviors = torch.zeros_like(items)
    times = torch.rand(2, width, device=args.device) * 100
    candidates = torch.randint(1, cfg.num_prediction_items, (2, 1000), device=args.device)
    results, reference = {}, None
    with torch.no_grad():
        for name in args.backends:
            backend(model, name)
            hidden, cache = model(items, behaviors, times, return_kv=True)
            scores = model.score_hidden(hidden[:, -1], candidates)
            new_hidden, new_cache = model.forward_with_cache(cache, items[:, :2], behaviors[:, :2], times[:, :2])
            snapshot = dict(hidden=hidden.cpu(), k=cache.k.cpu(), v=cache.v.cpu(), scores=scores.cpu(),
                            cached_hidden=new_hidden.cpu(), cached_k=new_cache.k.cpu(), cached_v=new_cache.v.cpu())
            if reference is None:
                reference = snapshot
            results[name] = {key: delta(reference[key], snapshot[key]) for key in snapshot}
            results[name]["top50_same_order"] = bool(torch.equal(
                reference["scores"].topk(50).indices, snapshot["scores"].topk(50).indices))
    report = dict(checkpoint=str(path), checkpoint_sha256=digest(path), configuration=asdict(cfg),
                  strict_load=True, input_scope="synthetic IDs; no Yambda held-out data read", backends=results)
    del model, saved
    gc.collect()
    torch.cuda.empty_cache()
    return report


def recflow_checkpoint(args):
    from development_probe import PreparedRecFlow, ProbeData, RecFlowGenerator

    path = args.recflow_checkpoint.resolve()
    saved = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    configuration = saved["configuration"]
    cfg = HSTUConfig(**configuration["model"])
    assert cfg.num_layers == 6
    prepared = PreparedRecFlow(configuration["data"])
    dataset = ProbeData(prepared, configuration["catalog_size"], configuration["cohort_users"], configuration["context"])
    assert hashlib.sha256(dataset.raw_ids.tobytes()).hexdigest() == configuration["catalog_sha256"]
    assert hashlib.sha256(np.asarray(dataset.uids).tobytes()).hexdigest() == configuration["cohort_sha256"]
    indices = dataset.indices(19, 19, args.real_requests * 4)
    indices = np.asarray([index for index in indices if len(dataset.targets(index)[1])])[:args.real_requests]
    assert len(indices) == args.real_requests and np.all(prepared.requests["role"][indices] == 0)
    batch = dataset.batch(indices, args.device, np.random.default_rng(args.seed))
    model = RecFlowGenerator(cfg, torch.tensor(dataset.paths), history_categories=configuration.get("history_categories", False))
    model.load_state_dict(saved["model"], strict=True)
    model.to(args.device).eval()
    inference_batch = {key: value[:2] for key, value in batch.items() if key != "targets"}
    # This is a fixed numerical probe, not an additional quality metric/protocol.
    rng = np.random.default_rng(args.seed)
    candidates = torch.tensor(np.stack([rng.choice(np.arange(1, model.num_known + 1), 1000, replace=False)
                                        for _ in range(len(inference_batch["item_ids"]))]), device=args.device)
    embedding_rows = torch.unique(torch.cat([batch["item_ids"].flatten(), batch["targets"], candidates.flatten()]))
    results, reference, reference_bf16_loss = {}, None, None
    for name in args.backends:
        backend(model, name)
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            log_probs = model.teacher_log_probs(**batch)
            cache = model._prefix_cache(*(batch[key] for key in ("item_ids", "behaviors", "time_deltas", "lengths")))
            score_time = timing(lambda: model.score_items(**inference_batch, candidate_ids=candidates), args)
            scores = model.score_items(**inference_batch, candidate_ids=candidates)
            snapshot = dict(log_probs=log_probs.cpu(), k=cache.k.cpu(), v=cache.v.cpu(), scores=scores.cpu())
        model.zero_grad(set_to_none=True)
        loss = model.loss_per_example(**batch).mean()
        loss.backward()
        snapshot.update(loss=loss.detach().cpu(), gradients=capture_gradients(model, embedding_rows))
        if reference is None:
            reference = snapshot
        parity = {key: delta(reference[key], snapshot[key]) for key in ("log_probs", "k", "v", "scores", "loss")}
        parity["gradients"] = gradient_delta(reference["gradients"], snapshot["gradients"])
        parity["top50_same_order"] = bool(torch.equal(reference["scores"].topk(50).indices, snapshot["scores"].topk(50).indices))
        def train_step():
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                current_loss = model.loss_per_example(**batch).mean()
            current_loss.backward()
            return current_loss

        training_time = timing(train_step, args)
        bf16_loss = train_step().detach().cpu()
        if reference_bf16_loss is None:
            reference_bf16_loss = bf16_loss
        results[name] = dict(parity=parity, candidate_scoring=score_time,
                             bf16_forward_backward=training_time,
                             bf16_loss=delta(reference_bf16_loss, bf16_loss))
        print(json.dumps(dict(scope="retained_recflow_checkpoint", backend=name,
                              loss=float(loss.detach()), candidate_scoring=score_time)), flush=True)
    return dict(checkpoint=str(path), checkpoint_sha256=digest(path), configuration=asdict(cfg), strict_load=True,
                day=19, role="development", request_indices=indices.tolist(), histories=batch["lengths"].tolist(),
                candidate_count=1000, candidate_seed=args.seed, precision="float32", backends=results,
                gradient_scope="All parameters; item embedding gradient compared on fixed history/target/candidate rows.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--backends", nargs="+", default=["torch", "triton"], choices=["torch", "triton", "auto"])
    parser.add_argument("--variants", nargs="+", default=["legacy", "hstu_reference"], choices=["legacy", "hstu_reference"])
    parser.add_argument("--precisions", nargs="+", default=["float32", "bfloat16"], choices=["float32", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--real-requests", type=int, default=8)
    parser.add_argument("--recflow-checkpoint", type=Path)
    parser.add_argument("--yambda-checkpoint", type=Path)
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.backends[0] != "torch":
        parser.error("The first backend must be torch, the retained numerical reference.")
    if args.output.exists():
        parser.error("Use a fresh output path; retained measurements are never overwritten.")
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    sources = [Path(__file__).resolve(), *(ROOT / "src/hstu_kvcache/models" / name for name in
               ("attention.py", "triton_attention.py", "hstu.py", "block.py")),
               ROOT / "src/hstu_kvcache/recflow/model.py"]
    report = dict(scope="Backend validation only; no fitting, admission, final-role data, or checkpoint writes.",
                  torch=torch.__version__, python=platform.python_version(),
                  gpu=torch.cuda.get_device_name(args.device), device=args.device, seed=args.seed,
                  timing="Warmed synchronous wall time; first call separate and may include compilation. Synthetic forward/backward excludes optimizer/data/head; RecFlow scoring includes its existing Python decoder.",
                  source_sha256={str(path.relative_to(ROOT)): digest(path) for path in sources})
    if not args.skip_synthetic:
        report["synthetic"] = synthetic(args)
    if args.yambda_checkpoint:
        report["yambda_checkpoint"] = yambda_checkpoint(args)
    if args.recflow_checkpoint:
        report["recflow_checkpoint"] = recflow_checkpoint(args)
    report["source_sha256_after"] = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    report["sources_unchanged_during_probe"] = report["source_sha256"] == report["source_sha256_after"]
    checks = []

    def check_tree(value, tolerance):
        if not isinstance(value, dict):
            return
        if "relative_l2" in value:
            checks.append(value["finite"] and value["relative_l2"] <= tolerance)
        for child in value.values():
            check_tree(child, tolerance)

    for row in report.get("synthetic", []):
        check_tree(row["parity"], 1e-5 if row["precision"] == "float32" else 0.02)
    for checkpoint in ("yambda_checkpoint", "recflow_checkpoint"):
        for row in report.get(checkpoint, {}).get("backends", {}).values():
            parity = row.get("parity", row)
            check_tree(parity, 1e-5)
            check_tree(row.get("bf16_loss", {}), 1e-3)
            checks.append(parity["top50_same_order"])
    report["acceptance"] = dict(
        scope="This bounded fixture only; no claim of bitwise training reproducibility or all-request ranking parity.",
        fp32_relative_l2_tolerance=1e-5, bf16_relative_l2_tolerance=0.02,
        retained_checkpoint_bf16_loss_relative_tolerance=1e-3,
        retained_checkpoint_fp32_top50_same_order_required=True,
        passed=all(checks) and report["sources_unchanged_during_probe"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(str(args.output), flush=True)
    if not report["acceptance"]["passed"]:
        raise RuntimeError("Backend probe did not meet its numerical/source-stability checks; inspect retained report")


if __name__ == "__main__":
    main()
