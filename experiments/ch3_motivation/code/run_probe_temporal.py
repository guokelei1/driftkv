#!/usr/bin/env python3
"""Fit the fixed E5 ladder with causal, per-candidate query-gap features.

Only query_delta changes relative to the pooled-fit64 calibration recipe.
Manifests are constructed from fitting users' pre-release feedback gaps;
validation draws use the same fitting-only law. No population quality runs here.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

from experiments.ch3_motivation.code.common import (
    ROOT, capture_samples, load_models, restore_pack, write_json,
)
from experiments.ch3_motivation.code import calibration_gaps
from experiments.ch3_motivation.code.prepare_candidates import candidates_for
from experiments.ch3_motivation.code.probes import fit_probe, native_query_flops, prepare_training_samples
from experiments.ch3_motivation.code.probe_variants import build_variant, load_variant, save_variant
from hstu_kvcache.evaluation.cost import CostModel
from hstu_kvcache.common.artifacts import sha256
import torch

FIXED = dict(variant="full_item", width=128, depth=1, fit_users=64, queries=16,
    learning_rate=.00003, steps=512, validation_every=8, patience=128,
    read_weight=1., score_weight=1., candidate_source="feedback", recent_fraction=.5,
    candidate_seed=17, row_chunk=1024, query_chunk=16, weight_decay=.0001, gradient_clip=5.)


def preparation_cost(model, samples, cfg, *, manifest_bytes):
    """Same Q16 native arithmetic: scalar gaps were already expanded before encoding."""
    cm = CostModel(model.cfg.hidden_size, len(model.blocks), model.cfg.num_heads, cfg["attention_backend"])
    cache_flops = target_flops = teacher_read_flops = 0
    delta_bytes = scalar_bytes = 0
    for sample in samples:
        n = sample["history_count"]
        cost = cm if n == 1024 else CostModel(cm.hidden_size, cm.num_layers, cm.num_heads, "torch")
        batch, queries = sample["candidates"].shape
        cache_flops += 2 * batch * cost.full_cache(n)
        target_flops += 2 * native_query_flops(model, history_length=n, queries=queries, batch=batch)
        teacher_read_flops += batch * queries * n * cm.num_layers * (4 * cm.hidden_size + 3 * cm.num_heads)
        delta_bytes += sample["query_delta"].numel() * sample["query_delta"].element_size()
        scalar_bytes += batch * sample["query_delta"].element_size()
    return dict(source_teacher_cache_reconstruction_flops=int(cache_flops),
        teacher_and_old_target_score_flops=int(target_flops),
        same_query_teacher_read_target_flops=int(teacher_read_flops),
        total_model_preparation_flops=int(cache_flops + target_flops + teacher_read_flops),
        query_delta_matrix_bytes=delta_bytes, release_gap_scalar_bytes=scalar_bytes,
        additional_delta_input_bytes=delta_bytes - scalar_bytes, gap_manifest_file_bytes=manifest_bytes,
        temporal_arithmetic="native embed_query_tokens expands scalar gaps to [B,Q] before Fourier features and temporal projection; per-candidate values change inputs, not Q16 model FLOPs",
        cpu_gap_preparation="manifest law construction/sampling wall time and source counts reported separately in gap provenance; not silently described as model FLOPs")


def run(args):
    started = time.perf_counter()
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    input_root, output = Path(args.input_root), Path(args.output)
    directory = input_root / f"edge_{args.edge}"
    population_path = input_root / "population.json"
    population = json.loads(population_path.read_text())
    binding_path = directory / "inputs.json"
    binding = json.loads(binding_path.read_text())
    if binding["population_sha256"] != sha256(population_path):
        raise ValueError("temporal fitting population changed")
    for name in ("history", "requests"):
        if sha256(directory / binding[name]["path"]) != binding[name]["sha256"]:
            raise ValueError(f"temporal fitting {name} pack changed")
    cfg = dict(binding["settings"], calibration_queries_per_user=16, seed=17)
    if cfg["edge"] != args.edge:
        raise ValueError("temporal fitting edge binding differs")
    fit_uids, validation_uids = population["fit"][:64], population["validation"]
    if (len(fit_uids) != 64 or len(validation_uids) != 16 or
            len(set(fit_uids + validation_uids)) != 80):
        raise ValueError("temporal fitting requires separate reserved fit64 and validation16 users")
    candidate_path = directory / "calibration_feedback_candidates.json"
    candidate_data = json.loads(candidate_path.read_text())
    if (candidate_data["population_sha256"] != sha256(population_path) or
            candidate_data["fitting_pool_uids"] != fit_uids or
            candidate_data["validation_uids"] != validation_uids):
        raise ValueError("temporal fitting requires the original pooled-fit64 candidate view")
    gaps, gap_provenance = calibration_gaps.load_gap_manifest(args.gap_manifest, input_root, args.edge, args.gap_rule)
    manifest_sha = sha256(Path(args.gap_manifest))
    if (gap_provenance["source_cutoff"] != cfg["release_day"] * 86400 or
            gap_provenance["rule"] != args.gap_rule or
            gap_provenance["manifest_sha256"] != manifest_sha or set(gaps) != set(fit_uids + validation_uids)):
        raise ValueError("temporal fitting manifest rule/cutoff/users differ")
    paths = [Path(__file__), Path(calibration_gaps.__file__), *[
        Path(__file__).with_name(name) for name in ("common.py", "probes.py", "probe_variants.py", "prepare_candidates.py")],
        ROOT / "src/hstu_kvcache/adaptation/reader.py", ROOT / "src/hstu_kvcache/models/hstu.py",
        ROOT / "src/hstu_kvcache/models/embeddings.py", ROOT / "src/hstu_kvcache/evaluation/cost.py",
        ROOT / "src/hstu_kvcache/methods/baselines/protocol/cost.py"]
    sources = {str(path.resolve().relative_to(ROOT)): sha256(path) for path in paths}
    parent, current, models = load_models(args.edge, device)
    controls = args.controls if args.controls is not None else args.gap_rule == "gap_mix50"
    identity = dict(FIXED, edge=args.edge, seed=args.seed, models=models, fit_uids=fit_uids,
        validation_users=validation_uids, stages="f,fg,fgh", controls=controls,
        input_binding_sha256=sha256(binding_path), population_sha256=sha256(population_path),
        calibration_candidate_sha256=sha256(candidate_path), numerical_sources=sources,
        fitting_driver="run_probe_temporal.py", fitting_driver_sha256=sha256(Path(__file__)),
        query_delta_rule=args.gap_rule, gap_manifest=str(Path(args.gap_manifest).resolve()),
        gap_manifest_sha256=manifest_sha, time_rule=gap_provenance["time_rule"],
        source_cutoff=gap_provenance["source_cutoff"], candidate_order=gap_provenance["candidate_order"],
        query_delta_layout="[B,16], same causal manifest for every nested stage and sibling control",
        fitting_uses_future_feedback=False)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != identity:
        raise ValueError("temporal attempt configuration changed; use a separate attempt")
    write_json(config_path, identity)
    histories = restore_pack(directory / binding["history"]["path"])
    samples = capture_samples(parent, current, histories, fit_uids + validation_uids, cfg, device)
    known = current.cfg.num_items - 256
    if known != candidate_data["known_items"]:
        raise ValueError("temporal candidate catalog differs from backbone")
    for uid, sample in samples.items():
        sample["uid"] = int(uid)
        sample["candidates"] = torch.as_tensor(candidates_for(uid,
            candidate_data["recent_items"][str(uid)], known, count=16, recent_fraction=.5, seed=17), dtype=torch.long)[None]
    samples = calibration_gaps.apply_gap_view(samples, gaps, gap_provenance)
    for sample in samples.values():
        if (sample["query_delta"].shape != sample["candidates"].shape or
                not bool(torch.isfinite(sample["query_delta"]).all()) or
                bool((sample["query_delta"] < 0).any())):
            raise ValueError("temporal candidate deltas must be finite nonnegative [B,16] matrices")
    # Gap matrices are installed BEFORE any teacher score, moment, input
    # normalization or fitting scale is computed.
    train = prepare_training_samples(current, [samples[uid] for uid in fit_uids])
    validation = prepare_training_samples(current, [samples[uid] for uid in validation_uids])
    preparation = preparation_cost(current, train + validation, cfg,
                                   manifest_bytes=Path(args.gap_manifest).stat().st_size)
    shared = preparation["total_model_preparation_flops"]
    preparation_seconds = time.perf_counter() - started
    del parent, histories, samples
    if device.type == "cuda":
        torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    probe = build_variant("full_item", current, stage="f", hidden_width=128, depth=1,
                          query_chunk=16, row_chunk=1024).to(device)
    total, summaries = shared, []

    def fit_one(module, label, steps, previous_flops):
        checkpoint, summary_path = output / f"{label}.pt", output / f"{label}.json"
        if checkpoint.exists():
            saved, metadata = load_variant(checkpoint, device)
            if metadata["identity"] != identity or metadata["method_label"] != label or saved.stage != module.stage:
                raise ValueError("existing temporal checkpoint identity/branch differs")
            summary = dict(metadata, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
            if summary_path.exists() and json.loads(summary_path.read_text()) != summary:
                raise ValueError("existing temporal checkpoint summary/hash differs")
            write_json(summary_path, summary)
            return saved, summary
        fit_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        metadata = fit_probe(current, module, train, validation, steps=steps, learning_rate=.00003,
            read_weight=1., score_weight=1., seed=args.seed, patience=128, validation_every=8,
            progress_callback=lambda row: print(json.dumps(dict(status="temporal_validation", edge=args.edge,
                query_delta_rule=args.gap_rule, method_label=label, **row)), flush=True))
        metadata.update(identity=identity, variant="full_item", method_label=label,
            gap_provenance=gap_provenance, calibration_preparation=preparation,
            source_teacher_and_target_preparation_flops=int(shared),
            cumulative_calibration_flops_estimate=int(previous_flops + metadata["total_calibration_flops_estimate"]),
            actual_preparation_seconds_this_process=preparation_seconds, actual_search_preparation_reused=False,
            seconds=time.perf_counter() - fit_started, quality_auc_computed=False,
            fitting_uses_future_feedback=False)
        if device.type == "cuda":
            metadata.update(physical_gpu=args.gpu, hardware=torch.cuda.get_device_name(0),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save_variant(checkpoint, module, metadata)
        summary = dict(metadata, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
        write_json(summary_path, summary)
        return module, summary

    for stage in ("f", "fg", "fgh"):
        if stage != probe.stage:
            probe = probe.expanded(stage)
        probe, summary = fit_one(probe, stage, 512, total)
        total = summary["cumulative_calibration_flops_estimate"]
        summaries.append(summary)
        if controls and stage in ("f", "fg"):
            _, control = fit_one(copy.deepcopy(probe), stage + "_extended", 1024 if stage == "f" else 512, total)
            summaries.append(control)
    for name, digest in sources.items():
        if sha256(ROOT / name) != digest:
            raise ValueError("temporal numerical source changed during fitting")
    if sha256(Path(args.gap_manifest)) != manifest_sha:
        raise ValueError("temporal gap manifest changed during fitting")
    result = dict(status="fitted", identity=identity, stages=summaries,
        scope="pre-release temporal-feature calibration only; no post-release labels or population AUC",
        calibration_preparation=preparation, gap_provenance=gap_provenance)
    write_json(output / "summary.json", result)
    print(json.dumps(dict(status="complete", edge=args.edge, query_delta_rule=args.gap_rule,
                         output=str(output), controls=controls)), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path,
        default=ROOT / "experiments/ch3_motivation/inputs_feedback_pool/fit64")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--edge", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--gap-manifest", type=Path, required=True)
    parser.add_argument("--gap-rule", choices=("gap_mix50", "gap_empirical100"), required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--controls", action=argparse.BooleanOptionalAction, default=None)
    run(parser.parse_args())
