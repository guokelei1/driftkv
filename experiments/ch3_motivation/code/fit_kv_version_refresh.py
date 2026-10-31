#!/usr/bin/env python3
"""Fit a small refresh map for a frozen, already-mapped cache snapshot.

This is a version-mismatch mechanism diagnostic.  It uses no labels and no
continuous serving trajectory.  The new map's INPUT is exactly the cache
materialized by a previous F, using that F's original target item embeddings.
It is not an adjacent native-KV map and is not the candidate-only g term.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import json
import os
from pathlib import Path
import sys
import time

from experiments.ch3_motivation.code.common import ROOT, capture_samples, restore_pack, write_json
from experiments.ch3_motivation.code.prepare_candidates import candidates_for
from experiments.ch3_motivation.code.probe_cost import cached_preparation_flops
from experiments.ch3_motivation.code.probe_variants import build_variant, load_variant, save_variant
from experiments.ch3_motivation.code.probes import native_query_flops, prepare_training_samples
from experiments.ch3_motivation.code.query_centering import fit_query_conditioned
from hstu_kvcache.evaluation.yambda import load_model
from hstu_kvcache.evaluation.cost import CostModel
from hstu_kvcache.common.artifacts import sha256
import torch


@torch.no_grad()
def materialize_predecessor_sources(samples, predecessor, old_target):
    """Replace native source aliases with actual previous-release mapped K/V.

    Materialization must precede target/scale preparation.  It uses old_target
    only for the ORIGINAL item embeddings; old_target does not prefill a
    teacher here.  The new target model will later supply its own item inputs.
    """
    device = next(old_target.parameters()).device
    flops = embedding_bytes = rows = 0
    for sample in samples.values():
        if any(name in sample for name in ("teacher_scores", "loss_scale_version", "item_embeddings")):
            raise ValueError("materialize the predecessor before preparing new-target moments")
        source = sample["parent_cache"].to(device)
        ids = sample["item_ids"].to(device)
        mapped = predecessor.prepare_cache(old_target, source, ids).to("cpu")
        sample["parent_cache"] = sample["parent"] = mapped
        flops += cached_preparation_flops(predecessor, source.seq_len)
        embedding_bytes += ids.numel() * old_target.cfg.hidden_size * source.k.element_size()
        rows += source.seq_len
    return dict(predecessor_materialization_flops=int(flops),
                predecessor_item_embedding_bytes=int(embedding_bytes),
                predecessor_materialized_rows=int(rows),
                predecessor_materialized_users=len(samples),
                embedding_lookup_scope="gathers are recorded in bytes, not floating-point arithmetic")


def preparation_cost(model, samples, cfg):
    """Same cache/query/read preparation convention as the existing F fitter."""
    cm = CostModel(model.cfg.hidden_size, len(model.blocks), model.cfg.num_heads,
                   cfg["attention_backend"])
    cache = queries = reads = 0
    for sample in samples:
        n = sample["history_count"]
        cost = cm if n == 1024 else CostModel(cm.hidden_size, cm.num_layers, cm.num_heads, "torch")
        batch, q = sample["candidates"].shape
        cache += 2 * batch * cost.full_cache(n)
        queries += 2 * native_query_flops(model, history_length=n, queries=q, batch=batch)
        reads += batch * q * n * cm.num_layers * (4 * cm.hidden_size + 3 * cm.num_heads)
    return dict(source_and_new_teacher_cache_reconstruction_flops=int(cache),
                new_target_and_teacher_score_preparation_flops=int(queries),
                same_query_teacher_read_target_flops=int(reads),
                total_model_preparation_flops=int(cache + queries + reads))


def _model_path(version):
    return ROOT / f"results/unified_training_2026_09/large/seed17/checkpoints/v{version}/checkpoint_100.pt"


def _load(version, device):
    model, payload = load_model(_model_path(version), device)
    del payload
    model.eval().requires_grad_(False)
    if model.cfg.hidden_size != 320 or len(model.blocks) != 10:
        raise ValueError("expected the fixed Large model family")
    return model


def run(args):
    started = time.perf_counter()
    if args.gpu != 1 or args.target_version != args.history_edge + 1:
        raise ValueError("this two-case study uses GPU1 and the next target version")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["EVOKV_ATTENTION_BACKEND"] = "triton"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    input_root = ROOT / "experiments/ch3_motivation/inputs_feedback_pool/fit256"
    population_path = input_root / "population.json"
    population = json.loads(population_path.read_text())
    directory = input_root / f"edge_{args.history_edge}"
    binding_path = directory / "inputs.json"
    binding = json.loads(binding_path.read_text())
    if binding["population_sha256"] != sha256(population_path):
        raise ValueError("frozen fitting roles changed")
    for name in ("history", "requests"):
        if sha256(directory / binding[name]["path"]) != binding[name]["sha256"]:
            raise ValueError("frozen input pack changed")
    cfg = dict(binding["settings"], calibration_queries_per_user=16, seed=17)
    fit_uids, validation_uids = population["fit"][:100], population["validation"]
    if len(fit_uids) != 100 or len(validation_uids) != 16 or len(set(fit_uids + validation_uids)) != 116:
        raise ValueError("requires the reserved, separate fit100/validation16 roles")
    candidate_path = directory / "calibration_feedback_candidates.json"
    candidate_data = json.loads(candidate_path.read_text())
    if candidate_data["population_sha256"] != sha256(population_path):
        raise ValueError("candidate calibration roles changed")
    source_version, old_target_version = args.history_edge - 1, args.history_edge
    model_paths = [_model_path(v) for v in (source_version, old_target_version, args.target_version)]
    models = {str(p.relative_to(ROOT)): sha256(p) for p in model_paths}
    predecessor, prior = load_variant(args.predecessor_checkpoint, "cpu")
    prior_identity = prior["identity"]
    prior_models = {str(p.relative_to(ROOT)): models[str(p.relative_to(ROOT))] for p in model_paths[:2]}
    if (predecessor.stage != "f" or prior_identity["models"] != prior_models
            or prior_identity["input_binding_sha256"] != sha256(binding_path)
            or prior_identity["fit_uids"] != fit_uids
            or prior_identity["validation_users"] != validation_uids):
        raise ValueError("predecessor must be the actual F for these source/history/roles")
    paths = [Path(__file__), *[Path(__file__).with_name(name) for name in
             ("common.py", "prepare_candidates.py", "probe_cost.py", "probes.py", "probe_variants.py",
              "query_centering.py", "resident_probe_fitting.py", "probe_first_affine.py")],
             ROOT / "src/hstu_kvcache/adaptation/reader.py",
             ROOT / "src/hstu_kvcache/evaluation/yambda.py",
             ROOT / "src/hstu_kvcache/evaluation/cost.py"]
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    identity = dict(protocol="frozen_mapped_source_version_refresh_v1",
        history_edge=args.history_edge, source_native_version=source_version,
        predecessor_target_version=old_target_version, target_version=args.target_version,
        history_cutoff=cfg["release_day"] * 86400,
        query_delta_rule="fixed_original_history_cutover_gap; identical to predecessor input view",
        source_kind="actual_predecessor_mapped_cache_not_native_old_target",
        predecessor_checkpoint=str(args.predecessor_checkpoint.resolve()),
        predecessor_checkpoint_sha256=sha256(args.predecessor_checkpoint),
        predecessor_prior_calibration_flops=prior["cumulative_calibration_flops_estimate"],
        models=models, input_root=str(input_root), input_binding_sha256=sha256(binding_path),
        population_sha256=sha256(population_path), fit_uids=fit_uids, validation_users=validation_uids,
        fit_users=100, teacher_fitting_users=100, gradient_fitting_users=100,
        variant="full_item", width=64, depth=1, queries=16, seed=17,
        candidate_seed=17, candidate_source="feedback", recent_fraction=.75,
        calibration_candidate_sha256=sha256(candidate_path), steps=256, learning_rate=1e-4,
        read_weight=1., score_weight=1., validation_every=32, patience=0,
        query_conditioning_mode="center_std", numerical_sources=sources,
        fitting_driver=Path(__file__).name, quality_auc_computed=False,
        scope="Fixed-history, label-free version diagnostic; refresh_map is a KV map, not candidate g.")
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != identity:
        raise ValueError("version-refresh configuration changed; use a new output attempt")
    write_json(config_path, identity)
    checkpoint = args.output / "refresh_map.pt"
    if checkpoint.exists():
        saved, metadata = load_variant(checkpoint, "cpu")
        if saved.stage != "f" or metadata["identity"] != identity or metadata["method_label"] != "refresh_map":
            raise ValueError("saved refresh map differs from this mapped-source case")
        print(json.dumps(dict(status="already_complete", checkpoint=str(checkpoint))), flush=True)
        return metadata

    source, current = _load(source_version, device), _load(args.target_version, device)
    histories = restore_pack(directory / binding["history"]["path"])
    samples = capture_samples(source, current, histories, fit_uids + validation_uids, cfg, device)
    del source, histories
    torch.cuda.empty_cache()
    old_target = _load(old_target_version, device)
    predecessor = predecessor.to(device).eval().requires_grad_(False)
    if old_target.cfg.num_items != current.cfg.num_items or candidate_data["known_items"] != current.cfg.num_items - 256:
        raise ValueError("fixed candidate catalog differs across these models")
    source_materialization = materialize_predecessor_sources(samples, predecessor, old_target)
    del predecessor, old_target
    torch.cuda.empty_cache()
    for uid, sample in samples.items():
        sample["candidates"] = torch.as_tensor(candidates_for(uid,
            candidate_data["recent_items"][str(uid)], candidate_data["known_items"], count=16,
            recent_fraction=.75, seed=17), dtype=torch.long)[None]
    train = prepare_training_samples(current, [samples[u] for u in fit_uids])
    validation = prepare_training_samples(current, [samples[u] for u in validation_uids])
    preparation = preparation_cost(current, train + validation, cfg)
    shared = preparation["total_model_preparation_flops"] + source_materialization["predecessor_materialization_flops"]
    preparation_seconds = time.perf_counter() - started
    del samples
    torch.manual_seed(17)
    probe = build_variant("full_item", current, stage="f", hidden_width=64, depth=1,
                          query_chunk=16, row_chunk=1024).to(device)
    fitting_started = time.perf_counter()
    metadata = fit_query_conditioned(current, probe, train, validation,
        mode="center_std", factorized_h=True, steps=256, learning_rate=1e-4,
        read_weight=1., score_weight=1., seed=17, patience=0, validation_every=32,
        progress_callback=lambda row: print(json.dumps(dict(status="refresh_validation",
            history_edge=args.history_edge, target_version=args.target_version, **row)), flush=True))
    metadata.update(identity=identity, variant="full_item", method_label="refresh_map",
        preparation=preparation, predecessor_source_materialization=source_materialization,
        source_teacher_and_target_preparation_flops=int(shared),
        cumulative_calibration_flops_estimate=int(shared + metadata["total_calibration_flops_estimate"]),
        preparation_seconds=preparation_seconds, fitting_seconds=time.perf_counter()-fitting_started,
        seconds=time.perf_counter()-started, physical_gpu=1,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        quality_auc_computed=False,
        cost_scope="Incremental refresh fitting includes reconstruction of calibration source/teacher and predecessor source materialization. Prior F training is recorded separately as sunk prior-release work; population teacher diagnostics are separate research cost.")
    for name, digest in sources.items():
        if sha256(ROOT / name) != digest:
            raise ValueError("version-refresh numerical sources changed during fitting")
    save_variant(checkpoint, probe, metadata)
    summary = dict(metadata, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), status="fitted")
    write_json(args.output / "refresh_map.json", summary)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(dict(status="complete", checkpoint=str(checkpoint), history_edge=args.history_edge,
                          target_version=args.target_version)), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-edge", type=int, choices=(2, 3), required=True)
    parser.add_argument("--target-version", type=int, choices=(3, 4), required=True)
    parser.add_argument("--predecessor-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    run(parser.parse_args())
