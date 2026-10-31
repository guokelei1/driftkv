#!/usr/bin/env python3
"""Supplementary fixed-predecessor E5 fitting from an audited trained f.

Reuse the identical f state and standalone f calibration cost, then fit only
new g and only new h, with immutable K/V inputs resident per stage. All outputs
have an explicit separate protocol identity. This is not the joint reference.
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
from types import SimpleNamespace

import torch
from experiments.ch3_motivation.code import calibration_sampling
from experiments.ch3_motivation.code.probe_initialization import (
    MODES, initialization_spec, apply_h_initialization,
)
from experiments.ch3_motivation.code.common import (
    ROOT, capture_samples, load_models, restore_pack, write_json,
)
from experiments.ch3_motivation.code.probes import native_query_flops, prepare_training_samples
from experiments.ch3_motivation.code.frozen_predecessor_fitting import fit_probe_new_term, protocol_binding
from experiments.ch3_motivation.code import resident_probe_fitting
from functools import partial
fit_probe = partial(fit_probe_new_term, resident_inputs=True)
from experiments.ch3_motivation.code.probe_variants import build_variant, load_variant, save_variant
from hstu_kvcache.evaluation.cost import CostModel
from hstu_kvcache.common.artifacts import sha256


def preparation_cost(model, samples, cfg, manifest):
    cm = CostModel(model.cfg.hidden_size, len(model.blocks), model.cfg.num_heads, cfg["attention_backend"])
    reconstruction = queries = reads = delta_bytes = 0
    for sample in samples:
        n = sample["history_count"]
        cost = cm if n == 1024 else CostModel(cm.hidden_size, cm.num_layers, cm.num_heads, "torch")
        batch, count = sample["candidates"].shape
        reconstruction += 2 * batch * cost.full_cache(n)
        queries += 2 * native_query_flops(model, history_length=n, queries=count, batch=batch)
        reads += batch * count * n * cm.num_layers * (4 * cm.hidden_size + 3 * cm.num_heads)
        delta_bytes += sample["query_delta"].numel() * sample["query_delta"].element_size()
    return dict(source_teacher_cache_reconstruction_flops=int(reconstruction),
        teacher_and_old_target_score_flops=int(queries), same_query_teacher_read_target_flops=int(reads),
        total_model_preparation_flops=int(reconstruction + queries + reads),
        query_delta_matrix_bytes=delta_bytes, sampling_manifest_file_bytes=Path(manifest).stat().st_size,
        scope="standalone selected point, actual gradient fitting users plus validation16; input-view search reuse reported separately; matrix deltas preserve Q16 native FLOP geometry")


def run(args, *, preparation_cache=None):
    started = time.perf_counter()
    if args.controls or args.stages != "f,fg,fgh" or args.h_initialization != "standard":
        raise ValueError("supplementary new-term protocol supports one standard complete ladder, no extended controls")
    source_f = Path(args.source_f).resolve()
    if sha256(source_f) != args.source_f_sha256:
        raise ValueError("source f checkpoint changed")
    if args.h_initialization not in MODES:
        raise ValueError("single run requires standard or balanced h initialization")
    if (args.fit_users not in (64, 128, 256) or args.width not in (128, 256) or
            args.learning_rate not in (.00003, .00001) or args.steps not in (512, 1024, 2048) or
            args.patience not in (0, 128, 512, 1024) or args.validation_every != 8 or
            args.stages not in ("f", "f,fg,fgh")):
        raise ValueError("use only the explicitly explored E5 population/capacity/budget settings")
    if args.fit_users == 256 and args.patience and args.patience < 512:
        raise ValueError("fit256 must have at least512 steps of early-stop patience")
    if args.controls and args.stages != "f,fg,fgh":
        raise ValueError("sibling controls accompany the complete ladder")
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        if torch.cuda.mem_get_info(device)[0] < 32*1024**3:
            raise ValueError("require measured32GiB free for concurrent resident fitting")
    else:
        device = torch.device("cpu")
    root, output = Path(args.input_root), Path(args.output)
    directory = root / f"edge_{args.edge}"
    population_path, binding_path = root / "population.json", directory / "inputs.json"
    population, binding = [json.loads(path.read_text()) for path in (population_path, binding_path)]
    if binding["population_sha256"] != sha256(population_path):
        raise ValueError("calibration sampling population binding changed")
    for kind in ("history", "requests"):
        if sha256(directory / binding[kind]["path"]) != binding[kind]["sha256"]:
            raise ValueError(f"calibration sampling {kind} changed")
    cfg = dict(binding["settings"], calibration_queries_per_user=16, seed=17)
    if cfg["edge"] != args.edge:
        raise ValueError("calibration sampling edge differs")
    fit_uids, val_uids = population["fit"][:args.fit_users], population["validation"]
    if len(fit_uids) != args.fit_users or len(val_uids) != 16 or len(set(fit_uids + val_uids)) != args.fit_users + 16:
        raise ValueError("reserved fitting and validation roles differ")
    views, provenance = calibration_sampling.load_sampling_manifest(args.sampling_manifest, root, args.edge,
        args.recent_fraction, args.gap_rule, fit_users=args.fit_users)
    if (provenance["fit_uids"] != fit_uids or provenance["validation_uids"] != val_uids or
            provenance["catalog_source_users"] != 256 or provenance["time_law_source_users"] != 256 or
            provenance["source_cutoff"] != cfg["release_day"] * 86400):
        raise ValueError("candidate/time-law source and actual gradient roles differ")
    candidate_path = directory / "calibration_feedback_candidates.json"
    here = Path(__file__).resolve()
    source_paths = [here, Path(calibration_sampling.__file__),
        here.with_name("frozen_predecessor_fitting.py"), Path(resident_probe_fitting.__file__), *[
        here.with_name(name) for name in ("probe_initialization.py", "calibration_gaps.py", "common.py", "probes.py",
                                        "probe_variants.py", "prepare_candidates.py")],
        ROOT / "src/hstu_kvcache/adaptation/reader.py", ROOT / "src/hstu_kvcache/models/hstu.py",
        ROOT / "src/hstu_kvcache/models/embeddings.py", ROOT / "src/hstu_kvcache/evaluation/cost.py",
        ROOT / "src/hstu_kvcache/methods/baselines/protocol/cost.py"]
    sources = {str(path.resolve().relative_to(ROOT)): sha256(path) for path in source_paths}
    base_key = dict(edge=args.edge, device=str(device), physical_gpu=args.gpu,
        input_binding_sha256=sha256(binding_path), population_sha256=sha256(population_path),
        fit_uids=fit_uids, val_uids=val_uids, numerical_sources=sources)
    reused = preparation_cache is not None and preparation_cache.get("base_key") == base_key
    if reused:
        current, models, raw = preparation_cache["base"]
    else:
        parent, current, models = load_models(args.edge, device)
        histories = restore_pack(directory / binding["history"]["path"])
        raw = capture_samples(parent, current, histories, fit_uids + val_uids, cfg, device)
        for uid, sample in raw.items():
            sample["uid"] = int(uid)
        del parent, histories
        if preparation_cache is not None:
            preparation_cache.clear()
            preparation_cache.update(base_key=base_key, base=(current, models, raw))
    samples = calibration_sampling.apply_sampling_view(raw, views, provenance)
    for sample in samples.values():
        if sample["query_delta"].shape != (1, 16) or sample["candidates"].shape != (1, 16):
            raise ValueError("calibration sampling requires Q16 candidate/delta matrices")
    # No target, moment or normalization from a different candidate/time view
    # can survive apply_sampling_view; teacher targets are computed only now.
    train = prepare_training_samples(current, [samples[uid] for uid in fit_uids])
    validation = prepare_training_samples(current, [samples[uid] for uid in val_uids])
    prep = preparation_cost(current, train + validation, cfg, args.sampling_manifest)
    preparation_seconds = time.perf_counter() - started
    manifest_sha = sha256(Path(args.sampling_manifest))
    h_spec = initialization_spec(current.blocks[0].attn.head_dim,
        2*current.blocks[0].attn.head_dim+current.cfg.hidden_size, args.h_initialization)
    identity = dict(training_protocol=protocol_binding(), resident_inputs=True,
        source_f_checkpoint=str(source_f), source_f_sha256=args.source_f_sha256,
        h_initialization=args.h_initialization, h_initialization_spec=h_spec,
        initialization_helper_sha256=h_spec["helper_sha256"], variant="full_item", width=args.width, depth=1, fit_users=args.fit_users,
        queries=16, learning_rate=args.learning_rate, steps=args.steps, patience=args.patience,
        validation_every=8, read_weight=1., score_weight=1., candidate_source="feedback",
        recent_fraction=args.recent_fraction, candidate_seed=17, row_chunk=1024, query_chunk=16,
        weight_decay=.0001, gradient_clip=5., edge=args.edge, seed=args.seed, models=models,
        fit_uids=fit_uids, validation_users=val_uids, stages=args.stages, controls=args.controls,
        input_binding_sha256=sha256(binding_path), population_sha256=sha256(population_path),
        calibration_candidate_sha256=sha256(candidate_path), numerical_sources=sources,
        fitting_driver=here.name, fitting_driver_sha256=sha256(here), query_delta_rule=args.gap_rule,
        sampling_manifest=str(Path(args.sampling_manifest).resolve()), sampling_manifest_sha256=manifest_sha,
        gap_manifest=str(Path(args.sampling_manifest).resolve()), gap_manifest_sha256=manifest_sha,
        time_rule=provenance["time_rule"], source_cutoff=provenance["source_cutoff"],
        candidate_order=provenance["candidate_order"], query_delta_layout="[B,16]",
        candidate_catalog_source_users=256, time_law_source_users=256,
        gradient_fitting_users=args.fit_users, teacher_fitting_users=args.fit_users,
        fitting_uses_future_feedback=False)
    configuration = output / "configuration.json"
    if configuration.exists() and json.loads(configuration.read_text()) != identity:
        raise ValueError("calibration attempt identity changed; use a new attempt")
    write_json(configuration, identity)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    probe = build_variant("full_item", current, stage="f", hidden_width=args.width,
                          depth=1, query_chunk=16, row_chunk=1024).to(device)
    shared, total, summaries = prep["total_model_preparation_flops"], prep["total_model_preparation_flops"], []

    def fit_one(module, label, steps, previous_flops):
        checkpoint, summary_path = output / f"{label}.pt", output / f"{label}.json"
        if checkpoint.exists():
            # Loading constructs a CPU module before restoring its tensors.
            # Keep that temporary initialization from changing subsequent
            # g/h hidden-weight initialization after a partial-stage resume.
            with torch.random.fork_rng(devices=[]):
                saved, metadata = load_variant(checkpoint, device)
            if metadata["identity"] != identity or metadata["method_label"] != label or saved.stage != module.stage:
                raise ValueError("existing calibration checkpoint identity/branch differs")
            summary = dict(metadata, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
            if summary_path.exists() and json.loads(summary_path.read_text()) != summary:
                raise ValueError("existing calibration checkpoint summary changed")
            write_json(summary_path, summary)
            return saved, summary
        if label == "f":
            with torch.random.fork_rng(devices=[]):
                inherited, metadata = load_variant(source_f, device)
            expected_source = {key:value for key,value in identity.items() if key not in
                {"training_protocol","resident_inputs","source_f_checkpoint","source_f_sha256",
                 "fitting_driver","fitting_driver_sha256","numerical_sources"}}
            source_identity = metadata["identity"]
            actual_source = {key:value for key,value in source_identity.items() if key not in
                {"fitting_driver","fitting_driver_sha256","numerical_sources"}}
            if (actual_source != expected_source or inherited.stage != "f" or
                    inherited.cfg != module.cfg or metadata["method_label"] != "f" or
                    source_identity["fitting_driver"] != "run_probe_calibration_init.py" or
                    metadata["calibration_preparation"] != prep):
                raise ValueError("source f is not the exact same causal recipe/model/statistics/standalone preparation")
            for name, digest in source_identity["numerical_sources"].items():
                if sha256(ROOT/name) != digest:
                    raise ValueError("source f numerical provenance changed")
            metadata = dict(metadata, identity=identity,
                training_protocol=dict(binding=protocol_binding(), trainable_term="f",
                    inherited_source_only=True, source_checkpoint=str(source_f),
                    source_checkpoint_sha256=args.source_f_sha256,
                    scope="f-only fitting is identical in both protocols; reuse exact trained tensors and original standalone cost, not a new optimizer run"),
                actual_fitting_reused_this_invocation=True,
                actual_preparation_seconds_this_process=preparation_seconds)
            save_variant(checkpoint, inherited, metadata)
            summary = dict(metadata, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
            write_json(summary_path, summary)
            return inherited, summary
        fit_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        metadata = fit_probe(current, module, train, validation, steps=steps,
            learning_rate=args.learning_rate, read_weight=1., score_weight=1., seed=args.seed,
            patience=args.patience, validation_every=8,
            progress_callback=lambda row: print(json.dumps(dict(status="calibration_validation", edge=args.edge,
                recent_fraction=args.recent_fraction, query_delta_rule=args.gap_rule,
                fit_users=args.fit_users, width=args.width, method_label=label, **row)), flush=True))
        initialization_flops=(sum(network.weights[0].numel() for network in module.h)
            if label=="fgh" and args.h_initialization=="balanced" else 0)
        metadata.update(identity=identity, variant="full_item", method_label=label,
            sampling_provenance=provenance, calibration_preparation=prep,
            source_teacher_and_target_preparation_flops=int(shared),
            h_initialization_weight_rescale_flops=int(initialization_flops),
            cumulative_calibration_flops_estimate=int(previous_flops + metadata["total_calibration_flops_estimate"] + initialization_flops),
            actual_search_preparation_reused=reused,
            actual_preparation_seconds_this_process=preparation_seconds,
            actual_teacher_users_captured=len(fit_uids), actual_validation_users_captured=16,
            seconds=time.perf_counter() - fit_started, quality_auc_computed=False,
            fitting_uses_future_feedback=False)
        if device.type == "cuda":
            metadata.update(physical_gpu=args.gpu, hardware=torch.cuda.get_device_name(0),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save_variant(checkpoint, module, metadata)
        summary = dict(metadata, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
        write_json(summary_path, summary)
        return module, summary

    for stage in args.stages.split(","):
        if stage != probe.stage:
            probe = probe.expanded(stage)
            if stage == "fgh":
                applied = apply_h_initialization(probe, args.h_initialization)
                if any(applied[key] != value for key,value in h_spec.items()):
                    raise ValueError("h initialization differs from the declared identity")
        probe, summary = fit_one(probe, stage, args.steps, total)
        total = summary["cumulative_calibration_flops_estimate"]
        summaries.append(summary)
        if args.controls and stage in ("f", "fg"):
            _, control = fit_one(copy.deepcopy(probe), stage + "_extended",
                                 2 * args.steps if stage == "f" else args.steps, total)
            summaries.append(control)
    for name, digest in sources.items():
        if sha256(ROOT / name) != digest:
            raise ValueError("calibration numerical source changed during fitting")
    if sha256(Path(args.sampling_manifest)) != manifest_sha:
        raise ValueError("sampling manifest changed during fitting")
    result = dict(status="fitted", identity=identity, stages=summaries,
        calibration_preparation=prep, sampling_provenance=provenance,
        scope="supplementary new-term-only protocol; exact source f inherited, own FG precedes own FGH; conservative full-backward estimate retained; no population quality or future labels")
    write_json(output / "summary.json", result)
    print(json.dumps(dict(status="complete", edge=args.edge, output=str(output),
                         fit_users=args.fit_users, recent_fraction=args.recent_fraction,
                         query_delta_rule=args.gap_rule)), flush=True)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input-root", type=Path, default=ROOT / "experiments/ch3_motivation/inputs_feedback_pool/fit256")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--source-f", type=Path, required=True)
    result.add_argument("--source-f-sha256", required=True)
    result.add_argument("--sampling-manifest", type=Path, required=True)
    result.add_argument("--edge", type=int, choices=range(1, 6), required=True)
    result.add_argument("--gap-rule", choices=("cutover", "gap_mix50", "gap_empirical100"), required=True)
    result.add_argument("--recent-fraction", type=float, choices=(.5, .75), required=True)
    result.add_argument("--fit-users", type=int, choices=(64, 128, 256), required=True)
    result.add_argument("--width", type=int, choices=(128, 256), default=128)
    result.add_argument("--learning-rate", type=float, choices=(.00003, .00001), default=.00003)
    result.add_argument("--steps", type=int, choices=(512, 1024, 2048), default=2048)
    result.add_argument("--patience", type=int, choices=(0, 128, 512, 1024), default=1024)
    result.add_argument("--validation-every", type=int, choices=(8,), default=8)
    result.add_argument("--stages", choices=("f", "f,fg,fgh"), default="f,fg,fgh")
    result.add_argument("--controls", action=argparse.BooleanOptionalAction, default=False)
    result.add_argument("--h-initialization", choices=("standard",), default="standard")
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--gpu", type=int, default=0)
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return result


if __name__ == "__main__":
    args=parser().parse_args()
    run(args)
