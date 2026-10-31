#!/usr/bin/env python3
"""Fit the native-source half of a matched source-state calibration study.

The actual-source refresh already exists.  This control uses the SAME cutover,
users, candidates, objective and budget, but its input is a native rebuild by
the old target.  No previous F is applied to its fitting source.
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
from experiments.ch3_motivation.code.fit_kv_version_refresh import _load, _model_path, preparation_cost
from experiments.ch3_motivation.code.prepare_candidates import candidates_for
from experiments.ch3_motivation.code.probe_variants import build_variant, load_variant, save_variant
from experiments.ch3_motivation.code.probes import prepare_training_samples
from experiments.ch3_motivation.code.query_centering import fit_query_conditioned
from hstu_kvcache.common.artifacts import sha256
import torch


@torch.no_grad()
def capture_native_sources(old_target, new_target, histories, uids, cfg, device):
    """Capture native old-target inputs and new-target teachers on one prefix."""
    return capture_samples(old_target, new_target, histories, uids, cfg, device)


def run(args):
    started = time.perf_counter()
    if args.gpu != 1 or args.target_version != args.history_edge + 1:
        raise ValueError("this matched study uses GPU1 and the next target version")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["EVOKV_ATTENTION_BACKEND"] = "triton"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    reference, actual = load_variant(args.actual_refresh_checkpoint, "cpu")
    paired = actual["identity"]
    expected = dict(protocol="frozen_mapped_source_version_refresh_v1",
        source_kind="actual_predecessor_mapped_cache_not_native_old_target",
        history_edge=args.history_edge, target_version=args.target_version,
        fit_users=100, variant="full_item", width=64, depth=1, queries=16, seed=17,
        candidate_seed=17, candidate_source="feedback", recent_fraction=.75,
        steps=256, learning_rate=1e-4, read_weight=1., score_weight=1.,
        validation_every=32, patience=0, query_conditioning_mode="center_std")
    if (reference.stage != "f" or actual.get("method_label") != "refresh_map"
            or any(paired.get(key) != value for key, value in expected.items())):
        raise ValueError("reference must be the matched actual-source refresh recipe")
    del reference
    input_root = Path(paired["input_root"])
    population_path = input_root / "population.json"
    population = json.loads(population_path.read_text())
    directory = input_root / f"edge_{args.history_edge}"
    binding_path = directory / "inputs.json"
    binding = json.loads(binding_path.read_text())
    if (sha256(population_path) != paired["population_sha256"]
            or sha256(binding_path) != paired["input_binding_sha256"]
            or binding["population_sha256"] != sha256(population_path)):
        raise ValueError("matched fitting input or population binding changed")
    for name in ("history", "requests"):
        if sha256(directory / binding[name]["path"]) != binding[name]["sha256"]:
            raise ValueError("frozen fitting input pack changed")
    fit_uids, validation_uids = population["fit"][:100], population["validation"]
    if (fit_uids != paired["fit_uids"] or validation_uids != paired["validation_users"]
            or len(fit_uids) != 100 or len(validation_uids) != 16
            or len(set(fit_uids + validation_uids)) != 116):
        raise ValueError("native and actual controls must use identical fit100/validation16")
    cfg = dict(binding["settings"], calibration_queries_per_user=16, seed=17)
    if cfg["release_day"] * 86400 != paired["history_cutoff"]:
        raise ValueError("matched original history cutoff changed")
    candidate_path = directory / "calibration_feedback_candidates.json"
    candidate_data = json.loads(candidate_path.read_text())
    if (sha256(candidate_path) != paired["calibration_candidate_sha256"]
            or candidate_data["population_sha256"] != sha256(population_path)):
        raise ValueError("matched candidate source changed")
    used_models = {str(_model_path(v).relative_to(ROOT)): sha256(_model_path(v))
                   for v in (args.history_edge, args.target_version)}
    if any(paired["models"].get(path) != digest for path, digest in used_models.items()):
        raise ValueError("native and actual source studies use different target backbones")
    paths = [Path(__file__), *[Path(__file__).with_name(name) for name in
        ("common.py", "fit_kv_version_refresh.py", "prepare_candidates.py", "probe_cost.py",
         "probes.py", "probe_variants.py", "query_centering.py", "resident_probe_fitting.py",
         "probe_first_affine.py")], ROOT / "src/hstu_kvcache/adaptation/reader.py",
         ROOT / "src/hstu_kvcache/evaluation/yambda.py",
         ROOT / "src/hstu_kvcache/evaluation/cost.py"]
    sources = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    identity = {key: paired[key] for key in expected if key not in ("protocol", "source_kind")}
    identity.update(protocol="frozen_native_source_refresh_control_v1",
        source_kind="native_old_target_same_history", native_source_version=args.history_edge,
        history_cutoff=paired["history_cutoff"], input_root=str(input_root),
        input_binding_sha256=sha256(binding_path), population_sha256=sha256(population_path),
        fit_uids=fit_uids, validation_users=validation_uids,
        teacher_fitting_users=100, gradient_fitting_users=100,
        query_delta_rule=paired["query_delta_rule"],
        calibration_candidate_sha256=sha256(candidate_path), models=used_models,
        paired_actual_refresh_checkpoint=str(args.actual_refresh_checkpoint.resolve()),
        paired_actual_refresh_checkpoint_sha256=sha256(args.actual_refresh_checkpoint),
        paired_predecessor_checkpoint_sha256=paired["predecessor_checkpoint_sha256"],
        normalization_scope="fit-only moments recomputed on native-source inputs using the same fitter algorithm",
        numerical_sources=sources, fitting_driver=Path(__file__).name,
        quality_auc_computed=False,
        scope="Native old-target source control for the same frozen-history actual-source refresh case.")
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != identity:
        raise ValueError("native-source control configuration changed; use another output")
    write_json(config_path, identity)
    checkpoint = args.output / "native_refresh_control.pt"
    if checkpoint.exists():
        saved, metadata = load_variant(checkpoint, "cpu")
        if (saved.stage != "f" or metadata["identity"] != identity
                or metadata.get("method_label") != "native_refresh_control"):
            raise ValueError("saved native-source control identity changed")
        print(json.dumps(dict(status="already_complete", checkpoint=str(checkpoint))), flush=True)
        return metadata
    old_target, current = _load(args.history_edge, device), _load(args.target_version, device)
    if (old_target.cfg.num_items != current.cfg.num_items
            or candidate_data["known_items"] != current.cfg.num_items - 256):
        raise ValueError("native control requires the same fixed item catalog")
    histories = restore_pack(directory / binding["history"]["path"])
    samples = capture_native_sources(old_target, current, histories,
        fit_uids + validation_uids, cfg, device)
    del old_target, histories
    torch.cuda.empty_cache()
    for uid, sample in samples.items():
        sample["candidates"] = torch.as_tensor(candidates_for(uid,
            candidate_data["recent_items"][str(uid)], candidate_data["known_items"],
            count=16, recent_fraction=.75, seed=17), dtype=torch.long)[None]
    train = prepare_training_samples(current, [samples[uid] for uid in fit_uids])
    validation = prepare_training_samples(current, [samples[uid] for uid in validation_uids])
    preparation = preparation_cost(current, train + validation, cfg)
    shared = preparation["total_model_preparation_flops"]
    preparation_seconds = time.perf_counter() - started
    del samples
    torch.manual_seed(17)
    probe = build_variant("full_item", current, stage="f", hidden_width=64, depth=1,
                          query_chunk=16, row_chunk=1024).to(device)
    fitting_started = time.perf_counter()
    metadata = fit_query_conditioned(current, probe, train, validation,
        mode="center_std", factorized_h=True, steps=256, learning_rate=1e-4,
        read_weight=1., score_weight=1., seed=17, patience=0, validation_every=32,
        progress_callback=lambda row: print(json.dumps(dict(status="native_refresh_validation",
            history_edge=args.history_edge, target_version=args.target_version, **row)), flush=True))
    metadata.update(identity=identity, variant="full_item", method_label="native_refresh_control",
        preparation=preparation, source_teacher_and_target_preparation_flops=int(shared),
        cumulative_calibration_flops_estimate=int(shared + metadata["total_calibration_flops_estimate"]),
        preparation_seconds=preparation_seconds, fitting_seconds=time.perf_counter()-fitting_started,
        seconds=time.perf_counter()-started, physical_gpu=1,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        quality_auc_computed=False,
        cost_scope="Native old-target/new-teacher preparation and fitting; no predecessor materialization. Actual-source control costs are retained separately, not shared or subtracted.")
    if any(sha256(ROOT / path) != digest for path, digest in sources.items()):
        raise ValueError("native-control numerical source changed during fitting")
    save_variant(checkpoint, probe, metadata)
    summary = dict(metadata, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), status="fitted")
    write_json(args.output / "native_refresh_control.json", summary)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(dict(status="complete", checkpoint=str(checkpoint),
        history_edge=args.history_edge, target_version=args.target_version)), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-edge", type=int, choices=(2, 3), required=True)
    parser.add_argument("--target-version", type=int, choices=(3, 4), required=True)
    parser.add_argument("--actual-refresh-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    run(parser.parse_args())
