#!/usr/bin/env python3
"""Bounded Large/edge2 input-residency oracle; never a quality/fitting run.

GPU execution is explicit.  The six cases use one CPU capture/target pool,
unchanged core fitting, and independently cloned initial probes.  Gradient
hooks are confined to the numerical pass; every timed resident call includes
its own stage-local placement.  No checkpoint or AUC is produced.
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
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


import torch

from experiments.ch3_motivation.code import calibration_sampling, probes, resident_probe_fitting
from experiments.ch3_motivation.code.common import (
    ROOT, capture_samples, load_models, restore_pack, write_json,
)
from experiments.ch3_motivation.code.probe_variants import build_variant
from hstu_kvcache.common.artifacts import sha256

HELPER_SHA = "3847f3bdafbd33fe4b16ade7ad9305787d2c7b88b871e473cb1922e90e1e011b"
CORE_SHA = "6ab7246f3f4d2d2867ca3e69f2db85f8513b1ff6ae3d3983cf5c76aac2ede934"
WIDTHS = (128, 256)
STAGES = ("f", "fg", "fgh")
FIT_KWARGS = dict(learning_rate=3e-5, read_weight=1., score_weight=1., seed=17,
                  patience=0, validation_every=8, weight_decay=1e-4, gradient_clip=5.)
DIAGNOSTIC = dict(diagnostic_only=True, quality_auc_computed=False,
                  quality_use_permitted=False, formal_weights_written=False,
                  long_run_reliability_established=False)


def sources():
    paths = [Path(__file__), Path(resident_probe_fitting.__file__), Path(probes.__file__),
             *[Path(__file__).with_name(name) for name in
               ("common.py", "probe_variants.py", "calibration_sampling.py", "calibration_gaps.py",
                "prepare_candidates.py")],
             ROOT / "src/hstu_kvcache/adaptation/reader.py",
             *[ROOT / "src/hstu_kvcache/models" / name for name in
               ("hstu.py", "embeddings.py", "attention.py", "block.py", "rmsnorm.py", "kv_cache.py",
                "backend_info.py", "triton_attention.py")],
             ROOT / "src/hstu_kvcache/evaluation/trajectory.py",
             ROOT / "src/hstu_kvcache/methods/baselines/layer_recompute/core.py",
             ROOT / "src/hstu_kvcache/evaluation/history.py",
             ROOT / "src/hstu_kvcache/evaluation/baselines.py",
             ROOT / "src/hstu_kvcache/methods/read_correction/calibration/v2/calibrate.py",
             ROOT / "src/hstu_kvcache/evaluation/yambda.py",
             ROOT / "src/hstu_kvcache/common/artifacts.py"]
    return {str(path.resolve()): sha256(path) for path in paths}


def check_frozen_fitter():
    if sha256(Path(resident_probe_fitting.__file__)) != HELPER_SHA or sha256(Path(probes.__file__)) != CORE_SHA:
        raise ValueError("resident/reference source differs from the reviewed diagnostic binding")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory(device):
    if device.type != "cuda":
        return None
    free, total = torch.cuda.mem_get_info(device)
    return dict(free_bytes=free, total_bytes=total,
                allocated_bytes=torch.cuda.memory_allocated(device),
                reserved_bytes=torch.cuda.memory_reserved(device))


def rng_fingerprint(device):
    result = dict(cpu=hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest())
    if device.type == "cuda":
        result["cuda"] = hashlib.sha256(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).hexdigest()
    return result


def tensor_difference(expected, actual, *, rtol=1e-5, atol=1e-7):
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return dict(passed=False, reason="shape/dtype differs",
                    expected_shape=list(expected.shape), actual_shape=list(actual.shape))
    left, right = expected.detach().cpu(), actual.detach().cpu()
    if not left.is_floating_point() and not left.is_complex():
        return dict(passed=torch.equal(left, right), max_abs=float((left.double()-right.double()).abs().max()))
    error = right.double()-left.double()
    relative = float(error.square().mean().sqrt() / left.double().square().mean().sqrt().clamp_min(1e-30))
    return dict(passed=bool(torch.allclose(left, right, rtol=rtol, atol=atol)),
                max_abs=float(error.abs().max()), relative_rms=relative)


def compare_states(expected, actual, *, rtol, atol):
    if expected.keys() != actual.keys():
        return dict(passed=False, reason="state/gradient parameter keys differ")
    rows = {name: tensor_difference(value, actual[name], rtol=rtol, atol=atol)
            for name, value in expected.items()}
    return dict(passed=all(row["passed"] for row in rows.values()), tensors=rows)


def compare_reports(expected, actual, *, rtol, atol):
    """Compare every original field; only the adapter's extra report is omitted."""
    actual = {key: value for key, value in actual.items() if key != "execution_adapter"}
    failures, maximum = [], 0.

    def visit(left, right, path):
        nonlocal maximum
        if isinstance(left, dict):
            if not isinstance(right, dict) or left.keys() != right.keys():
                failures.append(path + ": keys differ")
                return
            for key, value in left.items():
                visit(value, right[key], f"{path}.{key}")
        elif isinstance(left, list):
            if not isinstance(right, list) or len(left) != len(right):
                failures.append(path + ": lengths differ")
                return
            for index, value in enumerate(left):
                visit(value, right[index], f"{path}[{index}]")
        elif isinstance(left, float):
            if not isinstance(right, (int, float)) or not math.isfinite(right):
                failures.append(path + ": nonfinite/type differs")
            else:
                maximum = max(maximum, abs(left-right))
                if not math.isclose(left, right, rel_tol=rtol, abs_tol=atol):
                    failures.append(path + ": float differs")
        elif left != right:
            failures.append(path + ": value differs")

    visit(expected, actual, "report")
    # Counters/FLOPs are integers, so they must match exactly, including cost
    # estimates that will later receive the existing calibration correction.
    exact = all(expected[key] == actual.get(key) for key in
                ("best_step", "completed_steps", "operation_counts", "arithmetic",
                 "total_calibration_flops_estimate"))
    return dict(passed=not failures and exact, failures=failures, max_float_abs=maximum,
                counters_arithmetic_and_best_step_exact=exact)


def attach_gradient_trace(probe):
    """Pre-clipping gradients, indexed by parameter and backward invocation."""
    trace = {name: [] for name, parameter in probe.named_parameters() if parameter.requires_grad}
    handles = [parameter.register_hook(lambda gradient, name=name:
        trace[name].append(gradient.detach().cpu().clone()))
        for name, parameter in probe.named_parameters() if parameter.requires_grad]
    return trace, handles


def compare_gradients(expected, actual, steps, *, rtol, atol):
    if expected.keys() != actual.keys():
        return dict(passed=False, reason="gradient parameter keys differ")
    rows = {}
    for name, values in expected.items():
        other = actual[name]
        complete = len(values) == len(other) == steps
        comparisons = [tensor_difference(left, right, rtol=rtol, atol=atol)
                       for left, right in zip(values, other)]
        rows[name] = dict(passed=complete and all(row["passed"] for row in comparisons),
                          reference_steps=len(values), resident_steps=len(other), steps=comparisons)
    return dict(passed=all(row["passed"] for row in rows.values()),
                scope="raw gradients before gradient clipping, every trained parameter at every step",
                parameters=rows)


def numeric_pair(model, initial, train, validation, *, steps=4, rtol=1e-5, atol=1e-7):
    """Actual optimizer oracle, returning the reference best state for expansion."""
    device = next(model.parameters()).device
    reports, gradients, states, rng, reference_best = {}, {}, {}, {}, None
    for mode, fitter in (("reference", probes.fit_probe),
                         ("resident", resident_probe_fitting.fit_probe_resident)):
        probe = copy.deepcopy(initial).to(device)
        trace, handles = attach_gradient_trace(probe)
        before = rng_fingerprint(device)
        try:
            reports[mode] = fitter(model, probe, train, validation, steps=steps, **FIT_KWARGS)
            synchronize(device)
        finally:
            for handle in handles:
                handle.remove()
        rng[mode] = dict(before=before, after=rng_fingerprint(device))
        gradients[mode] = trace
        probe = probe.cpu()
        states[mode] = {name: value.detach().clone() for name, value in probe.state_dict().items()}
        if mode == "reference":
            reference_best = probe
        else:
            del probe
    results = dict(gradients=compare_gradients(gradients["reference"], gradients["resident"], steps,
                                             rtol=rtol, atol=atol),
                   final_state=compare_states(states["reference"], states["resident"], rtol=rtol, atol=atol),
                   original_report=compare_reports(reports["reference"], reports["resident"], rtol=rtol, atol=atol),
                   rng=rng)
    results["rng_unchanged"] = all(row["before"] == row["after"] for row in rng.values())
    results["passed"] = results["rng_unchanged"] and all(results[key]["passed"] for key in
                                                       ("gradients", "final_state", "original_report"))
    results["reference_fit"] = reports["reference"]
    results["resident_execution_adapter"] = reports["resident"]["execution_adapter"]
    results["tolerances"] = dict(rtol=rtol, atol=atol)
    return reference_best, results


def measured_fit(model, initial, train, validation, mode, *, steps):
    """No gradient hooks; wall encloses fitting and resident preplacement."""
    device = next(model.parameters()).device
    gc.collect()
    synchronize(device)
    clone_started = time.perf_counter()
    probe = copy.deepcopy(initial).to(device)
    synchronize(device)
    clone_seconds = time.perf_counter()-clone_started
    if any(parameter._backward_hooks for parameter in probe.parameters()):
        raise ValueError("timing probe unexpectedly contains gradient hooks")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    before = memory(device)
    started = time.perf_counter()
    fitter = probes.fit_probe if mode == "reference" else resident_probe_fitting.fit_probe_resident
    report = fitter(model, probe, train, validation, steps=steps, **FIT_KWARGS)
    synchronize(device)
    elapsed = time.perf_counter()-started
    after = memory(device)
    peak = (dict(allocated_bytes=torch.cuda.max_memory_allocated(device),
                 reserved_bytes=torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None)
    state = {name: value.detach().cpu().clone() for name, value in probe.state_dict().items()}
    del probe
    gc.collect()
    synchronize(device)
    return dict(mode=mode, wall_seconds=elapsed, clone_to_device_seconds_separate=clone_seconds,
                memory_before=before, memory_after_fit=after, peak_memory=peak,
                memory_after_probe_release=memory(device), fit=report), state


def timing_pairs(model, initial, train, validation, *, steps=32, repeats=2, rtol=1e-5, atol=1e-7):
    rows = []
    for repeat in range(repeats):
        order = ("reference", "resident") if repeat % 2 == 0 else ("resident", "reference")
        measured, states = {}, {}
        for mode in order:
            measured[mode], states[mode] = measured_fit(model, initial, train, validation, mode, steps=steps)
        check = dict(original_report=compare_reports(measured["reference"]["fit"], measured["resident"]["fit"],
                                                      rtol=rtol, atol=atol),
                     final_state=compare_states(states["reference"], states["resident"], rtol=rtol, atol=atol))
        passed = all(row["passed"] for row in check.values())
        rows.append(dict(repeat=repeat+1, order=list(order),
                         phase="first_short_replay" if repeat == 0 else "subsequent_kernel_warm_short_replay",
                         runs=measured, comparison=check, passed=passed,
                         reference_over_resident_wall=measured["reference"]["wall_seconds"] /
                                                      measured["resident"]["wall_seconds"]))
        if not passed:
            break
    return dict(passed=len(rows) == repeats and all(row["passed"] for row in rows),
                steps=steps, repeats_requested=repeats, matched_replays=rows,
                scope="no gradient hooks; identical initial state/seed/order/kwargs; every resident call includes prefetch; first timed replay follows numeric fitting, so kernels are already warm; allocator retained between calls; first versus subsequent short replay is not a cold-start or long-run speed guarantee")


def prepare_inputs(args, device):
    """The real diagnostic has exactly one declared Large/edge2 fit256 view."""
    started = time.perf_counter()
    root = Path(args.input_root).resolve()
    directory = root / "edge_2"
    population_path, binding_path = root / "population.json", directory / "inputs.json"
    population, binding = [json.loads(path.read_text()) for path in (population_path, binding_path)]
    if binding["population_sha256"] != sha256(population_path):
        raise ValueError("resident diagnostic population binding changed")
    inputs = {str(population_path): sha256(population_path), str(binding_path): sha256(binding_path)}
    for kind in ("history", "requests"):
        path = directory / binding[kind]["path"]
        digest = sha256(path)
        if digest != binding[kind]["sha256"]:
            raise ValueError(f"resident diagnostic {kind} changed")
        inputs[str(path.resolve())] = digest
    cfg = dict(binding["settings"], calibration_queries_per_user=16, seed=17)
    fit, validation = population["fit"][:256], population["validation"]
    if (cfg["edge"] != 2 or cfg["history_length"] != 1024 or len(fit) != 256 or len(validation) != 16
            or len(set(fit+validation)) != 272):
        raise ValueError("resident diagnostic requires frozen edge2 fit256 + validation16, window1024")
    views, provenance = calibration_sampling.load_sampling_manifest(args.sampling_manifest, root, 2,
        .5, "gap_empirical100", fit_users=256)
    if (provenance["fit_uids"] != fit or provenance["validation_uids"] != validation or
            provenance["catalog_source_users"] != 256 or provenance["time_law_source_users"] != 256 or
            provenance["source_cutoff"] != cfg["release_day"]*86400):
        raise ValueError("resident diagnostic sampling role/source/cutoff differs")
    inputs[str(Path(args.sampling_manifest).resolve())] = sha256(Path(args.sampling_manifest))
    candidate_path = directory / "calibration_feedback_candidates.json"
    inputs[str(candidate_path)] = sha256(candidate_path)
    manifest_finished = time.perf_counter()
    parent, model, model_hashes = load_models(2, device, backend=args.backend)
    synchronize(device)
    model_finished = time.perf_counter()
    if (model.cfg.hidden_size != 320 or model.cfg.num_heads != 10 or len(model.blocks) != 10
            or model.blocks[0].attn.head_dim != 32):
        raise ValueError("readiness requires actual Large geometry, not a tiny substitute")
    histories = restore_pack(directory / binding["history"]["path"])
    raw = capture_samples(parent, model, histories, fit+validation, cfg, device)
    for uid, sample in raw.items():
        sample["uid"] = int(uid)
    synchronize(device)
    capture_finished = time.perf_counter()
    del parent, histories
    samples = calibration_sampling.apply_sampling_view(raw, views, provenance)
    if any(sample["query_delta"].shape != (1, 16) or sample["candidates"].shape != (1, 16)
           for sample in samples.values()):
        raise ValueError("readiness requires Q16 candidate/time matrices")
    train = probes.prepare_training_samples(model, [samples[uid] for uid in fit])
    val = probes.prepare_training_samples(model, [samples[uid] for uid in validation])
    synchronize(device)
    finished = time.perf_counter()
    lengths = [sample["parent_cache"].seq_len for sample in train+val]
    if any(tensor.device.type != "cpu" for sample in train+val for tensor in
           (sample["parent_cache"].k, sample["teacher_cache"].k, sample["candidates"], sample["query_delta"])):
        raise ValueError("prepared diagnostic inputs must start on CPU")
    record = dict(edge=2, fit_users=256, validation_users=16, fit_uids=fit, validation_uids=validation,
        recent_fraction=.5, query_delta_rule="gap_empirical100", input_root=str(root),
        sampling_manifest=str(Path(args.sampling_manifest).resolve()), sampling_provenance=provenance,
        input_sha256=inputs, models=model_hashes, model_config=asdict(model.cfg),
        history_lengths=dict(min=min(lengths), max=max(lengths), total=sum(lengths),
                             full_window_users=sum(length == 1024 for length in lengths)),
        shared_preparation_seconds=dict(input_verification=manifest_finished-started,
            model_load=model_finished-manifest_finished, capture=capture_finished-model_finished,
            sampled_view_and_teacher_targets=finished-capture_finished, total=finished-started),
        preparation_scope="one parent/current load and one CPU cache capture for fit256+val16; sampling view applied before one target preparation; no requests/labels/scores loaded, no f/readfeatures/student-query teacher-read cache",
        core_statistics_scope="each cloned f fitter performs its own unchanged fit256 input-statistics pass; fg/fgh inherit nested reference statistics; timing includes this work")
    return model, train, val, record


def run(args):
    if not 2 <= args.numeric_steps <= 8 or args.timing_steps != 32 or args.repeats < 2:
        raise ValueError("use 2–8 numerical steps and at least two matched 32-step timing replays")
    check_frozen_fitter()
    output = Path(args.output).resolve()
    if (output / "summary.json").exists():
        raise ValueError("diagnostic output already exists; use a new attempt")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    before_sources = sources()
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(format="resident_probe_fitting_readiness_v1", status="preparing", **DIAGNOSTIC,
        source_sha256=before_sources, physical_gpu=str(args.gpu),
        hardware=dict(name=torch.cuda.get_device_name(device), torch_version=torch.__version__,
                      cuda_version=torch.version.cuda, initial_memory=memory(device)),
        settings=dict(widths=list(WIDTHS), stages=list(STAGES), numeric_steps=args.numeric_steps,
                      timing_steps=32, repeats=args.repeats, rtol=args.rtol, atol=args.atol,
                      fit_kwargs=FIT_KWARGS, reconstruction_backend=args.backend),
        memory_interpretation="allocator is retained across alternating rounds; peak_reserved can inherit an earlier high-water mark and is not resident incremental memory; use peak_allocated, allocation snapshots and explicit preplacement storage/copy bytes",
        cases=[])
    write_json(output / "summary.json", summary)
    try:
        model, train, validation, preparation = prepare_inputs(args, device)
        summary.update(status="checking", preparation=preparation)
        write_json(output / "summary.json", summary)
        for width in WIDTHS:
            torch.manual_seed(17)
            initial = build_variant("full_item", model, stage="f", hidden_width=width, depth=1,
                                    query_chunk=16, row_chunk=1024)
            for stage in STAGES:
                if initial.stage != stage:
                    raise AssertionError("nested diagnostic stage differs")
                summary["active_case"] = dict(width=width, stage=stage, phase="numeric_with_hooks")
                write_json(output / "summary.json", summary)
                best, numeric = numeric_pair(model, initial, train, validation, steps=args.numeric_steps,
                                             rtol=args.rtol, atol=args.atol)
                case = dict(width=width, stage=stage, numeric=numeric)
                summary["cases"].append(case)
                write_json(output / "summary.json", summary)
                if not numeric["passed"]:
                    raise AssertionError(f"numeric mismatch in width{width}/{stage}; retained in summary")
                summary["active_case"]["phase"] = "timing_without_hooks"
                case["timing"] = timing_pairs(model, initial, train, validation, steps=32, repeats=args.repeats,
                                               rtol=args.rtol, atol=args.atol)
                write_json(output / "summary.json", summary)
                if not case["timing"]["passed"]:
                    raise AssertionError(f"timed replay mismatch in width{width}/{stage}; retained in summary")
                print(json.dumps(dict(status="case_complete", width=width, stage=stage,
                    best_step=numeric["reference_fit"]["best_step"],
                    speed_ratios=[row["reference_over_resident_wall"] for row in case["timing"]["matched_replays"]])), flush=True)
                # Preserve the real nested f->fg->fgh initialization path. The
                # timed probes never supply weights to the next stage.
                if stage != "fgh":
                    initial = best.expanded("fg" if stage == "f" else "fgh")
                del best
        if sources() != before_sources:
            raise ValueError("readiness source changed during execution")
        for path, digest in summary["preparation"]["input_sha256"].items():
            if sha256(Path(path)) != digest:
                raise ValueError("readiness input changed during execution")
        summary["status"] = "completed_numeric_and_bounded_timing"
        summary.pop("active_case")
    except Exception as error:
        summary.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
        write_json(output / "summary.json", summary)
        raise
    write_json(output / "summary.json", summary)
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--input-root", type=Path,
                        default=ROOT / "experiments/ch3_motivation/inputs_feedback_pool/fit256")
    result.add_argument("--sampling-manifest", type=Path, default=ROOT /
        "experiments/ch3_motivation/runs/e5/calibration_sampling_fixed256/attempt_1/edge_2/sampling_manifest.json")
    result.add_argument("--backend", choices=("torch", "triton"), default="triton")
    result.add_argument("--numeric-steps", type=int, default=4)
    result.add_argument("--timing-steps", type=int, default=32)
    result.add_argument("--repeats", type=int, default=2)
    result.add_argument("--rtol", type=float, default=1e-5)
    result.add_argument("--atol", type=float, default=1e-7)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
