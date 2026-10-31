#!/usr/bin/env python3
"""Re-evaluate old/new E5 weights on the fixed fit64-ID, three-gap problem.

One edge uses 64 distinct fitting users crossed with three contexts to estimate
one set of loss denominators. The same 16 reserved validation users are then
evaluated in each context. No probe fitting or population quality is performed.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import sys
import time

import torch
from experiments.ch3_motivation.code import calibration_gaps
from experiments.ch3_motivation.code.common import (
    ROOT, capture_samples, load_models, restore_pack, write_json,
)
from experiments.ch3_motivation.code.prepare_candidates import candidates_for
from experiments.ch3_motivation.code.prepare_temporal_common_plan import RULES
from experiments.ch3_motivation.code.probes import bind_fit_scales
from experiments.ch3_motivation.code.probe_variants import load_variant
from experiments.ch3_motivation.code.select_probe_common import sha256
from experiments.ch3_motivation.code.validate_probe_common import (
    CommonValidation, _check_weight_binding, _digest, _provenance, evaluate_common_probe, native_baseline,
)

FORMAT = "ch3_temporal_common_pre_release_validation_v1"


def _sources():
    here = Path(__file__).resolve()
    return [here, *[here.with_name(name) for name in (
        "validate_probe_common.py", "prepare_temporal_common_plan.py", "calibration_gaps.py",
        "common.py", "probes.py", "probe_variants.py", "prepare_candidates.py")],
        ROOT / "src/hstu_kvcache/adaptation/reader.py"]


@torch.no_grad()
def prepare_contexts(input_root, edge, device, gap_manifest, *, backend="torch"):
    """Capture once; install exact deltas before target preparation; pool fit RMS."""
    started = time.perf_counter()
    input_root, gap_manifest = Path(input_root), Path(gap_manifest)
    directory = input_root / f"edge_{edge}"
    population_path, binding_path = input_root / "population.json", directory / "inputs.json"
    candidate_path = directory / "calibration_feedback_candidates.json"
    population, binding, candidates = [json.loads(p.read_text()) for p in
                                      (population_path, binding_path, candidate_path)]
    fit_uids, val_uids = population["fit"][:64], population["validation"]
    population_sha = sha256(population_path)
    cfg = dict(binding["settings"], calibration_queries_per_user=16, seed=17, attention_backend=backend)
    if (len(fit_uids) != 64 or len(val_uids) != 16 or len(set(fit_uids + val_uids)) != 80 or
            cfg["edge"] != edge or binding["population_sha256"] != population_sha or
            candidates["population_sha256"] != population_sha or
            candidates["fitting_pool_uids"] != fit_uids or candidates["validation_uids"] != val_uids):
        raise ValueError("new temporal common validation requires the exact fit64 candidate/UID binding")
    frozen = {str(p.resolve()): sha256(p) for p in
              (population_path, binding_path, candidate_path, gap_manifest, *_sources())}
    for name in ("history", "requests"):
        path = directory / binding[name]["path"]
        if sha256(path) != binding[name]["sha256"]:
            raise ValueError(f"temporal common {name} pack changed")
        frozen[str(path.resolve())] = binding[name]["sha256"]
    source = Path(candidates["source_candidate_path"])
    if sha256(source) != candidates["source_candidate_sha256"]:
        raise ValueError("source feedback candidate file changed")
    frozen[str(source.resolve())] = candidates["source_candidate_sha256"]
    views, provenance = {}, {}
    for rule in RULES:
        gaps, prov = calibration_gaps.load_gap_manifest(gap_manifest, input_root, edge, rule)
        if (prov["fit_uids"] != fit_uids or prov["validation_uids"] != val_uids or
                prov["source_cutoff"] != cfg["release_day"] * 86400 or prov["fallback_used"] or
                prov["rule"] != rule or prov["manifest_sha256"] != sha256(gap_manifest)):
            raise ValueError("temporal common law must be the same no-fallback fit64 pre-release law")
        views[rule], provenance[rule] = gaps, prov
    parent, model, models = load_models(edge, device, backend=backend)
    known = model.cfg.num_items - 256
    if known != candidates["known_items"]:
        raise ValueError("temporal common candidate catalog differs from backbone")
    histories = restore_pack(directory / binding["history"]["path"])
    samples = capture_samples(parent, model, histories, fit_uids + val_uids, cfg, device)
    for uid, sample in samples.items():
        sample["uid"] = int(uid)
        sample["candidates"] = torch.tensor([candidates_for(uid,
            candidates["recent_items"][str(uid)], known, count=16, recent_fraction=.5, seed=17)])
    # Each mode shallow-shares immutable CPU caches, but gets its own delta and
    # candidate-dependent moments. No old target or per-user scale survives.
    modes = {rule: calibration_gaps.apply_gap_view(samples, views[rule], provenance[rule])
             for rule in RULES}
    fit = [modes[rule][uid] for rule in RULES for uid in fit_uids]
    validation = [modes[rule][uid] for rule in RULES for uid in val_uids]
    fit, validation, scales = bind_fit_scales(model, fit, validation)
    scales = dict(scales, unique_fitting_users=64, context_count=3,
        user_context_observations=192, contexts=RULES,
        aggregation="equal user-context mean; 64 distinct fitting UIDs, each in all3 contexts")
    numerical = {str(p.relative_to(ROOT)): sha256(p) for p in _sources()}
    common = dict(format=FORMAT, edge=edge, models=models, population_sha256=population_sha,
        input_binding_sha256=sha256(binding_path), history_sha256=binding["history"]["sha256"],
        requests_sha256=binding["requests"]["sha256"], candidate_file=str(candidate_path.resolve()),
        candidate_file_sha256=sha256(candidate_path), fitting_candidate_pool_uids=fit_uids,
        scale_fit_uids=fit_uids, validation_uids=val_uids, queries=16, recent_fraction=.5,
        candidate_seed=17, candidate_ids=provenance[RULES[0]]["candidate_ids"],
        gap_manifest=str(gap_manifest.resolve()), gap_manifest_sha256=sha256(gap_manifest),
        time_rule=provenance[RULES[0]]["time_rule"], source_cutoff=cfg["release_day"] * 86400,
        candidate_order=provenance[RULES[0]]["candidate_order"], loss_scales=scales,
        objective=dict(read_weight=1., score_weight=1., aggregation="equal mean over reserved16 validation UIDs"),
        backend=backend, numerical_sources=numerical,
        scope="pre-release targets only; synthetic gap features on cutover caches; no historical replay, weight fitting or population AUC",
        probe_input_statistics="preserve trained input normalization and output scales",
        distinct_from_old_common="old fit256-ID/cutover-only common losses must not enter this comparison")
    contexts = {}
    for index, rule in enumerate(RULES):
        deltas = {str(uid): views[rule][uid].tolist() for uid in fit_uids + val_uids}
        identity = dict(common, query_delta_rule=rule, query_deltas=deltas,
            query_deltas_sha256=_digest(deltas), gap_provenance=provenance[rule])
        contexts[rule] = CommonValidation(model, fit[index*64:(index+1)*64],
            validation[index*16:(index+1)*16], population, identity, frozen,
            time.perf_counter() - started)
    return contexts


def _check_planned_weight(context, probe, metadata, spec):
    _check_weight_binding(context, probe, metadata)
    identity = metadata["identity"]
    # Actual legacy head/full/layerwise files omit fitting_driver. Only that
    # documented release-gap format can use the explicit old-driver default.
    driver = identity.get("fitting_driver", "run_probe_variants.py")
    if (driver != spec["fitting_driver"] or identity.get("seed") != spec["seed"] or
            identity.get("edge") != spec["edge"] or
            metadata.get("method_label", probe.stage) != spec["method_label"]):
        raise ValueError("planned checkpoint driver/seed/edge/branch differs")
    if driver == "run_probe_temporal.py":
        shared = context.identity
        if (identity.get("query_delta_rule") != spec["query_delta_rule"] or
                identity.get("gap_manifest_sha256") != spec["training_gap_manifest_sha256"] or
                identity.get("gap_manifest_sha256") != shared["gap_manifest_sha256"] or
                any(identity.get(key) != shared[key] for key in
                    ("time_rule", "source_cutoff", "candidate_order")) or
                identity.get("calibration_candidate_sha256") != shared["candidate_file_sha256"]):
            raise ValueError("temporal checkpoint candidate/time/manifest identity differs")
        source = ROOT / "experiments/ch3_motivation/code/run_probe_temporal.py"
        if identity.get("fitting_driver_sha256") != sha256(source):
            raise ValueError("temporal fitting driver SHA differs")
    elif "query_delta_rule" in identity:
        raise ValueError("temporal identity cannot be loaded as a legacy release-gap recipe")


@torch.no_grad()
def evaluate_planned_weight(contexts, spec):
    """One typed load, independent losses; pending/failed and step0 remain rows."""
    started = time.perf_counter()
    path = Path(spec["checkpoint"])
    row = dict(spec, status="pending" if not path.exists() else "failed", contexts={})
    if not path.exists():
        row.update(error="planned checkpoint not available yet", selection_eligible=False)
        return row
    try:
        row["checkpoint_sha256"] = sha256(path)
        first = contexts[RULES[0]]
        probe, metadata = load_variant(path, next(first.model.parameters()).device)
        row.update(variant=getattr(probe, "variant_name", "head"), stage=probe.stage,
            config=asdict(probe.cfg), fitting_users=len(metadata.get("identity", {}).get("fit_uids", [])),
            best_step=metadata.get("best_step"), completed_steps=metadata.get("completed_steps"),
            step_zero=metadata.get("best_step") == 0,
            original_validation_loss=_provenance(metadata.get("best_validation_loss")),
            original_loss_scales=_provenance(metadata.get("fit_pool_loss_scales")),
            training_identity=_provenance(metadata.get("identity")),
            cumulative_calibration_flops_estimate=metadata.get("cumulative_calibration_flops_estimate"),
            selection_eligible=not metadata.get("diagnostic_only", False) and
                               metadata.get("quality_use_permitted", True) is not False)
        _check_planned_weight(first, probe, metadata, spec)
        for rule in RULES:
            context = contexts[rule]
            row["contexts"][rule] = dict(status="validated", common_validation_sha256=context.fingerprint,
                                        metrics=evaluate_common_probe(context, probe))
        if sha256(path) != row["checkpoint_sha256"]:
            raise ValueError("checkpoint changed during temporal common validation")
        row.update(status="validated", mean_context_loss=sum(
            row["contexts"][rule]["metrics"]["loss"] for rule in RULES) / 3)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        row.update(error_type=type(error).__name__, error=str(error), selection_eligible=False)
        # A late context failure keeps the preceding context evidence, but the
        # overall weight is failed and can never enter a partial mean.
    row["seconds"] = time.perf_counter() - started
    return row


def run(args):
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text())
    if (plan["format"] != "ch3_temporal_common_plan_v1" or plan["rules"] != RULES or
            len(plan["weights"]) != 280 or len({p["checkpoint"] for p in plan["weights"]}) != 280):
        raise ValueError("expected the complete frozen old240 plus temporal40 plan")
    for path, digest in plan["source_fitting_plans_sha256"].items():
        if sha256(path) != digest:
            raise ValueError("source fitting plan changed")
    for path, digest in plan["executor_sources_sha256"].items():
        if sha256(path) != digest:
            raise ValueError("planned temporal common executor source changed")
    if sha256(Path(__file__).with_name("prepare_temporal_common_plan.py")) != plan["planner_sha256"]:
        raise ValueError("temporal common planner changed")
    binding = plan["edge_bindings"][str(args.edge)]
    for field, digest in (("candidate_file", "candidate_file_sha256"), ("gap_manifest", "gap_manifest_sha256")):
        if sha256(binding[field]) != binding[digest]:
            raise ValueError("planned temporal common input changed")
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    contexts = prepare_contexts(binding["input_root"], args.edge, device,
                               binding["gap_manifest"], backend=args.backend)
    output = Path(args.output)
    configuration = dict(plan=str(plan_path.resolve()), plan_sha256=sha256(plan_path),
        contexts={rule: context.identity for rule, context in contexts.items()})
    config_path = output / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise ValueError("new temporal common attempt configuration changed")
    write_json(config_path, configuration)
    specs = [p for p in plan["weights"] if p["edge"] == args.edge]
    if len(specs) != 56:
        raise ValueError("temporal common edge must cover48 old plus8 new weights")
    summary = dict(format=FORMAT, status="in_progress", edge=args.edge,
        plan_sha256=sha256(plan_path), quality_auc_computed=False, post_release_labels_loaded=False,
        fitting_performed=False, independent_fitting_users=64, scale_user_context_observations=192,
        contexts=configuration["contexts"], preparation_seconds=contexts[RULES[0]].preparation_seconds,
        native_reuse={rule: native_baseline(context) for rule, context in contexts.items()}, points=[])
    for spec in specs:
        point = evaluate_planned_weight(contexts, spec)
        summary["points"].append(point)
        write_json(output / "summary.json", summary)
        print(json.dumps(dict(status="temporal_common_point", checkpoint=spec["checkpoint"],
            result=point["status"], mean_context_loss=point.get("mean_context_loss")), allow_nan=False), flush=True)
    for path, digest in contexts[RULES[0]].frozen_paths.items():
        if sha256(path) != digest:
            raise ValueError(f"temporal common frozen input/source changed: {path}")
    if sha256(plan_path) != configuration["plan_sha256"]:
        raise ValueError("temporal common plan changed during evaluation")
    counts = {status: sum(p["status"] == status for p in summary["points"])
              for status in ("validated", "pending", "failed")}
    summary.update(counts, status="waiting_weights" if counts["pending"] else
                   ("complete_with_failures" if counts["failed"] else "complete"))
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--edge", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--backend", choices=("torch", "triton"), default="triton")
    result = run(parser.parse_args())
    raise SystemExit(0 if result["status"] == "complete" else 2)
