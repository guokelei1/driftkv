#!/usr/bin/env python3
"""Compare frozen E5 weights on one common, pre-release validation problem.

Different fitting recipes have different targets and loss denominators. Their
saved validation losses are provenance, not a common selection criterion. This
driver uses the fit256 feedback candidate view, Q16/50%/seed17, the same 16
reserved validation users, and loss scales from the first 64 reserved fit users.
It neither fits weights nor uses post-release feedback labels or population AUC.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time

from experiments.ch3_motivation.code.common import (
    ROOT, capture_samples, load_models, restore_pack, write_json,
)
from experiments.ch3_motivation.code.prepare_candidates import candidates_for
from experiments.ch3_motivation.code.probes import (
    ProbeConfig, ProbeLadder, bind_fit_scales, probe_loss, sample_to_device,
)
from experiments.ch3_motivation.code.probe_variants import load_variant
from hstu_kvcache.common.artifacts import sha256
import torch


FIT_SCALE_USERS = 64
VALIDATION_USERS = 16
QUERIES = 16
CANDIDATE_SEED = 17
RECENT_FRACTION = .5
FORMAT = "ch3_common_pre_release_validation_v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _provenance(value):
    """Keep a failed fit's NaN/inf provenance explicit in strict JSON output."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _provenance(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_provenance(child) for child in value]
    return value


@dataclass
class CommonValidation:
    model: object
    fit_samples: list
    validation_samples: list
    population: dict
    identity: dict
    frozen_paths: dict
    preparation_seconds: float = 0.

    @property
    def fingerprint(self):
        return _digest(self.identity)


def _source_paths():
    here = Path(__file__).resolve()
    return [here, *[here.with_name(name) for name in (
        "common.py", "probes.py", "probe_variants.py", "prepare_candidates.py")],
        ROOT / "src/hstu_kvcache/adaptation/reader.py"]


@torch.no_grad()
def prepare_common_validation(input_root, edge, device, *, backend="torch"):
    """Prepare one shared model/target/scale context for all weights on an edge.

    Only the history pack is read. The future-request pack is hashed for input
    identity but its rows and labels are not loaded. The first 64 fit users
    determine loss denominators; they do not fit or alter any probe parameter.
    """
    started = time.perf_counter()
    input_root = Path(input_root)
    directory = input_root / f"edge_{edge}"
    population_path = input_root / "population.json"
    binding_path = directory / "inputs.json"
    candidate_path = directory / "calibration_feedback_candidates.json"
    population = json.loads(population_path.read_text())
    binding = json.loads(binding_path.read_text())
    candidate_data = json.loads(candidate_path.read_text())
    population_hash = sha256(population_path)
    if binding["population_sha256"] != population_hash:
        raise ValueError("common validation population binding changed")
    frozen = {str(path.resolve()): sha256(path) for path in (
        population_path, binding_path, candidate_path, *_source_paths())}
    for name in ("history", "requests"):
        path = directory / binding[name]["path"]
        if sha256(path) != binding[name]["sha256"]:
            raise ValueError(f"common validation {name} pack changed")
        frozen[str(path.resolve())] = binding[name]["sha256"]
    cfg = dict(binding["settings"], calibration_queries_per_user=QUERIES,
               seed=CANDIDATE_SEED, attention_backend=backend)
    if int(cfg["edge"]) != int(edge):
        raise ValueError("common validation edge binding differs")
    fit_uids = list(population["fit"][:FIT_SCALE_USERS])
    validation_uids = list(population["validation"])
    if len(fit_uids) != FIT_SCALE_USERS or len(validation_uids) != VALIDATION_USERS:
        raise ValueError("common validation requires 64 reserved fit and 16 validation users")
    if len(set(fit_uids + validation_uids)) != FIT_SCALE_USERS + VALIDATION_USERS:
        raise ValueError("common validation fit/validation roles overlap or repeat")
    if (candidate_data["population_sha256"] != population_hash or
            candidate_data["edge"] != edge or
            candidate_data["release_day"] != cfg["release_day"] or
            candidate_data["seed"] != CANDIDATE_SEED or
            candidate_data["fitting_pool_uids"] != population["fit"][:256] or
            len(candidate_data["fitting_pool_uids"]) != 256 or
            candidate_data["validation_uids"] != validation_uids):
        raise ValueError("expected the common fit256 pre-release feedback candidate view")
    source_candidate = Path(candidate_data["source_candidate_path"])
    if sha256(source_candidate) != candidate_data["source_candidate_sha256"]:
        raise ValueError("source feedback candidate file changed")
    frozen[str(source_candidate.resolve())] = candidate_data["source_candidate_sha256"]

    parent, current, models = load_models(edge, device, backend=backend)
    known = int(current.cfg.num_items) - 256
    if known != candidate_data["known_items"] or known < QUERIES:
        raise ValueError("common candidate catalog differs from the frozen backbone")
    histories = restore_pack(directory / binding["history"]["path"])
    samples = capture_samples(parent, current, histories, fit_uids + validation_uids, cfg, device)
    candidate_ids = {}
    for uid in fit_uids + validation_uids:
        sample = samples[uid]
        values = candidates_for(uid, candidate_data["recent_items"][str(uid)], known,
                                count=QUERIES, recent_fraction=RECENT_FRACTION,
                                seed=CANDIDATE_SEED)
        sample["uid"] = int(uid)
        sample["candidates"] = torch.as_tensor(values, dtype=torch.long)[None]
        # Do not reuse a moment or teacher score computed for capture's older
        # candidate helper, even if a caller has already prepared that sample.
        for name in ("teacher_scores", "read_moment_sum", "score_moment_sum", "scale_users",
                     "target_moment_version", "read_scales", "score_scale", "loss_scale_version"):
            sample.pop(name, None)
        candidate_ids[str(uid)] = values
    fit, validation, scale_metadata = bind_fit_scales(current,
        [samples[uid] for uid in fit_uids], [samples[uid] for uid in validation_uids])
    del parent, histories, samples
    identity = dict(format=FORMAT, edge=edge, models=models,
        population_sha256=population_hash, input_binding_sha256=sha256(binding_path),
        history_sha256=binding["history"]["sha256"], requests_sha256=binding["requests"]["sha256"],
        candidate_file=str(candidate_path.resolve()), candidate_file_sha256=sha256(candidate_path),
        source_candidate_file=str(source_candidate.resolve()),
        source_candidate_sha256=candidate_data["source_candidate_sha256"],
        fitting_candidate_pool_uids=candidate_data["fitting_pool_uids"],
        scale_fit_uids=fit_uids, validation_uids=validation_uids,
        queries=QUERIES, recent_fraction=RECENT_FRACTION, candidate_seed=CANDIDATE_SEED,
        candidate_ids=candidate_ids, candidate_ids_sha256=_digest(candidate_ids),
        loss_scales=scale_metadata, objective=dict(read_weight=1., score_weight=1.,
            aggregation="equal mean over the same 16 reserved validation users"),
        backend=backend, history_lengths={str(s["uid"]): s["parent_cache"].seq_len
                                       for s in fit + validation},
        numerical_sources={str(p.relative_to(ROOT)): sha256(p) for p in _source_paths()},
        scope="pre-release validation only; no fitting, post-release labels or population AUC",
        probe_input_statistics="preserve each checkpoint's trained normalization and output scales")
    return CommonValidation(current, fit, validation, population, identity, frozen,
                            time.perf_counter() - started)


@torch.no_grad()
def evaluate_common_probe(context, probe):
    """Return an equal-user mean and its per-UID values, without changing weights."""
    probe.eval().requires_grad_(False)
    device = next(context.model.parameters()).device
    probe.to(device)
    rows = []
    for original in context.validation_samples:
        sample = sample_to_device(original, device)
        loss, metrics = probe_loss(context.model, probe, sample, read_weight=1., score_weight=1.)
        numbers = dict(loss=float(loss), **{name: float(value) for name, value in metrics.items()})
        if not all(math.isfinite(value) for value in numbers.values()):
            raise ValueError(f"nonfinite common validation loss for UID {original['uid']}")
        rows.append(dict(uid=original["uid"], history_length=original["parent_cache"].seq_len,
                         queries=sample["candidates"].shape[1], **numbers))
    return dict(users=len(rows), **{key: math.fsum(row[key] for row in rows) / len(rows)
        for key in ("loss", "read_mse", "score_mse", "raw_score_mse")}, per_user=rows)


def _check_weight_binding(context, probe, metadata):
    identity = metadata["identity"]
    shared = context.identity
    for name in ("edge", "models", "population_sha256", "input_binding_sha256"):
        if identity.get(name) != shared[name]:
            raise ValueError(f"checkpoint {name} differs from common validation inputs")
    if identity.get("validation_users") != shared["validation_uids"]:
        raise ValueError("checkpoint reserved validation UIDs differ")
    fit_uids = identity["fit_uids"]
    if not fit_uids or fit_uids != context.population["fit"][:len(fit_uids)]:
        raise ValueError("checkpoint fitting UIDs are not the frozen reserved prefix")
    cfg, model = probe.cfg, context.model
    if (cfg.num_layers != len(model.blocks) or cfg.num_heads != model.cfg.num_heads or
            cfg.head_dim != model.blocks[0].attn.head_dim):
        raise ValueError("checkpoint backbone geometry differs")
    # Fit-driver source can differ across recipes. The numerical definitions
    # that interpret frozen weights must match when the fit recorded them.
    for name, digest in identity.get("numerical_sources", {}).items():
        if (Path(name).name in ("probes.py", "probe_variants.py", "reader.py") and
                name in shared["numerical_sources"] and digest != shared["numerical_sources"][name]):
            raise ValueError(f"checkpoint numerical source differs: {name}")


@torch.no_grad()
def evaluate_checkpoint(context, checkpoint):
    """Retain failed/incompatible weights and step-zero results as explicit rows."""
    started = time.perf_counter()
    checkpoint = Path(checkpoint)
    row = dict(checkpoint=str(checkpoint.resolve()), status="failed",
               common_validation_sha256=context.fingerprint)
    try:
        row["checkpoint_sha256"] = sha256(checkpoint)
        probe, metadata = load_variant(checkpoint, next(context.model.parameters()).device)
        row.update(variant=getattr(probe, "variant_name", "head"), stage=probe.stage,
            method_label=metadata.get("method_label", probe.stage), config=asdict(probe.cfg),
            seed=metadata.get("seed", metadata.get("identity", {}).get("seed")),
            fitting_users=len(metadata.get("identity", {}).get("fit_uids", [])),
            best_step=metadata.get("best_step"), completed_steps=metadata.get("completed_steps"),
            step_zero=metadata.get("best_step") == 0,
            original_validation_loss=_provenance(metadata.get("best_validation_loss")),
            original_loss_scales=_provenance(metadata.get("fit_pool_loss_scales")),
            original_objective=_provenance(metadata.get("objective")),
            training_identity=_provenance(metadata.get("identity")),
            diagnostic_only=bool(metadata.get("diagnostic_only", False)),
            selection_eligible=not metadata.get("diagnostic_only", False) and
                               metadata.get("quality_use_permitted", True) is not False)
        _check_weight_binding(context, probe, metadata)
        row["metrics"] = evaluate_common_probe(context, probe)
        if sha256(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError("checkpoint changed during common validation")
        row["status"] = "validated"
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        row.pop("metrics", None)
        row.update(error_type=type(error).__name__, error=str(error), selection_eligible=False)
    row["seconds"] = time.perf_counter() - started
    return row


def native_baseline(context):
    """Zero f is exactly Reuse, evaluated by the same loss/trace definition."""
    zero = ProbeLadder(ProbeConfig.from_model(context.model, hidden_width=1, depth=1,
        include_item=False, normalize=False, query_chunk=QUERIES, row_chunk=1024), "f")
    return dict(method_label="native_reuse", common_validation_sha256=context.fingerprint,
                metrics=evaluate_common_probe(context, zero))


def run(args):
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    paths = [Path(path).resolve() for path in args.checkpoint]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate checkpoint paths")
    context = prepare_common_validation(args.input_root, args.edge, device, backend=args.backend)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    configuration = dict(common_validation=context.identity,
        common_validation_sha256=context.fingerprint, checkpoints=[str(path) for path in paths])
    configuration_path = output / "configuration.json"
    if configuration_path.exists() and json.loads(configuration_path.read_text()) != configuration:
        raise ValueError("common validation attempt configuration changed")
    write_json(configuration_path, configuration)
    summary = dict(format=FORMAT, status="in_progress", quality_auc_computed=False,
        post_release_labels_loaded=False, fitting_performed=False,
        common_validation_sha256=context.fingerprint, common_validation=context.identity,
        preparation_seconds=context.preparation_seconds, native_reuse=native_baseline(context), points=[])
    write_json(output / "summary.json", summary)
    for path in paths:
        point = evaluate_checkpoint(context, path)
        summary["points"].append(point)
        write_json(output / "summary.json", summary)
        print(json.dumps(dict(status="common_validation_point", checkpoint=str(path),
            result=point["status"], loss=point.get("metrics", {}).get("loss"),
            best_step=point.get("best_step")), allow_nan=False), flush=True)
    for path, digest in context.frozen_paths.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"common validation frozen input/source changed: {path}")
    failed = sum(point["status"] != "validated" for point in summary["points"])
    summary.update(status="complete" if failed == 0 else "complete_with_failures",
                   validated=len(paths) - failed, failed=failed)
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path,
        default=ROOT / "experiments/ch3_motivation/inputs_feedback_pool/fit256")
    parser.add_argument("--edge", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--backend", choices=("torch", "triton"), default="torch")
    run(parser.parse_args())
