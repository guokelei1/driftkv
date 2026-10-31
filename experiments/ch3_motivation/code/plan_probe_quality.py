#!/usr/bin/env python3
"""Prepare E5 quality jobs from a selected formal recipe and real checkpoints.

This writes three separate plans; it never launches fitting or inference.
Missing weights are reported explicitly. Actual metadata must match the planned
driver, temporal rule, fitting users and controls before a plan becomes ready.
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
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
import torch

from experiments.ch3_motivation.code.common import write_json
from experiments.ch3_motivation.code import probes, resident_probe_fitting
from experiments.ch3_motivation.code.sampling_identity import (
    SAMPLING_DRIVERS, validate_sampling_capture, validate_sampling_identity, sampling_source_paths,
)
from hstu_kvcache.common.artifacts import sha256

MAIN_METHODS = ("f", "fg", "fgh", "f_extended", "fg_extended")
FIELDS = ("learning_rate", "steps", "read_weight", "score_weight", "patience",
          "validation_every", "candidate_source", "recent_fraction", "queries", "stages", "controls")
RESIDENT_HELPER_SHA = "3847f3bdafbd33fe4b16ade7ad9305787d2c7b88b871e473cb1922e90e1e011b"
RESIDENT_REFERENCE_SHA = "6ab7246f3f4d2d2867ca3e69f2db85f8513b1ff6ae3d3983cf5c76aac2ede934"


def resident_execution_binding():
    """Exact binding of the helper/core pair reviewed on actual Large GPU fits."""
    helper = resident_probe_fitting._source(resident_probe_fitting.__file__)
    reference = resident_probe_fitting._source(probes.__file__)
    if helper["sha256"] != RESIDENT_HELPER_SHA or reference["sha256"] != RESIDENT_REFERENCE_SHA:
        raise ValueError("resident execution adapter source differs from the reviewed helper/core")
    return dict(format="resident_probe_fitting_v1", helper_source=helper, reference_source=reference,
        reference_function="experiments.ch3_motivation.code.probes.fit_probe", placed_fields=list(resident_probe_fitting.PLACED_FIELDS),
        placement_policy="one-stage shallow sample copies; identical storage-view memo; no computed-feature cache",
        core_arithmetic="unchanged original fit_probe operation_counts and arithmetic; placement adds no arithmetic charge",
        lifetime="resident sample lists released after this fit call; no cross-stage cache")


def _audit_execution_adapter(metadata, job, expected):
    planned = job.get("execution_adapter")
    if planned is not None and (not isinstance(planned, dict) or set(planned) != {"binding"}):
        raise ValueError("quality job execution adapter must contain its expected binding")
    if expected is not None:
        if planned is not None and planned["binding"] != expected:
            raise ValueError("quality explicit/job execution adapter bindings differ")
        planned = dict(binding=expected)
    actual = metadata.get("execution_adapter")
    if planned is not None and planned["binding"] is None:
        if actual is not None:
            raise ValueError("quality job requires reference fitting without an execution adapter")
        return None
    if planned is not None and planned["binding"] != resident_execution_binding():
        raise ValueError("quality expected execution adapter differs from the reviewed fitter")
    if actual is None:
        if planned is not None:
            raise ValueError("quality weight is missing its expected execution adapter")
        return None
    if (not isinstance(actual, dict) or set(actual) != {"binding", "metrics"} or
            not isinstance(actual["metrics"], dict) or actual["binding"] != resident_execution_binding()):
        raise ValueError("quality weight execution adapter binding differs from the reviewed fitter")
    return actual


def sources():
    paths = [Path(__file__), Path(__file__).with_name("common.py"),
        ROOT/"src/hstu_kvcache/common/artifacts.py", *sampling_source_paths(),
        Path(resident_probe_fitting.__file__), Path(probes.__file__)]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def checkpoint_path(root, job, method):
    folder = Path(job["output"]) if "output" in job else (
        Path(root) / job["name"] / f"edge_{job['edge']}" / f"seed_{job['seed']}")
    return folder / f"{method}.pt"


def audit_weight(path, job, binding, method, *, expected_execution_adapter=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata, config = payload["metadata"], payload["config"]
    identity = metadata["identity"]
    execution_adapter = _audit_execution_adapter(metadata, job, expected_execution_adapter)
    stage = {"f_extended": "f", "fg_extended": "fg"}.get(method, method)
    if (payload.get("format") != "ch3_probe_variant_v1" or payload.get("variant") != job["variant"] or
            payload["stage"] != stage or metadata.get("method_label") != method or
            metadata["seed"] != job["seed"] or metadata["training_users"] != job["fit_users"] or
            config["hidden_width"] != job["width"] or config["depth"] != job["depth"] or
            identity["edge"] != job["edge"] or identity["seed"] != job["seed"]):
        raise ValueError(f"Quality weight architecture/seed/branch differs from formal job: {path}")
    if any(identity.get(key) != job[key] for key in FIELDS):
        raise ValueError(f"Quality fitting recipe differs from formal job: {path}")
    if (identity["input_binding_sha256"] != binding["input_binding_sha256"] or
            identity["population_sha256"] != binding["population_sha256"] or
            identity.get("calibration_candidate_sha256") !=
            (binding["candidate_file_sha256"] if job["candidate_source"] == "feedback" else None)):
        raise ValueError(f"Quality weight calibration inputs differ from formal plan: {path}")
    population = json.loads((Path(binding["input_root"]) / "population.json").read_text())
    for file, digest in ((Path(binding["input_root"]) / "population.json", binding["population_sha256"]),
                         (Path(binding["input_root"]) / f"edge_{job['edge']}" / "inputs.json",
                          binding["input_binding_sha256"]),
                         (Path(binding["candidate_file"]), binding["candidate_file_sha256"])):
        if sha256(file) != digest:
            raise ValueError(f"Formal quality input changed: {file}")
    if (identity["fit_uids"] != population["fit"][:job["fit_users"]] or
            identity["validation_users"] != population["validation"]):
        raise ValueError(f"Quality weight fitting roles differ: {path}")
    rule = job.get("query_delta_rule", "cutover")
    if identity.get("query_delta_rule", "cutover") != rule:
        raise ValueError(f"Quality temporal rule differs from formal plan: {path}")
    if "fitting_driver" in job:
        driver = identity.get("fitting_driver", "run_probe_variants.py")
        if driver != job["fitting_driver"]:
            raise ValueError(f"Quality fitting driver differs from formal plan: {path}")
        driver_source = f"experiments/ch3_motivation/code/{driver}"
        recorded_driver_sha = identity.get("fitting_driver_sha256",
            identity.get("numerical_sources", {}).get(driver_source))
        if "fitting_driver_sha256" in job and (
                recorded_driver_sha != job["fitting_driver_sha256"] or
                sha256(ROOT / "experiments/ch3_motivation/code" / driver) != job["fitting_driver_sha256"]):
            raise ValueError(f"Quality fitting driver source changed: {path}")
    driver = identity.get("fitting_driver", "run_probe_variants.py")
    sampling = None
    if driver in SAMPLING_DRIVERS:
        if (job.get("fitting_driver") != driver or
                job.get("fitting_driver_sha256") != identity.get("fitting_driver_sha256")):
            raise ValueError(f"Quality sampling driver/SHA must be explicitly planned: {path}")
        sampling = validate_sampling_identity(identity, config=config)
        validate_sampling_capture(identity, metadata)
        # Cutover is a view of the same fixed-source manifest. All three time
        # rules carry the same mandatory aliases and actual-N source records.
        for name in ("sampling_manifest_sha256", "gap_manifest_sha256", "time_rule", "source_cutoff",
                "candidate_order", "query_delta_layout", "candidate_catalog_source_users",
                "time_law_source_users", "gradient_fitting_users", "teacher_fitting_users"):
            if job.get(name) != identity.get(name):
                raise ValueError(f"Quality sampling job/checkpoint {name} differs: {path}")
        for name in ("sampling_manifest", "gap_manifest"):
            if (not job.get(name) or Path(job[name]).resolve() != Path(identity[name]).resolve()):
                raise ValueError(f"Quality sampling job/checkpoint manifest path differs: {path}")
        if (binding.get("sampling_manifest_sha256") != identity["sampling_manifest_sha256"] or
                not binding.get("sampling_manifest") or
                Path(binding["sampling_manifest"]).resolve() != Path(identity["sampling_manifest"]).resolve()):
            raise ValueError(f"Quality sampling manifest differs from formal input binding: {path}")
        if driver == "run_probe_calibration_init.py" and any(job.get(name) != identity.get(name)
                for name in ("h_initialization", "h_initialization_spec", "initialization_helper_sha256")):
            raise ValueError(f"Quality sampling initialization differs from formal job: {path}")
        scales = metadata.get("fit_pool_loss_scales", {})
        if (scales.get("users") != job["fit_users"] or scales.get("source") != "fitting_users_only" or
                scales.get("validation_uses_fitting_scales") is not True):
            raise ValueError(f"Quality sampling loss RMS does not use its own N fitting users: {path}")
        cap = job["steps"] * (2 if method == "f_extended" else 1)
        if job["patience"] == 0 and metadata["completed_steps"] != cap:
            raise ValueError(f"Quality sampling patience0 checkpoint did not execute its full cap: {path}")
    elif any(key in identity for key in ("sampling_manifest", "sampling_manifest_sha256")):
        raise ValueError(f"Quality sampling identity has no recognized explicit fitting driver: {path}")
    elif rule != "cutover":
        manifest = Path(job["gap_manifest"])
        digest = sha256(manifest)
        if (identity.get("fitting_driver") != job["fitting_driver"] or
                identity.get("gap_manifest_sha256") != digest or
                job["gap_manifest_sha256"] != digest):
            raise ValueError(f"Quality temporal driver/manifest differs: {path}")
        if any(identity.get(key) != job[key] for key in ("time_rule", "source_cutoff", "candidate_order")):
            raise ValueError(f"Quality temporal inputs differ from formal plan: {path}")
    if (metadata.get("diagnostic_only") or metadata.get("quality_use_permitted") is False or
            metadata.get("fitting_uses_future_feedback") or
            metadata.get("fit_pool_loss_scales", {}).get("version") !=
            "fit_pool_equal_user_length_normalized_v1"):
        raise ValueError(f"Quality weights lack the corrected, causal fitting protocol: {path}")
    return dict(checkpoint_sha256=sha256(path), best_step=metadata["best_step"],
                completed_steps=metadata["completed_steps"], method_label=method,
                **(dict(execution_adapter=execution_adapter) if execution_adapter is not None else {}),
                **(dict(sampling_identity=sampling) if sampling is not None else {}))


def generate(formal_path, main_root, curve_root, output):
    formal_path, output = Path(formal_path), Path(output)
    formal = json.loads(formal_path.read_text())
    code = sources()
    if formal["status"] != "prepared_for_root_review":
        result = dict(status="waiting_selection", gpu_launch_performed=False,
                      formal_plan=str(formal_path.resolve()), formal_plan_sha256=sha256(formal_path),
                      source_sha256=code)
        write_json(output / "summary.json", result)
        return result
    main, curve = formal["main_jobs"], formal["curve_jobs"]
    curve_pending = not curve and formal.get("curve_plan_status", "").startswith("pending_")
    if (len(main) != 15 or {(j["edge"], j["seed"]) for j in main} !=
            {(e, s) for e in range(1, 6) for s in (17, 23, 29)} or
            (not curve_pending and (len(curve) != 45 or
             {(j["edge"], j["seed"], j["fit_users"]) for j in curve} !=
             {(e, s, n) for e in range(1, 6) for s in (17, 23, 29) for n in (64, 128, 256)}))):
        raise ValueError("Formal quality generation needs the exact planned five-edge/three-seed fitting grids")
    families = {}
    for family, panel, source, root, methods, bindings, expected in (
        ("main", "main", main, main_root, MAIN_METHODS, formal["main_input_bindings"], 75),
        ("confirmation", "confirmation", [j for j in main if j["seed"] == 17], main_root,
         MAIN_METHODS[:3], formal["main_input_bindings"], 15),
        ("f_population_curve", "main", curve, curve_root, ("f",), formal.get("curve_input_bindings", {}), 45)):
        if family == "f_population_curve" and curve_pending:
            value = dict(status="waiting_curve_preparation", jobs=[], expected_points=45,
                reason=formal["curve_plan_status"], formal_plan_sha256=sha256(formal_path),
                source_sha256=code,
                scope="main and confirmation may proceed independently; no unsupported temporal curve is dispatched")
            write_json(output / f"{family}.json", value)
            families[family] = dict(status=value["status"], points=0, expected_points=45,
                                    available_weights=0, missing_weights=45)
            continue
        jobs, missing, audit = [], [], []
        for job in source:
            for method in methods:
                path = checkpoint_path(root, job, method).resolve()
                name = f"edge_{job['edge']}_seed_{job['seed']}_fit{job['fit_users']}_{method}"
                jobs.append(dict(name=name, edge=job["edge"], panel=panel, checkpoint=str(path),
                    input_root=str(ROOT / "experiments/ch3_motivation/inputs"), backend="projected"))
                if not path.exists():
                    missing.append(str(path))
                else:
                    audit.append(dict(checkpoint=str(path),
                        **audit_weight(path, job, bindings[str(job["edge"])], method)))
        if len(jobs) != expected or len({j["name"] for j in jobs}) != expected:
            raise ValueError("Quality grid has duplicate or missing planned cells")
        value = dict(status="prepared_for_root_review" if not missing else "waiting_weights",
            family=family, users_per_point=2000, jobs=jobs, missing_weights=missing,
            actual_weight_audit=audit, formal_plan=str(formal_path.resolve()),
            formal_plan_sha256=sha256(formal_path), planner_source_sha256=sha256(Path(__file__)),
            source_sha256=code,
            scope="selected fixed recipe; all five edges and predeclared seeds; main/confirmation/curve kept separate; no GPU launch")
        write_json(output / f"{family}.json", value)
        families[family] = dict(status=value["status"], points=len(jobs),
                                available_weights=len(audit), missing_weights=len(missing))
    result = dict(status="prepared_for_root_review" if all(v["status"] == "prepared_for_root_review"
        for v in families.values()) else "waiting_weights", gpu_launch_performed=False, families=families,
        formal_plan=str(formal_path.resolve()), formal_plan_sha256=sha256(formal_path),source_sha256=code)
    if sources() != code:
        raise ValueError("Quality plan identity/numerical validation sources changed while auditing weights")
    write_json(output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-plan", type=Path, required=True)
    parser.add_argument("--main-fitting-root", type=Path, required=True)
    parser.add_argument("--curve-fitting-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.formal_plan, args.main_fitting_root,
                              args.curve_fitting_root, args.output)))
