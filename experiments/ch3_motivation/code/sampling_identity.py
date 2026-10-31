"""Exact identities for the two concrete fixed256 calibration drivers.

Legacy release-gap and law64 weights keep their existing separate checks.
"""
from __future__ import annotations

import json
from pathlib import Path
from hstu_kvcache.common.artifacts import sha256

SAMPLING_DRIVERS = ("run_probe_calibration.py", "run_probe_calibration_init.py")


def sampling_source_paths():
    """Both supported branches, frozen before a checkpoint chooses its branch.

    The drivers are recorded fitting provenance; the remaining files execute
    manifest, causal candidate/time, or initialization identity validation.
    """
    names = ("sampling_identity.py", "calibration_sampling.py", "calibration_gaps.py",
        "prepare_candidates.py", "probe_initialization.py", *SAMPLING_DRIVERS)
    return [Path(__file__).with_name(name) for name in names]


def validate_sampling_capture(identity, metadata):
    n = len(identity["fit_uids"])
    if (metadata.get("training_users") != n or metadata.get("validation_users") != 16 or
            metadata.get("teacher_preparation_users") != n+16 or
            metadata.get("actual_teacher_users_captured") != n or
            metadata.get("actual_validation_users_captured") != 16):
        raise ValueError("sampling calibration actual teacher/gradient/validation user coverage differs")


def _initialization_identity(identity, config):
    fields = ("h_initialization", "h_initialization_spec", "initialization_helper_sha256")
    if identity["fitting_driver"] == "run_probe_calibration.py":
        if any(key in identity for key in fields):
            raise ValueError("original sampling driver cannot carry an initialization-driver identity")
        return {}
    from experiments.ch3_motivation.code.probe_initialization import initialization_spec
    if config is None or identity.get("variant") != "full_item":
        raise ValueError("initialization sampling calibration needs the actual full-item configuration")
    query_dim = config["head_dim"]
    row_dim = 2*query_dim + (config["item_dim"] if config["include_item"] else 0)
    expected = initialization_spec(query_dim, row_dim, identity.get("h_initialization"))
    if (identity.get("h_initialization_spec") != expected or
            identity.get("initialization_helper_sha256") != expected["helper_sha256"] or
            identity.get("numerical_sources", {}).get("experiments/ch3_motivation/code/probe_initialization.py") != expected["helper_sha256"]):
        raise ValueError("initialization sampling calibration spec/actual dimensions/helper SHA256 differs")
    return {key:identity[key] for key in fields}


def validate_sampling_identity(identity, *, sampling_cache=None, config=None):
    sampling_cache = {} if sampling_cache is None else sampling_cache
    from experiments.ch3_motivation.code.calibration_sampling import load_sampling_manifest

    digest, alias = identity.get("sampling_manifest_sha256"), identity.get("gap_manifest_sha256")
    driver = identity.get("fitting_driver")
    if driver not in SAMPLING_DRIVERS:
        raise ValueError("unknown explicit sampling calibration fitting driver")
    driver_sha = identity.get("fitting_driver_sha256")
    if (not isinstance(digest, str) or len(digest) != 64 or
            any(char not in "0123456789abcdef" for char in digest) or alias != digest):
        raise ValueError("sampling calibration requires the exact nonnull sampling/gap manifest SHA256")
    paths = [identity.get(key) for key in ("sampling_manifest", "gap_manifest")]
    if (not all(isinstance(path, str) and path for path in paths) or
            Path(paths[0]).resolve() != Path(paths[1]).resolve()):
        raise ValueError("sampling calibration manifest paths differ or are missing")
    path = Path(paths[0]).resolve()
    if sha256(path) != digest:
        raise ValueError("sampling calibration manifest SHA256 changed")
    if (not isinstance(driver_sha, str) or len(driver_sha) != 64 or
            any(char not in "0123456789abcdef" for char in driver_sha) or
            identity.get("numerical_sources", {}).get(f"experiments/ch3_motivation/code/{driver}") != driver_sha):
        raise ValueError("sampling calibration fitting driver SHA256 is not bound to its numerical sources")
    n = identity.get("fit_users")
    if (n not in (64, 128, 256) or identity.get("gradient_fitting_users") != n or
            identity.get("teacher_fitting_users") != n or
            identity.get("candidate_catalog_source_users") != 256 or identity.get("time_law_source_users") != 256 or
            identity.get("queries") != 16 or identity.get("query_delta_layout") != "[B,16]" or
            identity.get("candidate_source") != "feedback" or identity.get("recent_fraction") not in (.5, .75) or
            identity.get("candidate_seed") != 17 or identity.get("query_delta_rule") not in ("cutover", "gap_mix50", "gap_empirical100") or
            identity.get("fitting_uses_future_feedback") is not False):
        raise ValueError("sampling calibration actual-N/Q16/fixed256-source protocol differs")
    source = json.loads(path.read_text())["sourcebinding"]
    key = (str(path), digest, identity["edge"], n, identity["recent_fraction"], identity["query_delta_rule"])
    if key not in sampling_cache:
        _, sampling_cache[key] = load_sampling_manifest(path, source["input_root"], identity["edge"],
            identity["recent_fraction"], identity["query_delta_rule"], fit_users=n)
    provenance = sampling_cache[key]
    if (identity["fit_uids"] != provenance["fit_uids"] or
            identity["validation_users"] != provenance["validation_uids"] or
            identity["input_binding_sha256"] != provenance["sourcebinding"]["sha256"] or
            identity.get("population_sha256") != provenance["sourcebinding"]["population_sha256"] or
            identity.get("calibration_candidate_sha256") != provenance["sourcebinding"]["candidate_sha256"] or
            any(identity.get(key) != provenance[key] for key in ("time_rule", "source_cutoff", "candidate_order"))):
        raise ValueError("sampling calibration roles/input/time/candidate provenance differs")
    initialization = _initialization_identity(identity, config)
    keys = ("fitting_driver", "fitting_driver_sha256", "query_delta_rule", "sampling_manifest_sha256",
        "gap_manifest_sha256", "time_rule", "source_cutoff", "candidate_order", "query_delta_layout",
        "candidate_catalog_source_users", "time_law_source_users", "gradient_fitting_users", "teacher_fitting_users")
    return dict({key: identity[key] for key in keys}, **initialization, sampling_manifest=str(path), gap_manifest=str(path),
        sampling_view_key=provenance["view_key"], sampling_view_users_sha256=provenance["view_users_sha256"])

