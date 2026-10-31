"""The trained E5 FG baseline owns its E14 state independently of full-H.

This helper is intentionally separate from the frozen E6 readiness driver.
The main E6 retention denominator will use these independently trained FG
scores. Deleting h on a full-H snapshot is a different structural diagnostic.
Both branches may borrow the same immutable old prefill, but never a graph
pool, prepared f cache, or subsequent behavior writes.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.ch3_motivation.code.probe_projected_fast import (
    ProjectedFastProbeTrajectory, ProjectedGraphPool,
)
from experiments.ch3_motivation.code.probe_replay import ProbeTrajectory
from experiments.ch3_motivation.code.sampling_identity import (
    SAMPLING_DRIVERS, validate_sampling_capture, validate_sampling_identity,
)
from experiments.ch3_motivation.code.probe_variants import load_variant
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.common.artifacts import sha256

TRAINED_FG_METHOD = "e5_trained_fg_own_trajectory"
H_DISABLED_DIAGNOSTIC = "h_disabled_on_full_h_trajectory"


def _query_delta_identity(identity, *, sampling_cache=None, config=None):
    rule = identity.get("query_delta_rule", "cutover")
    manifest = identity.get("gap_manifest_sha256")
    if rule not in ("cutover", "gap_mix50", "gap_empirical100"):
        raise ValueError("unknown calibration query delta rule")
    if identity.get("fitting_driver") in SAMPLING_DRIVERS:
        # The new sampler binds all six views, including cutover. It must not
        # inherit the legacy release-gap convention of a null manifest.
        return validate_sampling_identity(identity, sampling_cache=sampling_cache, config=config)
    if any(key in identity for key in ("sampling_manifest", "sampling_manifest_sha256")):
        raise ValueError("sampling calibration has no recognized explicit fitting driver")
    if rule == "cutover":
        if manifest is not None:
            raise ValueError("cutover calibration unexpectedly carries a temporal gap manifest")
    elif (not isinstance(manifest, str) or len(manifest) != 64 or
            any(value not in "0123456789abcdef" for value in manifest)):
        raise ValueError("temporal calibration requires the exact gap manifest SHA256")
    return dict(query_delta_rule=rule, gap_manifest_sha256=manifest)


def _fitting_identity(probe, metadata, source_population, evaluation_uids, *, edge,
                      models, calibration_binding_sha256, allow_diagnostic):
    identity = metadata["identity"]
    if identity["edge"] != edge or identity["models"] != models:
        raise ValueError("FG/full-H probe edge or backbone identity differs")
    if identity["input_binding_sha256"] != calibration_binding_sha256:
        raise ValueError("probe calibration binding differs from the preserved fitting inputs")
    variant = getattr(probe, "variant_name", "head")
    if metadata.get("variant", variant) != variant or identity.get("variant", variant) != variant:
        raise ValueError("probe variant tag differs from its loaded numerical architecture")
    fit = list(identity["fit_uids"]) if "fit_uids" in identity else list(
        source_population["fit"][:int(identity["fit_users"])])
    # The actual runners put UID membership inside identity. The top-level
    # validation_users field is a count and must never be treated as membership.
    validation = list(identity["validation_users"])
    if identity.get("fitting_driver") in SAMPLING_DRIVERS:
        validate_sampling_capture(identity, metadata)
    if not fit or not validation or len(set(fit)) != len(fit) or len(set(validation)) != len(validation):
        raise ValueError("probe fitting/validation membership is empty or duplicated")
    if not set(fit) <= set(source_population["fit"]) or not set(validation) <= set(source_population["validation"]):
        raise ValueError("probe fitting users are outside the frozen reserved roles")
    if set(fit) & set(validation) or set(evaluation_uids) & set(fit+validation):
        raise ValueError("probe fitting, validation, and evaluation users overlap")
    if metadata.get("fitting_uses_future_feedback") is True:
        raise ValueError("future-feedback-fitted probes cannot be used as E6 controls")
    if not allow_diagnostic and (metadata.get("diagnostic_only") or metadata.get("quality_use_permitted") is False):
        raise ValueError("a diagnostic checkpoint cannot define a formal E6 baseline")
    seed = identity.get("seed", metadata.get("seed"))
    if seed is None:
        raise ValueError("probe fitting seed is missing")
    return dict(variant=variant, seed=int(seed), fit_uids=fit,
        validation_uids=validation, calibration_binding_sha256=calibration_binding_sha256)


def validate_fg_pair(fg_probe, fg_metadata, full_h_probe, full_h_metadata, *, source_population,
                     evaluation_uids, edge, models, calibration_binding_sha256,
                     expected_method_label="fg", allow_diagnostic=False):
    """Validate actual head/tagged formats and the paired E5 fitting setting.

    fg_extended has the same stage as fg, so its explicit method_label cannot
    be silently substituted. A caller choosing that E5 control must request
    its label explicitly and retain the name in the resulting provenance.
    """
    if fg_probe.stage != "fg" or full_h_probe.stage != "fgh":
        raise ValueError("the independent baseline must be stage fg and its paired full probe stage fgh")
    label = fg_metadata.get("method_label", fg_metadata.get("stage", fg_probe.stage))
    if label != expected_method_label:
        raise ValueError("FG checkpoint method label differs from the requested E5 control")
    options = dict(source_population=source_population, evaluation_uids=evaluation_uids,
        edge=edge, models=models, calibration_binding_sha256=calibration_binding_sha256,
        allow_diagnostic=allow_diagnostic)
    fg = _fitting_identity(fg_probe, fg_metadata, **options)
    full = _fitting_identity(full_h_probe, full_h_metadata, **options)
    left, right = fg_metadata["identity"], full_h_metadata["identity"]
    if (any(value.get("fitting_driver") in SAMPLING_DRIVERS for value in (left, right)) and
            left != right):
        raise ValueError("FG/full-H sampling calibration fitting identities differ")
    sampling_cache = {}
    temporal = _query_delta_identity(left, sampling_cache=sampling_cache, config=asdict(fg_probe.cfg))
    if temporal != _query_delta_identity(right, sampling_cache=sampling_cache, config=asdict(full_h_probe.cfg)):
        raise ValueError("FG/full-H temporal query delta rule or gap manifest differs")
    if temporal["query_delta_rule"] != "cutover" and any(
            fg_metadata["identity"].get(key) != full_h_metadata["identity"].get(key)
            for key in ("time_rule", "source_cutoff", "candidate_order")):
        raise ValueError("FG/full-H temporal calibration provenance differs")
    if fg != full:
        raise ValueError("FG/full-H variants, seeds, or fitting/validation membership differ")
    fields = ("num_layers", "num_heads", "head_dim", "item_dim", "include_item",
              "normalize", "hidden_width", "depth")
    if any(getattr(fg_probe.cfg, name) != getattr(full_h_probe.cfg, name) for name in fields):
        raise ValueError("FG/full-H paired E5 probe architectures differ")
    for name in ("candidate_seed", "candidate_source", "recent_fraction", "calibration_candidate_sha256"):
        if left.get(name) != right.get(name):
            raise ValueError("FG/full-H calibration candidate protocol differs")
    fg_queries = left.get("queries", left.get("sample_candidates"))
    full_queries = right.get("queries", right.get("sample_candidates"))
    if fg_queries != full_queries:
        raise ValueError("FG/full-H calibration query counts differ")
    return dict(**fg, **temporal, edge=edge, models=models, method=TRAINED_FG_METHOD,
        e5_method_label=label, independent_state=True,
        fg_metadata_sha256=hashlib.sha256(json.dumps(fg_metadata, sort_keys=True).encode()).hexdigest(),
        full_h_metadata_sha256=hashlib.sha256(json.dumps(full_h_metadata, sort_keys=True).encode()).hexdigest(),
        state_protocol="same immutable old initial prefill; independent f preparation, graph pool, reads and behavior writes",
        extra_structural_control=H_DISABLED_DIAGNOSTIC)


def load_fg_control(path, full_h_probe, full_h_metadata, *, device="cpu", **validation):
    """Load the actual frozen E5 fg checkpoint, never derive it by deleting h."""
    path = Path(path).resolve()
    fg_probe, metadata = load_variant(path, device)
    identity = validate_fg_pair(fg_probe, metadata, full_h_probe, full_h_metadata, **validation)
    fg_probe.eval().requires_grad_(False)
    identity.update(checkpoint_path=str(path), checkpoint_sha256=sha256(path))
    return fg_probe, metadata, identity


def _initial(parent, events, cfg, initial):
    at = int(cfg["release_day"])*86400
    window = int(cfg.get("history_length", 1024))
    band = int(cfg.get("append_band_size", 128))
    if initial is None:
        return NativeTrajectory(parent, int(cfg["edge"])-1, events, at,
            max_length=window, append_band_size=band)
    if (initial.model is not parent or initial.version != int(cfg["edge"])-1 or initial.time != at
            or initial.max_length != window or initial.append_band_size != band):
        raise ValueError("shared old prefill does not match this user's edge/cutover/window")
    if any(not np.array_equal(a, b) for a, b in zip(initial.events, events, strict=True)):
        raise ValueError("shared old prefill belongs to different raw user events")
    return initial


@dataclass
class PairedTrajectories:
    initial: NativeTrajectory
    fg: ProbeTrajectory
    full_h: ProbeTrajectory


def make_paired_trajectories(parent, current, fg_probe, full_h_probe, events, cfg, *,
                             fg_pool=None, full_h_pool=None, initial=None, capture=True,
                             execution="projected"):
    """One old prefill; two independent prepared caches and mutable trajectories.

    Keep each pool alive across users, but provide a different pool for FG and
    full-H. The returned initial object is borrowed read-only by both branches.
    """
    if fg_probe.stage != "fg" or full_h_probe.stage != "fgh":
        raise ValueError("paired trajectories require independently fitted fg and fgh probes")
    initial = _initial(parent, events, cfg, initial)
    if execution == "dense":
        if fg_pool is not None or full_h_pool is not None:
            raise ValueError("dense reference trajectories do not use projected graph pools")
        return PairedTrajectories(initial,
            ProbeTrajectory(parent, current, fg_probe, events, cfg, initial=initial),
            ProbeTrajectory(parent, current, full_h_probe, events, cfg, initial=initial))
    if execution != "projected":
        raise ValueError("unknown FG/full-H trajectory execution")
    window = int(cfg.get("history_length", 1024))
    fg_pool = fg_pool or ProjectedGraphPool(current, fg_probe, window=window, capture=capture)
    full_h_pool = full_h_pool or ProjectedGraphPool(current, full_h_probe, window=window, capture=capture)
    if fg_pool is full_h_pool:
        raise ValueError("FG and full-H must not share a mutable graph pool")
    fg = ProjectedFastProbeTrajectory(parent, current, fg_probe, events, cfg, pool=fg_pool, initial=initial)
    full = ProjectedFastProbeTrajectory(parent, current, full_h_probe, events, cfg, pool=full_h_pool, initial=initial)
    return PairedTrajectories(initial, fg, full)


@torch.no_grad()
def score_fg_timestamp(trajectory, requests, *, uid, timestamp, query_batch=16):
    """Own-state FG scores for every actual row at one real feedback time."""
    timestamp = int(timestamp)
    at, end = int(trajectory.cfg["release_day"])*86400, int(trajectory.cfg["end_day"])*86400
    if trajectory.probe.stage != "fg" or not at <= timestamp < end or query_batch < 1:
        raise ValueError("FG scoring stage, timestamp, or query batch is invalid")
    if any(int(row["query_timestamp"]) != timestamp or int(row["uid"]) != int(uid) for row in requests):
        raise ValueError("FG requests do not match the real user/timestamp")
    trajectory.advance(timestamp)
    output = []
    for start in range(0, len(requests), query_batch):
        rows = requests[start:start+query_batch]
        ids = torch.as_tensor([[int(row["item_idx"]) for row in rows]], device=trajectory.device)
        gap = 0 if trajectory.native.last_timestamp is None else timestamp-trajectory.native.last_timestamp
        delta = torch.as_tensor([gap], device=trajectory.device, dtype=torch.float32)
        # Graph outputs are borrowed. Convert each result before the next replay.
        scores = trajectory.score(ids, delta)[0].detach().cpu().tolist()
        output.extend(dict(row, method=TRAINED_FG_METHOD, score=float(score))
            for row, score in zip(rows, scores, strict=True))
    return output


@torch.no_grad()
def replay_fg_user(parent, current, fg_probe, events, requests_by_timestamp, uid, cfg, *,
                    pool=None, initial=None, capture=True, stop_after_last_query=True):
    """E5 FG replay with the same actual-time query batching and own writes."""
    if fg_probe.stage != "fg":
        raise ValueError("own-trajectory E5 FG replay requires stage fg")
    initial = _initial(parent, events, cfg, initial)
    pool = pool or ProjectedGraphPool(current, fg_probe,
        window=int(cfg.get("history_length", 1024)), capture=capture)
    trajectory = ProjectedFastProbeTrajectory(parent, current, fg_probe, events, cfg, pool=pool, initial=initial)
    rows = []
    for timestamp, requests in sorted(requests_by_timestamp.items()):
        rows.extend(score_fg_timestamp(trajectory, requests, uid=uid, timestamp=timestamp,
            query_batch=int(cfg.get("query_batch", 16))))
    if not stop_after_last_query:
        trajectory.advance(int(cfg["end_day"])*86400)
    report = trajectory.report()
    report.update(method=TRAINED_FG_METHOD, independent_state=True,
        unscored_suffix_advanced=not stop_after_last_query,
        scope="independently trained E5 FG trajectory; every actual E14 feedback and all causal writes before it")
    return rows, report
