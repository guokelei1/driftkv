#!/usr/bin/env python3
"""Fixed-fit256 candidate/time calibration views; CPU inputs, no labels or GPU."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]

import duckdb
import numpy as np
import pyarrow as pa
import torch

from experiments.ch3_motivation.code.calibration_gaps import (
    RULES, TIME_RULE, STALE_FIELDS, _rng, _week_files, fit_gap_law,
)
from experiments.ch3_motivation.code.common import DATASET, write_json
from experiments.ch3_motivation.code.prepare_candidates import candidates_for
from hstu_kvcache.common.artifacts import sha256

FRACTIONS = (.5, .75)


def digest(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def view_key(fraction, rule):
    if fraction not in FRACTIONS or rule not in RULES:
        raise ValueError("only Q16 feedback50/75 and the three declared gap rules are supported")
    return f"feedback{round(100 * fraction)}_{rule}"


def make_sampling_views(fit, validation, candidates, release_gaps, law, *, edge, cutoff, seed=17):
    """All272 targets are retained; the law and candidate catalog stay fit256."""
    if (len(fit) != 256 or len(validation) != 16 or len(set(fit + validation)) != 272
            or candidates["fitting_pool_uids"] != fit or candidates["validation_uids"] != validation
            or not set(law["users"]) <= set(fit) or law["coverage"]["fit_users"] != 256
            or law["fallback_used"]):
        raise ValueError("sampling requires the original disjoint fit256/validation16 and fit-only feedback law")
    known = int(candidates["known_items"])
    views = {view_key(fraction, rule):dict(feedback_fraction=fraction, gap_rule=rule, users={})
             for fraction in FRACTIONS for rule in RULES}
    for uid in fit + validation:
        recent = list(dict.fromkeys(int(item) for item in candidates["recent_items"][str(uid)] if 1 <= int(item) <= known))
        if len(recent) < 12:
            raise ValueError(f"UID{uid} cannot supply twelve unique pooled feedback candidates; do not drop it")
        empirical = []
        for index in range(16):
            rng = _rng("sampling256_gap_bank", seed, edge, uid, index)
            donor = law["users"][int(rng.integers(len(law["users"])))]
            pool = law["observations"][str(donor)]
            empirical.append(dict(pool[int(rng.integers(len(pool)))]))
        for fraction in FRACTIONS:
            quota = int(16 * fraction)
            ids = candidates_for(uid, recent, known, count=16, recent_fraction=fraction, seed=seed)
            if ids[:quota] != recent[:quota] or len(set(ids)) != 16:
                raise ValueError("feedback prefix or unique candidate order differs")
            keep = set()
            for first, count in ((0, quota), (quota, 16 - quota)):
                keep.update((first + _rng("sampling256_mix_mask", seed, edge, uid, quota, first).permutation(count)[:count // 2]).tolist())
            for rule in RULES:
                assignments = []
                for index, item in enumerate(ids):
                    original = rule == "cutover" or rule == "gap_mix50" and index in keep
                    source = dict(gap_seconds=float(release_gaps[uid]), source="cutover_release_gap", source_uid=int(uid),
                        source_feedback_timestamp=None, source_view_timestamp=int(cutoff),
                        source_previous_behavior_timestamp=int(cutoff - release_gaps[uid])) if original else empirical[index]
                    assignments.append(dict(candidate_index=index, item_idx=item,
                        candidate_type="feedback" if index < quota else "uniform", **source))
                views[view_key(fraction, rule)]["users"][str(uid)] = dict(role="fit" if uid in fit else "validation",
                    release_gap_seconds=float(release_gaps[uid]), candidate_ids=ids, assignments=assignments)
    for view in views.values():
        view["users_sha256"] = digest(view["users"])
    return views


def validate_sampling_views(manifest, candidates):
    """Causality, exact mixture counts and fixed ordering for all six views."""
    fit, validation = manifest["target_fit_uids"], manifest["validation_uids"]
    cutoff = manifest["source_cutoff"]
    if (manifest["candidate_count"] != 16 or manifest["candidate_seed"] != 17 or manifest["gap_seed"] != 17
            or manifest["catalog_source_users"] != 256 or manifest["time_law_source_users"] != 256
            or manifest["time_law_source_fit_uids"] != fit or manifest["candidate_catalog_source_fit_uids"] != fit
            or candidates["fitting_pool_uids"] != fit or candidates["validation_uids"] != validation
            or len(fit) != 256 or len(validation) != 16 or len(set(fit + validation)) != 272
            or set(manifest["views"]) != {view_key(f, r) for f in FRACTIONS for r in RULES}):
        raise ValueError("fixed256 law/catalog/target roles or six-view schema differ")
    observed = set()
    law = manifest["gap_law"]
    if law["fallback_used"] or law["coverage"]["fit_users"] != 256 or not set(law["users"]) <= set(fit):
        raise ValueError("time law includes another role or a fallback")
    for uid, observations in law["observations"].items():
        if int(uid) not in fit:
            raise ValueError("validation/evaluation user contributed to the time law")
        for row in observations:
            timestamp, previous = row["source_feedback_timestamp"], row["source_previous_behavior_timestamp"]
            if (row["source_uid"] != int(uid) or row["source"] != "fit_pre_release_feedback"
                    or not cutoff - 14 * 86400 <= timestamp < cutoff
                    or previous is not None and not previous < timestamp
                    or row["gap_seconds"] != (timestamp - previous if previous is not None else 0)):
                raise ValueError("empirical law is not a strict-past pre-release feedback gap")
            observed.add(digest(row))
    for fraction in FRACTIONS:
        quota = int(16 * fraction)
        for rule in RULES:
            view = manifest["views"][view_key(fraction, rule)]
            if (view["feedback_fraction"] != fraction or view["gap_rule"] != rule
                    or view["users_sha256"] != digest(view["users"])
                    or set(view["users"]) != {str(uid) for uid in fit + validation}):
                raise ValueError("sampling user coverage, view identity or hash differs")
            for uid in fit + validation:
                user = view["users"][str(uid)]
                ids = candidates_for(uid, candidates["recent_items"][str(uid)], candidates["known_items"],
                    count=16, recent_fraction=fraction, seed=17)
                baseline = manifest["views"][view_key(fraction, "cutover")]["users"][str(uid)]
                rows = user["assignments"]
                if (user["candidate_ids"] != ids or user["candidate_ids"] != baseline["candidate_ids"]
                        or len(set(ids)) != 16 or len(rows) != 16
                        or [row["candidate_index"] for row in rows] != list(range(16))
                        or [row["item_idx"] for row in rows] != ids
                        or user["role"] != ("fit" if uid in fit else "validation")):
                    raise ValueError("candidate IDs/order/role changed across time rules")
                cutover_counts = dict(feedback=0, uniform=0)
                for index, row in enumerate(rows):
                    kind = "feedback" if index < quota else "uniform"
                    if row["candidate_type"] != kind or not np.isfinite(row["gap_seconds"]) or row["gap_seconds"] < 0:
                        raise ValueError("candidate quota/type or finite nonnegative gap differs")
                    source = {key:value for key,value in row.items() if key not in ("candidate_index", "item_idx", "candidate_type")}
                    if row["source"] == "cutover_release_gap":
                        cutover_counts[kind] += 1
                        if (row["source_uid"] != uid or row["source_view_timestamp"] != cutoff
                                or row["source_previous_behavior_timestamp"] >= cutoff
                                or row["gap_seconds"] != cutoff - row["source_previous_behavior_timestamp"]
                                or row["gap_seconds"] != user["release_gap_seconds"]):
                            raise ValueError("cutover gap differs from the frozen strict-past history")
                    elif digest(source) not in observed:
                        raise ValueError("empirical assignment is not in the fitting-only time law")
                wanted = dict(feedback=quota, uniform=16 - quota) if rule == "cutover" else (
                    dict(feedback=quota // 2, uniform=(16 - quota) // 2) if rule == "gap_mix50" else dict(feedback=0, uniform=0))
                if cutover_counts != wanted:
                    raise ValueError("mixed timing must retain exactly half within each candidate source")
    return dict(users_per_view=272, views=6, all_candidate_shapes=[1, 16],
        candidate_ids_shared_across_time_rules=True, feedback50_quota=8, feedback75_quota=12,
        mix50_cutover_counts=dict(feedback50=[4, 4], feedback75=[6, 2]),
        validation_contributes_to_shared_catalog_or_time_law=False, all_source_timestamps_causal=True)


def prepare_edge(input_root, edge, output, *, threads=4, hash_cache=None):
    started = time.perf_counter()
    input_root, output = Path(input_root), Path(output)
    if output.exists():
        raise ValueError("keep existing sampling inputs; use a new attempt")
    population_path = input_root / "population.json"
    population = json.loads(population_path.read_text())
    fit, validation = population["fit"][:256], population["validation"]
    folder = input_root / f"edge_{edge}"
    binding_path, candidate_path = folder / "inputs.json", folder / "calibration_feedback_candidates.json"
    binding, candidates = json.loads(binding_path.read_text()), json.loads(candidate_path.read_text())
    cutoff = int(binding["settings"]["release_day"]) * 86400
    if (binding["population_sha256"] != sha256(population_path) or candidates["population_sha256"] != sha256(population_path)
            or candidates["dataset_sha256"] != sha256(DATASET) or candidates["release_day"] * 86400 != cutoff
            or binding["settings"]["edge"] != edge):
        raise ValueError("sampling source population, dataset or release differs")
    dataset = json.loads(DATASET.read_text())
    feedback_files = _week_files((DATASET.parent / dataset["shared_feedback_glob"]).resolve(), cutoff - 14 * 86400, cutoff)
    behavior_files = _week_files((DATASET.parent / dataset["shared_listens_glob"]).resolve(), 0, cutoff)
    cache = {} if hash_cache is None else hash_cache
    for name in feedback_files + behavior_files:
        if name not in cache:
            cache[name] = sha256(Path(name))
    connection = duckdb.connect()
    connection.execute(f"PRAGMA threads={int(threads)}")
    connection.register("selected_fit", pa.table({"uid":pa.array(fit, type=pa.uint64())}))
    feedback_sql = "SELECT uid,timestamp FROM read_parquet(?,hive_partitioning=true) SEMI JOIN selected_fit USING(uid) WHERE timestamp>=? AND timestamp<? ORDER BY uid,timestamp"
    behavior_sql = "SELECT uid,timestamp FROM read_parquet(?,hive_partitioning=true) SEMI JOIN selected_fit USING(uid) WHERE timestamp<? ORDER BY uid,timestamp"
    feedback, behavior = defaultdict(list), defaultdict(list)
    for uid, timestamp in connection.execute(feedback_sql, [feedback_files, cutoff - 14 * 86400, cutoff]).fetchall():
        feedback[int(uid)].append(int(timestamp))
    for uid, timestamp in connection.execute(behavior_sql, [behavior_files, cutoff]).fetchall():
        behavior[int(uid)].append(int(timestamp))
    connection.close()
    law = fit_gap_law(feedback, behavior, fit, cutoff, allow_behavior_fallback=False)
    history_path = folder / binding["history"]["path"]
    if sha256(history_path) != binding["history"]["sha256"]:
        raise ValueError("frozen sampling history changed")
    with np.load(history_path, allow_pickle=False) as arrays:
        positions = {int(uid):index for index,uid in enumerate(arrays["uids"])}
        offsets, timestamps = arrays["offsets"], arrays["timestamps"]
    release_gaps = {}
    for uid in fit + validation:
        index = positions[uid]
        times = timestamps[int(offsets[index]):int(offsets[index + 1])]
        before = int(np.searchsorted(times, cutoff, side="left"))
        if not before:
            raise ValueError("calibration source has an empty pre-release prefix")
        previous = int(times[before - 1])
        if uid in fit and (not behavior[uid] or previous != behavior[uid][-1]):
            raise ValueError("raw and frozen-pack behavior predecessors differ")
        release_gaps[uid] = cutoff - previous
    manifest = dict(schema_version=1, status="prepared", edge=edge, source_cutoff=cutoff, release_day=cutoff // 86400,
        candidate_count=16, candidate_seed=17, gap_seed=17, catalog_source_users=256, time_law_source_users=256,
        candidate_catalog_source_fit_uids=fit, time_law_source_fit_uids=fit, target_fit_uids=fit, validation_uids=validation,
        time_rule=TIME_RULE, candidate_order="fixed pooled-fit256 feedback prefix then unique uniform known items; Q16 feedback50=8+8,feedback75=12+4; seed17; identical IDs across time rules",
        views=make_sampling_views(fit, validation, candidates, release_gaps, law, edge=edge, cutoff=cutoff), gap_law=law,
        sourcebinding=dict(input_root=str(input_root.resolve()), population_path=str(population_path.resolve()),
            population_sha256=sha256(population_path), path=str(binding_path.resolve()), sha256=sha256(binding_path),
            candidate_path=str(candidate_path.resolve()), candidate_sha256=sha256(candidate_path),
            history_path=str(history_path.resolve()), history_sha256=binding["history"]["sha256"]),
        raw_sources=dict(dataset_path=str(DATASET),dataset_sha256=sha256(DATASET),
            feedback_files={name:cache[name] for name in feedback_files},behavior_files={name:cache[name] for name in behavior_files},
            feedback_sql=feedback_sql,behavior_sql=behavior_sql,projected_columns=["uid", "timestamp"],selected_uids=fit,
            feedback_predicate="cutoff-14days<=timestamp<cutoff",behavior_predicate="timestamp<cutoff; strict predecessor<t_feedback",
            label_columns_projected=False,label_values_used=False,validation_contributes_to_law=False),
        source_sha256={str(path.resolve().relative_to(ROOT)):sha256(path) for path in
            (Path(__file__),Path(__file__).with_name("calibration_gaps.py"),Path(__file__).with_name("prepare_candidates.py"))},
        scope="CPU calibration input coverage exploration; law/catalog fixedfit256 for every gradient N; no future feedback labels, predictions, or quality selection")
    manifest["validation"] = validate_sampling_views(manifest, candidates)
    manifest["cpu_seconds"] = dict(total=time.perf_counter() - started)
    write_json(output, manifest)
    return manifest


def load_sampling_manifest(path, input_root, edge, feedback_fraction, gap_rule, *, fit_users=256):
    path, input_root = Path(path), Path(input_root)
    if fit_users not in (64, 128, 256):
        raise ValueError("gradient users must be the fixed64/128/256 prefix")
    value = json.loads(path.read_text())
    if value["schema_version"] != 1 or value["status"] != "prepared" or value["edge"] != edge:
        raise ValueError("sampling manifest schema/status/edge differs")
    binding = value["sourcebinding"]
    if Path(binding["input_root"]).resolve() != input_root.resolve():
        raise ValueError("sampling view must use the original fixed-fit256 input root")
    for name, expected in ((input_root / "population.json",binding["population_sha256"]),
            (input_root / f"edge_{edge}/inputs.json",binding["sha256"]),
            (input_root / f"edge_{edge}/calibration_feedback_candidates.json",binding["candidate_sha256"]),
            (Path(binding["history_path"]),binding["history_sha256"])):
        if sha256(name) != expected:
            raise ValueError("sampling population/input/candidate/history binding changed")
    for name, expected in value["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError("sampling preparation source changed")
    population = json.loads((input_root / "population.json").read_text())
    candidates = json.loads((input_root / f"edge_{edge}/calibration_feedback_candidates.json").read_text())
    settings = json.loads((input_root / f"edge_{edge}/inputs.json").read_text())["settings"]
    if (value["target_fit_uids"] != population["fit"][:256] or value["validation_uids"] != population["validation"]
            or value["source_cutoff"] != settings["release_day"] * 86400):
        raise ValueError("sampling target role prefix or release cutoff differs")
    validate_sampling_views(value, candidates)
    key = view_key(feedback_fraction, gap_rule)
    selected = value["target_fit_uids"][:fit_users] + value["validation_uids"]
    chosen = value["views"][key]
    views = {uid:dict(candidates=torch.tensor([chosen["users"][str(uid)]["candidate_ids"]],dtype=torch.long),
        query_delta=torch.tensor([[row["gap_seconds"] for row in chosen["users"][str(uid)]["assignments"]]],dtype=torch.float32)) for uid in selected}
    provenance = dict(manifest_path=str(path.resolve()),manifest_sha=sha256(path),view_key=key,
        feedback_fraction=feedback_fraction,gap_rule=gap_rule,time_rule=value["time_rule"],
        cutoff=value["source_cutoff"],source_cutoff=value["source_cutoff"],candidate_order=value["candidate_order"],
        candidate_ids={str(uid):chosen["users"][str(uid)]["candidate_ids"] for uid in selected},
        fit_uids=value["target_fit_uids"][:fit_users],validation_uids=value["validation_uids"],
        target_fit_uids=value["target_fit_uids"],time_law_source_fit_uids=value["time_law_source_fit_uids"],
        candidate_catalog_source_fit_uids=value["candidate_catalog_source_fit_uids"],catalog_source_users=256,time_law_source_users=256,
        candidate_seed=17,gap_seed=17,candidate_count=16,cpu_seconds=value["cpu_seconds"],sourcebinding=binding,
        gap_law_coverage=value["gap_law"]["coverage"],view_users_sha256=chosen["users_sha256"])
    return views, provenance


def apply_sampling_view(samples, views, provenance):
    """Install CPU inputs BEFORE targets, input moments and loss scales are made."""
    if set(samples) != set(views) or set(samples) != set(provenance["fit_uids"] + provenance["validation_uids"]):
        raise ValueError("captured samples must contain exactly gradient fitN plus validation16")
    result = {}
    for uid, sample in samples.items():
        updated = {name:value for name,value in sample.items() if name not in STALE_FIELDS}
        updated["candidates"] = views[uid]["candidates"].detach().cpu().clone()
        updated["query_delta"] = views[uid]["query_delta"].detach().cpu().clone()
        if updated["candidates"].shape != (1,16) or updated["query_delta"].shape != (1,16):
            raise ValueError("sampling inputs must both have shape[1,16]")
        updated["query_delta_rule"] = provenance["gap_rule"]
        updated["sampling_manifest_sha256"] = provenance["manifest_sha"]
        updated["sampling_view_key"] = provenance["view_key"]
        result[uid] = updated
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root",type=Path,default=ROOT / "experiments/ch3_motivation/inputs_feedback_pool/fit256")
    parser.add_argument("--edges",type=int,nargs="+",default=[1,2,3,4,5])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--threads",type=int,default=4)
    args = parser.parse_args()
    cache, records = {}, []
    for edge in args.edges:
        output = args.output / f"edge_{edge}/sampling_manifest.json"
        value = prepare_edge(args.input_root,edge,output,threads=args.threads,hash_cache=cache)
        records.append(dict(edge=edge,path=str(output.resolve()),sha256=sha256(output),validation=value["validation"],
            law_coverage=value["gap_law"]["coverage"],cpu_seconds=value["cpu_seconds"]))
        print(json.dumps(records[-1]),flush=True)
    write_json(args.output / "summary.json",dict(status="complete",role="CPU-only fixed256 calibration sampling inputs",edges=records,
        candidate_catalog_source_users=256,time_law_source_users=256,targets_fit_users=256,validation_users=16,
        fractions=list(FRACTIONS),rules=list(RULES),candidate_count=16,seed=17,gpu_started=False))
