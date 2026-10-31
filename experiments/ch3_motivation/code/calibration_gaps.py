#!/usr/bin/env python3
"""Fit-only pre-release temporal calibration views; no labels or GPU work.

The gap law uses actual feedback timestamps and their strict-past behavior
predecessors from the fitting64 users only. Validation16 receives the same law.
These are synthetic query-delta features on fixed cutover caches, not future
query observations or chronological serving trajectories.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import defaultdict
import glob
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[3]

import duckdb
import numpy as np
import pyarrow as pa
import torch

from experiments.ch3_motivation.code.common import DATASET, write_json
from experiments.ch3_motivation.code.prepare_candidates import candidates_for
from hstu_kvcache.common.artifacts import sha256

RULES = ("cutover", "gap_mix50", "gap_empirical100")
TIME_RULE = "query_delta=t_query-last_behavior_strictly_before_t_query; no prior behavior gives0; raw seconds, no seven-day append clipping"
STALE_FIELDS = ("teacher_scores", "read_moment_sum", "score_moment_sum", "scale_users",
    "target_moment_version", "read_scales", "score_scale", "loss_scale_version")


def _rng(*parts):
    digest = hashlib.sha256("/".join(map(str, parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def fit_gap_law(feedback_times, behavior_times, fit_uids, cutoff, *, lookback=14*86400,
                allow_behavior_fallback=False):
    """Input-only CPU core; extra UIDs/future rows cannot enter the fit law."""
    by_user = {}
    before, no_prior = 0, 0
    for uid in fit_uids:
        times = np.sort(np.asarray(behavior_times.get(uid, []), dtype=np.int64))
        times = times[times < cutoff]
        observations = []
        for timestamp in sorted(feedback_times.get(uid, [])):
            timestamp = int(timestamp)
            if not cutoff-lookback <= timestamp < cutoff:
                continue
            position = int(np.searchsorted(times, timestamp, side="left"))
            previous = int(times[position-1]) if position else None
            no_prior += int(previous is None)
            before += 1
            observations.append(dict(source_uid=int(uid), source_feedback_timestamp=timestamp,
                source_previous_behavior_timestamp=previous,
                gap_seconds=timestamp-previous if previous is not None else 0,
                source="fit_pre_release_feedback"))
        if observations:
            by_user[int(uid)] = observations
    coverage = dict(fit_users=len(fit_uids), users_with_feedback_gaps=len(by_user),
        observed_feedback_instances=before, no_prior_behavior_instances=no_prior,
        feedback_users_without_observations=[int(uid) for uid in fit_uids if uid not in by_user])
    source = "fit_pre_release_feedback"
    if not by_user and allow_behavior_fallback:
        source = "fit_pre_release_behavior_gap_fallback"
        for uid in fit_uids:
            times = np.sort(np.asarray(behavior_times.get(uid, []), dtype=np.int64))
            for position in range(1, len(times)):
                timestamp = int(times[position])
                if cutoff-lookback <= timestamp < cutoff:
                    by_user.setdefault(int(uid), []).append(dict(source_uid=int(uid),
                        source_feedback_timestamp=None, source_behavior_timestamp=timestamp,
                        source_previous_behavior_timestamp=int(times[position-1]),
                        gap_seconds=timestamp-int(times[position-1]), source=source))
    if not by_user:
        raise ValueError("no fitting pre-release feedback gap observations; coverage="+json.dumps(coverage))
    return dict(source=source, weighting="uniform fitting UID among those with observations, then uniform observation within that UID",
        users=sorted(by_user), observations={str(uid): rows for uid, rows in by_user.items()}, coverage=coverage,
        fallback_used=source.endswith("fallback"))


def make_views(fit_uids, validation_uids, candidate_data, release_gaps, law, *, edge, cutoff, seed=17):
    """Preserve exact 8-feedback/8-uniform IDs; share empirical draws across rules."""
    if len(fit_uids) != 64 or len(validation_uids) != 16:
        raise ValueError("the gap diagnostic uses exactly fitting64 and validation16")
    if (candidate_data["fitting_pool_uids"] != fit_uids or candidate_data["validation_uids"] != validation_uids or
            set(fit_uids) & set(validation_uids)):
        raise ValueError("pooled candidate roles differ from fitting64/validation16")
    known = int(candidate_data["known_items"])
    law_users = law["users"]
    if not set(law_users) <= set(fit_uids):
        raise ValueError("only fitting UIDs may contribute to the empirical gap law")
    views = {}
    for uid in fit_uids+validation_uids:
        recent = candidate_data["recent_items"][str(uid)]
        admitted = list(dict.fromkeys(int(value) for value in recent if 1 <= int(value) <= known))
        if len(admitted) < 8:
            raise ValueError("pooled candidate sequence cannot supply eight feedback candidates")
        ids = candidates_for(uid, recent, known, count=16, recent_fraction=.5, seed=seed)
        if ids[:8] != admitted[:8]:
            raise ValueError("candidate ordering differs from the actual fitting runner")
        keep = set()
        for first in (0, 8):
            keep.update((first+_rng("gap_mix50_mask", seed, edge, uid, first).permutation(8)[:4]).tolist())
        empirical = []
        for index in range(16):
            generator = _rng("gap_bank", seed, edge, uid, index)
            source_uid = law_users[int(generator.integers(len(law_users)))]
            pool = law["observations"][str(source_uid)]
            empirical.append(dict(pool[int(generator.integers(len(pool)))]))
        rules = {}
        for rule in RULES:
            assignments = []
            for index, item in enumerate(ids):
                original = rule == "cutover" or rule == "gap_mix50" and index in keep
                source = dict(gap_seconds=float(release_gaps[uid]), source="cutover_release_gap", source_uid=int(uid),
                    source_feedback_timestamp=None, source_view_timestamp=int(cutoff),
                    source_previous_behavior_timestamp=int(cutoff-release_gaps[uid])) if original else empirical[index]
                assignments.append(dict(candidate_index=index, item_idx=item,
                    candidate_type="feedback" if index < 8 else "uniform", **source))
            rules[rule] = assignments
        views[str(uid)] = dict(role="fit" if uid in fit_uids else "validation", release_gap_seconds=float(release_gaps[uid]),
            candidate_ids=ids, rules=rules)
    return views


def _week_files(pattern, start, end):
    files = []
    for name in glob.glob(str(pattern), recursive=True):
        match = re.search(r"/week=(\d+)/", name)
        if match is None:
            raise ValueError("expected the actual weekly Yambda parquet layout")
        week = int(match.group(1))
        if start//604800 <= week <= (end-1)//604800:
            files.append(str(Path(name).resolve()))
    if not files:
        raise ValueError("no source parquet files for the pre-release window")
    return sorted(files)


def prepare_edge(input_root, edge, output, *, threads=4, allow_behavior_fallback=False, hash_cache=None):
    started = time.perf_counter()
    input_root, output = Path(input_root), Path(output)
    population_path = input_root/"population.json"
    population = json.loads(population_path.read_text())
    fit, validation = population["fit"][:64], population["validation"]
    folder = input_root/f"edge_{edge}"
    binding_path, candidate_path = folder/"inputs.json", folder/"calibration_feedback_candidates.json"
    binding, candidates = json.loads(binding_path.read_text()), json.loads(candidate_path.read_text())
    if binding["population_sha256"] != sha256(population_path) or candidates["population_sha256"] != sha256(population_path):
        raise ValueError("frozen population binding changed")
    cutoff = int(binding["settings"]["release_day"])*86400
    dataset = json.loads(DATASET.read_text())
    if candidates["dataset_sha256"] != sha256(DATASET) or candidates["release_day"]*86400 != cutoff:
        raise ValueError("candidate dataset/cutover binding differs")
    feedback_files = _week_files((DATASET.parent/dataset["shared_feedback_glob"]).resolve(), cutoff-14*86400, cutoff)
    behavior_files = _week_files((DATASET.parent/dataset["shared_listens_glob"]).resolve(), 0, cutoff)
    hash_cache = {} if hash_cache is None else hash_cache
    hashing = time.perf_counter()
    for name in feedback_files+behavior_files:
        if name not in hash_cache:
            hash_cache[name] = sha256(Path(name))
    raw_hash_seconds = time.perf_counter()-hashing
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={int(threads)}")
    con.register("selected_fit", pa.table({"uid": pa.array(fit, type=pa.uint64())}))
    feedback_sql = "SELECT uid,timestamp FROM read_parquet(?,hive_partitioning=true) SEMI JOIN selected_fit USING(uid) WHERE timestamp>=? AND timestamp<? ORDER BY uid,timestamp"
    behavior_sql = "SELECT uid,timestamp FROM read_parquet(?,hive_partitioning=true) SEMI JOIN selected_fit USING(uid) WHERE timestamp<? ORDER BY uid,timestamp"
    querying = time.perf_counter()
    feedback_rows = con.execute(feedback_sql, [feedback_files, cutoff-14*86400, cutoff]).fetchall()
    behavior_rows = con.execute(behavior_sql, [behavior_files, cutoff]).fetchall()
    con.close()
    raw_query_seconds = time.perf_counter()-querying
    feedback, behavior = defaultdict(list), defaultdict(list)
    for uid, timestamp in feedback_rows:
        feedback[int(uid)].append(int(timestamp))
    for uid, timestamp in behavior_rows:
        behavior[int(uid)].append(int(timestamp))
    law_started = time.perf_counter()
    law = fit_gap_law(feedback, behavior, fit, cutoff, allow_behavior_fallback=allow_behavior_fallback)
    law_seconds = time.perf_counter()-law_started
    history_path = folder/binding["history"]["path"]
    if sha256(history_path) != binding["history"]["sha256"]:
        raise ValueError("calibration history pack changed")
    with np.load(history_path, allow_pickle=False) as arrays:
        uids, offsets, times = arrays["uids"], arrays["offsets"], arrays["timestamps"]
    positions = {int(uid): index for index, uid in enumerate(uids)}
    release_gaps = {}
    for uid in fit+validation:
        i = positions[uid]
        user_times = times[int(offsets[i]):int(offsets[i+1])]
        before = int(np.searchsorted(user_times, cutoff, side="left"))
        if not before:
            raise ValueError("actual calibration requires a nonempty pre-release prefix")
        previous = int(user_times[before-1])
        if uid in fit and (not behavior[uid] or previous != behavior[uid][-1]):
            raise ValueError("raw behavior predecessor differs from the frozen calibration prefix")
        release_gaps[uid] = cutoff-previous
    sampling_started = time.perf_counter()
    views = make_views(fit, validation, candidates, release_gaps, law, edge=edge, cutoff=cutoff)
    sampling_seconds = time.perf_counter()-sampling_started
    result = dict(schema_version=1, status="prepared", edge=edge, release_day=cutoff//86400,
        source_cutoff=cutoff, lookback_start=cutoff-14*86400, lookback_days=14,
        rules=list(RULES), time_rule=TIME_RULE, fit_uids=fit, validation_uids=validation,
        candidate_order="exact existing candidates_for(uid,pooledfit64_recent,known,count16,recent_fraction0.5,seed17); indices0..7 feedback,8..15 uniform",
        candidate_seed=17, gap_seed=17, candidate_count=16, users=views, gap_law=law,
        input_binding=dict(input_root=str(input_root.resolve()), population_path=str(population_path.resolve()),
            population_sha256=sha256(population_path), path=str(binding_path.resolve()), sha256=sha256(binding_path),
            candidate_path=str(candidate_path.resolve()), candidate_sha256=sha256(candidate_path),
            history_path=str(history_path.resolve()), history_sha256=binding["history"]["sha256"]),
        sources=dict(dataset_path=str(DATASET), dataset_sha256=sha256(DATASET),
            feedback_files={name: hash_cache[name] for name in feedback_files},
            behavior_files={name: hash_cache[name] for name in behavior_files},
            feedback_sql=feedback_sql, behavior_sql=behavior_sql,
            projected_columns=dict(feedback=["uid", "timestamp"], behavior=["uid", "timestamp"]),
            selected_uids=fit, feedback_timestamp_predicate="lookback_start<=timestamp<source_cutoff",
            behavior_timestamp_predicate="timestamp<source_cutoff; predecessor selected with searchsorted(side=left)",
            validation_feedback_read=False, label_columns_projected=False, label_values_used=False,
            source_sha256={str(Path(__file__).resolve().relative_to(ROOT)):sha256(Path(__file__)),
                "experiments/ch3_motivation/code/prepare_candidates.py":sha256(Path(__file__).with_name("prepare_candidates.py"))}),
        cpu_seconds=dict(raw_source_hashing=raw_hash_seconds, raw_timestamp_query=raw_query_seconds,
            fit_law=law_seconds, assignments=sampling_seconds, total=time.perf_counter()-started),
        scope="pre-release fitting64-only synthetic temporal input views; validation16 applies the same fitting law; no future query/feedback or quality claim")
    if output.exists():
        raise ValueError("retain an existing gap manifest; use a new output attempt")
    write_json(output, result)
    print(json.dumps(dict(edge=edge,output=str(output),coverage=law["coverage"],fallback_used=law["fallback_used"],seconds=result["cpu_seconds"])),flush=True)
    return result


def load_gap_manifest(path, input_root, edge, rule):
    path, input_root = Path(path), Path(input_root)
    value = json.loads(path.read_text())
    if value["schema_version"] != 1 or value["status"] != "prepared" or value["edge"] != edge or rule not in RULES:
        raise ValueError("gap manifest schema/edge/rule differs")
    binding = value["input_binding"]
    if Path(binding["input_root"]).resolve() != input_root.resolve():
        raise ValueError("gap view uses another pooled input root")
    current = dict(population_path=input_root/"population.json",path=input_root/f"edge_{edge}/inputs.json",
        candidate_path=input_root/f"edge_{edge}/calibration_feedback_candidates.json",history_path=Path(binding["history_path"]))
    for name, file in current.items():
        expected = binding[{"population_path":"population_sha256","path":"sha256","candidate_path":"candidate_sha256","history_path":"history_sha256"}[name]]
        if sha256(file) != expected:
            raise ValueError("gap manifest raw input/candidate binding changed")
    cfg = json.loads(current["path"].read_text())["settings"]
    if value["source_cutoff"] != cfg["release_day"]*86400:
        raise ValueError("gap manifest release cutoff differs")
    for name, digest in value["sources"]["source_sha256"].items():
        if sha256(ROOT/name) != digest:
            raise ValueError("gap preparation source changed")
    gap_views, ids = {}, {}
    for uid, user in value["users"].items():
        assignments = user["rules"][rule]
        if [entry["candidate_index"] for entry in assignments] != list(range(16)) or [entry["item_idx"] for entry in assignments] != user["candidate_ids"]:
            raise ValueError("gap manifest candidate order changed")
        gaps = torch.tensor([[entry["gap_seconds"] for entry in assignments]], dtype=torch.float32)
        if not torch.isfinite(gaps).all() or (gaps < 0).any():
            raise ValueError("gap view has invalid raw query deltas")
        gap_views[int(uid)] = gaps
        ids[uid] = user["candidate_ids"]
    if set(gap_views) != set(value["fit_uids"]+value["validation_uids"]):
        raise ValueError("gap view user coverage differs")
    provenance = dict(rule=rule, manifest_path=str(path.resolve()), manifest_sha256=sha256(path),
        time_rule=value["time_rule"], source_cutoff=value["source_cutoff"], release_day=value["release_day"],
        candidate_order=value["candidate_order"], candidate_ids=ids,
        fit_uids=value["fit_uids"], validation_uids=value["validation_uids"],
        input_binding=binding, cpu_seconds=value["cpu_seconds"],
        gap_law_source=value["gap_law"]["source"], gap_law_coverage=value["gap_law"]["coverage"],
        fallback_used=value["gap_law"]["fallback_used"], scope=value["scope"])
    return gap_views, provenance


def apply_gap_view(samples, gap_views, gap_provenance):
    """Return shallow sample copies with new CPU deltas and invalidated targets."""
    if set(samples) != set(gap_views):
        raise ValueError("captured calibration user coverage differs from gap manifest")
    result = {}
    for uid, sample in samples.items():
        if sample["candidates"].detach().cpu().tolist() != [gap_provenance["candidate_ids"][str(uid)]]:
            raise ValueError("captured candidate IDs/order differ from frozen gap view")
        updated = {name: value for name, value in sample.items() if name not in STALE_FIELDS}
        updated["query_delta"] = gap_views[uid].detach().cpu().clone()
        updated["query_delta_rule"] = gap_provenance["rule"]
        updated["gap_manifest_sha256"] = gap_provenance["manifest_sha256"]
        result[uid] = updated
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root",type=Path,default=ROOT/"experiments/ch3_motivation/inputs_feedback_pool/fit64")
    parser.add_argument("--edges",type=int,nargs="+",default=[1,2,3,4,5])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--allow-behavior-fallback",action="store_true")
    args=parser.parse_args()
    cache={}
    for edge in args.edges:
        prepare_edge(args.input_root,edge,args.output/f"edge_{edge}/gap_manifest.json",
            threads=args.threads,allow_behavior_fallback=args.allow_behavior_fallback,hash_cache=cache)
