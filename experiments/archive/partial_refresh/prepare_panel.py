#!/usr/bin/env python3
"""Prepare a label-free fresh-user panel for an already frozen refresh policy.

CPU only. Preserve the existing model chain, request identities, timestamp
ordering, item mapping, and query-before-same-time-event convention. No model
weights or prediction outcomes are read by selection.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from hstu_kvcache.evaluation.history import load_histories_semijoin, restore_pack
from hstu_kvcache.evaluation.panels import checked, record
from hstu_kvcache.data.yambda_history import load_yambda_histories
from hstu_kvcache.common.artifacts import DAY, sha256, write_json

REFERENCE = ROOT / "results/design_system_2026_10/phase2/representative4096/panel"
OUTPUT = ROOT / "results/partial_refresh_2026_10/fresh1024_v2/panel"
NAMESPACE = "partial_refresh_fresh_2026_10"
START, END = 259 * DAY, 301 * DAY


def select_users(reference, count):
    population_record = reference["selection"]["population"]
    population = set(pq.read_table(checked(population_record), columns=["uid"])["uid"].to_pylist())
    exclusions, sources = {}, {}

    def add(path, mapping, keys):
        path = Path(path)
        source = record(path)
        sources[str(path.resolve())] = source
        for key in keys:
            exclusions[f"{path.relative_to(ROOT)}:{key}"] = set(map(int, mapping[key]))

    original_path = ROOT / "configs/design/medium_v0_v5_users.json"
    original = json.loads(original_path.read_text())
    # Every legacy calibration/reservation plus all old development groups:
    # this strict pool has enough users and avoids outcome-conditioned legacy
    # Motivation/evaluation membership as well as current development overlap.
    add(original_path, original, list(original))
    for relative in (
        "results/design_system_2026_10/phase2/representative4096/panel/users.json",
        "results/design_two_2026_10/benchmark_panel/users.json",
        "results/design_system_2026_10/phase2/benchmark_panel/users.json",
        "results/design_two_2026_10/confirmation/panel/users.json",
    ):
        path = ROOT / relative
        groups = json.loads(path.read_text())
        add(path, groups, ["evaluation", "fit", "validation"])
    for source in reference["selection"]["exclusion_sources"]:
        path = checked(source)
        roles = json.loads(path.read_text())
        add(path, roles, ["fit_uids", "validation_uids"])
    # Reservations were expanded in historical v2 Q/H calibration. Include
    # those retained lists as well as all scales of the same Yambda population.
    for path in sorted((ROOT / "results/read_correction_2026_09").rglob("calibration_users.json")):
        add(path, json.loads(path.read_text()), ["uids"])
    for path in sorted((ROOT / "results/selective_recompute_2026_09/panels").glob("*/*/users.json")):
        add(path, json.loads(path.read_text()), ["calibration", "canary"])
    for relative in ("results/design_two_2026_10/coordination/data/binding.json",
                     "results/design_two_2026_10/coordination/data_fit1024/binding.json"):
        path = ROOT / relative
        roles = json.loads(path.read_text())
        add(path, roles["groups"], ["fit", "validation"])
    excluded = set().union(*exclusions.values())
    eligible = population - excluded
    print(json.dumps(dict(status="selection_pool", population_users=len(population),
        excluded_population_users=len(population & excluded), eligible_users=len(eligible),
        requested_users=count)), flush=True)
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} eligible users; cannot select {count}")
    rank = lambda uid: (hashlib.sha256(f"{NAMESPACE}/{uid}".encode()).hexdigest(), uid)
    chosen = sorted(eligible, key=rank)[:count]
    chosen_set = set(chosen)
    selection = dict(namespace=NAMESPACE, population=population_record,
        population_users=len(population), eligible_users=len(eligible),
        excluded_population_users=len(population & excluded), users=len(chosen),
        selected_uids=chosen,
        rule="First SHA256(namespace/uid), then UID, after all named frozen role/development/confirmation exclusions. No activity, request availability, labels, scores or Full/Reuse gap filter.",
        uses_labels_or_outcomes=False, retain_inactive_users=True,
        exclusion_sources=list(sources.values()),
        excluded_roles={name:len(members) for name,members in exclusions.items()},
        selected_overlap_by_excluded_role={name:len(chosen_set & members)
                                          for name,members in exclusions.items()},
        excluded_population_uid_sha256=hashlib.sha256(
            np.asarray(sorted(population & excluded), dtype="<i8").tobytes()).hexdigest(),
        scope="Fresh users relative to listed prior research roles. Fixed previously trained models and frozen refresh policy; no dataset/seed replication.")
    return chosen, selection, exclusions


def write_cohorts(panel_dir):
    """Fixed pre-release length/activity boundaries, using no labels or scores."""
    binding = json.loads((panel_dir / "binding.json").read_text())
    groups = json.loads(checked(binding["users_file"]).read_text())
    history = restore_pack(checked(binding["files"]["history"]))
    cohorts = {name:[] for name in ("history_le256","history_257_to768","history_gt768",
                                   "past72h_inactive","past72h_active")}
    for uid in groups["evaluation"]:
        timestamps = history.rows[uid][0]
        length = int(np.searchsorted(timestamps, binding["cutover"], side="left"))
        recent = length - int(np.searchsorted(timestamps, binding["cutover"]-3*DAY, side="left"))
        name = "history_le256" if length <=256 else "history_257_to768" if length<=768 else "history_gt768"
        cohorts[name].append(uid)
        cohorts["past72h_active" if recent else "past72h_inactive"].append(uid)
    write_json(panel_dir / "cohorts.json", cohorts)
    write_json(panel_dir / "cohort_definitions.json", dict(
        definitions=dict(history="Initial retained length at first cutover; bins <=256, 257..768, >768.",
            activity="Any retained event during [cutover-72h,cutover); inactive vs active. Counts can be capped by retained context, but zero/nonzero membership is unaffected."),
        cutover=binding["cutover"], labels_or_scores_used=False,
        users_file=binding["users_file"], history=binding["files"]["history"],
        cohorts=record(panel_dir/"cohorts.json"),
        users_by_cohort={name:len(uids) for name,uids in cohorts.items()},
        preparation_source=record(Path(__file__))))
    print(json.dumps(dict(status="cohorts_complete", output=str(panel_dir/"cohorts.json"),
                         users_by_cohort={name:len(uids) for name,uids in cohorts.items()})),flush=True)


def main(args):
    began = time.perf_counter()
    if args.cohorts_only:
        write_cohorts(args.output)
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    reference_path = args.reference / "binding.json"
    reference = json.loads(reference_path.read_text())
    original_groups = json.loads(checked(reference["users_file"]).read_text())
    chosen, selection, exclusions = select_users(reference, args.users)
    if args.selection_only:
        print(json.dumps(selection), flush=True)
        return
    groups = dict(evaluation=chosen, pilot=chosen[:8], fit=original_groups["fit"],
                  validation=original_groups["validation"])
    if set(chosen) & set(groups["fit"] + groups["validation"]):
        raise RuntimeError("fresh evaluation overlaps calibration")
    fidelity_path = ROOT / "data/manifests/yambda500m_medium_hstu_native_d7_d14_v1/requests_fidelity.parquet"
    filters = [("time_block", "=", "matrix_horizon"), ("target_known", "=", True),
               ("query_timestamp", ">=", START), ("query_timestamp", "<", END),
               ("uid", "in", chosen)]
    keys = ["request_id", "uid", "query_timestamp", "item_idx"]
    requests = pq.read_table(fidelity_path, columns=keys, filters=filters)
    labels_path = checked(reference["sources"]["labels"])
    labels = pq.read_table(labels_path, columns=[*keys, "label"], filters=filters)
    ids = requests["request_id"].to_pylist()
    if len(set(ids)) != len(ids):
        raise RuntimeError("manifest request identities repeat")
    requests = requests.join(labels, keys=keys, join_type="inner")
    if len(requests) != len(ids) or len(set(requests["request_id"].to_pylist())) != len(ids):
        raise RuntimeError("every request must match exactly one retained label and item")
    if not np.isin(requests["label"].to_numpy(), [0, 1]).all():
        raise RuntimeError("nonbinary request label")
    times = requests["query_timestamp"].to_numpy()
    windows = reference["windows"]
    window_index = np.searchsorted([row["end_timestamp"] for row in windows], times, side="right")
    requests = requests.append_column("edge", pa.array([windows[i]["edge"] for i in window_index]))
    requests = requests.append_column("version", pa.array([windows[i]["current"] for i in window_index]))
    requests = requests.select(["request_id", "uid", "item_idx", "query_timestamp", "label", "edge", "version"])
    requests = requests.sort_by([("uid", "ascending"), ("query_timestamp", "ascending"),
                                  ("request_id", "ascending")])
    print(json.dumps(dict(status="requests_loaded", users=len(chosen), requests=len(requests),
                         seconds=time.perf_counter()-began)), flush=True)
    dataset_path = checked(reference["sources"]["dataset"])
    kwargs = dict(dataset_path=dataset_path, known_vocab_size=reference["known_vocab_size"],
        oov_buckets=reference["oov_buckets"], start_timestamp=START, end_timestamp=END,
        max_history=1024, threads=args.threads)
    history = load_histories_semijoin(chosen, **kwargs)
    empty = (np.empty(0, dtype=np.int64),) * 3
    for uid in chosen:
        history.rows.setdefault(uid, empty)
    # One independent loader sample checks mapping and exact timestamp tie order.
    canaries = chosen[:2]
    independent = load_yambda_histories(dataset_path, canaries,
        known_vocab_size=reference["known_vocab_size"], oov_buckets=reference["oov_buckets"],
        start_timestamp=START, end_timestamp=END, max_pre_events=1024, threads=args.threads)
    for uid in canaries:
        if any(not np.array_equal(a,b) for a,b in zip(history.rows[uid], independent.rows.get(uid,empty), strict=True)):
            raise RuntimeError(f"independent history mapping/tie parity failed: {uid}")
    for uid,(timestamps,items,behaviors) in history.rows.items():
        if not (len(timestamps)==len(items)==len(behaviors)) or np.any(timestamps[1:]<timestamps[:-1]):
            raise RuntimeError(f"invalid history ordering: {uid}")
        if len(timestamps) and timestamps[-1] >= END:
            raise RuntimeError("history extends beyond evaluation window")
        if np.searchsorted(timestamps,START,side="left") > 1024:
            raise RuntimeError("initial history exceeds context")
    contemporaneous = 0
    for row in requests.to_pylist():
        timestamps = history.rows[int(row["uid"])][0]
        contemporaneous += int(np.searchsorted(timestamps,row["query_timestamp"],side="right")
                               - np.searchsorted(timestamps,row["query_timestamp"],side="left"))
    args.output.mkdir(parents=True)
    write_json(args.output / "users.json", groups)
    write_json(args.output / "excluded_roles.json", {name:sorted(uids)
                                                    for name,uids in exclusions.items()})
    selection["excluded_role_membership"] = record(args.output/"excluded_roles.json")
    write_json(args.output / "selection.json", selection)
    pq.write_table(requests, args.output / "requests.parquet", compression="zstd")
    ordered = sorted(chosen)
    lengths = np.asarray([len(history.rows[uid][0]) for uid in ordered], dtype=np.int64)
    np.savez(args.output / "histories.npz", uids=np.asarray(ordered,dtype=np.int64),
        offsets=np.r_[np.int64(0),lengths.cumsum()],
        timestamps=np.concatenate([history.rows[uid][0] for uid in ordered]),
        item_ids=np.concatenate([history.rows[uid][1] for uid in ordered]),
        behaviors=np.concatenate([history.rows[uid][2] for uid in ordered]))
    restored = restore_pack(args.output / "histories.npz")
    for uid in ordered:
        if any(not np.array_equal(a,b) for a,b in zip(history.rows[uid],restored.rows[uid],strict=True)):
            raise RuntimeError("serialized history changed")
    new_windows = []
    for window in windows:
        selected_requests = requests.filter(pc.equal(requests["version"],window["current"]))
        new_windows.append({**{k:v for k,v in window.items() if k!="groups"},
            "groups": dict(evaluation=dict(users=len(chosen), requests=len(selected_requests),
                active_users=len(set(selected_requests["uid"].to_pylist()))))})
    copy_keys = ("scale","model_chain","edges","initial_version","cutover","end_timestamp",
        "days","context","known_vocab_size","oov_buckets","model_config","versions","models",
        "state_rule","request_rule")
    binding = {key:reference[key] for key in copy_keys}
    binding.update(status="complete", windows=new_windows, users_file=record(args.output/"users.json"),
        files={key:record(args.output/name) for key,name in (
            ("requests","requests.parquet"),("history","histories.npz"))},
        selection=selection, selection_file=record(args.output/"selection.json"),
        configuration=record(Path(__file__)),
        sources=dict(reference_panel=record(reference_path), dataset=reference["sources"]["dataset"],
            labels=reference["sources"]["labels"], request_manifest=record(fidelity_path),
            processed_manifest=reference["sources"]["processed_manifest"],
            history_and_mapping=reference["sources"]["history_and_mapping"]),
        checks=dict(fit_validation_evaluation_disjoint=True, history_contains_evaluation_only=True,
            reference_loader_uids=canaries, exact_reference_tuple_equality=True,
            all_users_roundtrip_equality=True, contemporaneous_events_excluded=contemporaneous,
            every_request_has_one_retained_label=True),
        cost=dict(history_events=int(lengths.sum()), history_bytes=(args.output/"histories.npz").stat().st_size,
                  preparation_seconds=time.perf_counter()-began),
        preparation_sources={str(path.relative_to(ROOT)):sha256(path) for path in (
            Path(__file__),ROOT/"src/hstu_kvcache/evaluation/history.py",
            ROOT/"src/hstu_kvcache/data/yambda_history.py")})
    write_json(args.output/"binding.json",binding)
    (args.output/"README.md").write_text(
        f"# Frozen fresh-user panel\n\n{len(chosen)} users from the fixed 30k Medium population; "
        f"{len(requests)} requests across V2→V3→V4→V5. Selection excludes the prior calibration, "
        "reservation, development, Motivation and formal confirmation users named in selection.json. "
        "Users were selected before reading labels, without activity or request filtering.\n\n"
        "users.json records all selected users and original fit/validation roles; histories.npz contains "
        "evaluation histories only. requests.parquet preserves original candidates, labels and request "
        "identities. binding.json freezes source identities, model metadata and timestamp rules. "
        "No previous scores or weights are copied.\n")
    print(json.dumps(dict(status="complete",output=str(args.output),windows=new_windows,
                         **binding["cost"])),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,default=REFERENCE)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--users",type=int,default=1024)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--selection-only",action="store_true")
    parser.add_argument("--cohorts-only",action="store_true")
    main(parser.parse_args())
