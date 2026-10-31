#!/usr/bin/env python3
"""Prepare a separate fixed pre-first-release active scenario, without models.

Membership uses only the existing first-release prehistory features and hashes.
Original roles and E6 reservations are excluded. Future requests are loaded only
after membership is saved; future activity is never an admission criterion.
"""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from experiments.ch3_motivation.code import common
from experiments.ch3_motivation.code.scenario_inputs import history_signature, load_scenario_population
from hstu_kvcache.evaluation.history import restore_pack, load_histories_semijoin
from hstu_kvcache.common.artifacts import sha256

BASE = ROOT / "experiments/ch3_motivation"
INPUTS = BASE / "inputs"
STRATA = BASE / "runs/scenario_qualification/strata_pre_release/attempt_1/scenario_strata_manifest.json"


def freeze(output):
    population_path = INPUTS / "population.json"
    population = json.loads(population_path.read_text())
    strata = json.loads(STRATA.read_text())
    if strata["source_population"]["sha256"] != sha256(population_path):
        raise ValueError("pre-release features use different original roles")
    excluded = {uid for group in population["population_sensitivity"].values() for uid in group}
    excluded.update(uid for role in ("fit", "validation", "development", "main", "mismatch", "confirmation")
                    for uid in population[role])
    reservations = {}
    for edge in range(1, 6):
        path = INPUTS / f"edge_{edge}/inputs.json"
        excluded.update(json.loads(path.read_text())["e6_users"])
        reservations[str(path)] = sha256(path)
        path = BASE / f"runs/scenario_qualification/attempt_1/edge_{edge}/feedback_union_2000/scenario_manifest.json"
        excluded.update(json.loads(path.read_text())["uids"])
        reservations[str(path)] = sha256(path)
    scenario = "long_feedback_active"
    eligible = set(strata["subgroups_by_edge"]["1"][scenario]["uids"]) & set(population["reserve"])
    eligible -= excluded
    ordered = sorted(eligible, key=lambda uid: hashlib.sha256(f"e5_active_pre_v1_seed17:{uid}".encode()).digest())
    if len(ordered) < 4512:
        raise ValueError("not enough distinct users for development and two future held-out panels")
    roles = dict(development=ordered[:512], main=ordered[512:2512], confirmation=ordered[2512:4512])
    value = dict(schema_version=1, scenario=scenario, definition=strata["definitions"][scenario],
        membership_time="pre-first-release only, same membership on all five edges", roles=roles,
        eligible_users=len(eligible), original_population_sha256=sha256(population_path),
        strata_manifest=str(STRATA), strata_sha256=sha256(STRATA), excluded_reservations=reservations,
        selection="SHA256 e5_active_pre_v1_seed17:UID ascending, without outcomes or future activity",
        source_sha256=sha256(Path(__file__)), model_outcomes_used=False,
        future_activity_used=False, replaces_original_panels=False)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "cohort.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("previously frozen active cohort changed")
    common.write_json(path, value)
    return value, population


def prepare(output, threads):
    cohort, population = freeze(output)  # Must precede any future-feedback read.
    dataset = json.loads(common.DATASET.read_text())
    results = []
    for edge in range(1, 6):
        started = time.perf_counter()
        original = INPUTS / f"edge_{edge}"
        source = original / "inputs.json"
        binding = json.loads(source.read_text())
        settings = binding["settings"]
        if sha256(original / binding["history"]["path"]) != binding["history"]["sha256"]:
            raise ValueError("original causal history pack changed")
        directory = output / f"edge_{edge}"
        directory.mkdir(exist_ok=True)
        if (directory / "candidate_manifest.json").exists():
            load_scenario_population(directory / "candidate_manifest.json", edge, require_qualification=False)
            results.append(str(directory / "candidate_manifest.json"))
            continue
        quality = cohort["roles"]["development"]
        users = sorted(set(quality + population["fit"] + population["validation"]))
        old = restore_pack(original / binding["history"]["path"])
        missing = [uid for uid in users if uid not in old.rows]
        additional = load_histories_semijoin(missing, dataset_path=common.DATASET,
            known_vocab_size=dataset["foundation_items"], oov_buckets=256,
            start_timestamp=settings["release_day"] * common.DAY,
            end_timestamp=settings["end_day"] * common.DAY,
            max_history=settings["history_length"], threads=threads) if missing else SimpleNamespace(rows={})
        histories = SimpleNamespace(rows={uid: old.rows[uid] if uid in old.rows else additional.rows[uid] for uid in users})
        calibration = population["fit"] + population["validation"]
        signature = history_signature(histories, calibration)
        if signature != history_signature(old, calibration):
            raise ValueError("supplement changed existing calibration history")
        common.save_histories(directory / "histories.npz", histories, users)
        connection = duckdb.connect()
        connection.execute(f"PRAGMA threads={int(threads)}")
        connection.register("selected", pa.table({"uid": pa.array(users, type=pa.uint64())}))
        feedback = common.quoted((common.DATASET.parent / dataset["shared_feedback_glob"]).resolve())
        mapping = common.quoted((common.DATASET.parent / dataset["item_mapping_path"]).resolve())
        try:
            requests = connection.execute(f"""
                WITH grouped AS (
                  SELECT uid,timestamp,raw_item_id,min(label) AS label
                  FROM read_parquet('{feedback}',hive_partitioning=true) SEMI JOIN selected USING(uid)
                  WHERE timestamp>=? AND timestamp<? GROUP BY uid,timestamp,raw_item_id HAVING min(label)=max(label)
                ) SELECT g.uid,g.timestamp AS query_timestamp,g.raw_item_id,m.item_idx,g.label
                  FROM grouped g JOIN read_parquet('{mapping}') m USING(raw_item_id)
                  ORDER BY uid,query_timestamp,raw_item_id
            """, [settings["release_day"] * common.DAY, settings["end_day"] * common.DAY]).fetch_arrow_table()
        finally:
            connection.close()
        pq.write_table(requests, directory / "requests.parquet")
        inputs = dict(settings=settings, population_sha256=sha256(INPUTS / "population.json"),
            history=dict(path="histories.npz", sha256=sha256(directory / "histories.npz")),
            requests=dict(path="requests.parquet", sha256=sha256(directory / "requests.parquet")),
            calibration_source_binding=dict(path=str(source), sha256=sha256(source)),
            calibration_inputs_unchanged=True, calibration_history_sha256=signature,
            cohort_sha256=sha256(output / "cohort.json"), packed_users=len(users),
            newly_loaded_users=len(missing), seconds=time.perf_counter() - started)
        common.write_json(directory / "inputs.json", inputs)
        def reference(path):
            return dict(path=str(path), sha256=sha256(path))
        manifest = dict(schema_version=1, edge=edge, uids=quality, roles=dict(quality=quality),
            qualification=dict(status="not_yet_measured"), input_binding=reference(directory / "inputs.json"),
            history=reference(directory / "histories.npz"), requests=reference(directory / "requests.parquet"),
            source_population=reference(INPUTS / "population.json"),
            selection=dict(panel_name="pre_first_release_active_development512", definition=cohort["definition"],
                cohort_path=str(output / "cohort.json"), cohort_sha256=sha256(output / "cohort.json"),
                description=cohort["selection"], membership_time=cohort["membership_time"]))
        common.write_json(directory / "candidate_manifest.json", manifest)
        load_scenario_population(directory / "candidate_manifest.json", edge, require_qualification=False)
        results.append(str(directory / "candidate_manifest.json"))
        print(json.dumps(dict(edge=edge, users=len(quality), packed_users=len(users), status="causal_inputs_checked")), flush=True)
    common.write_json(output / "summary.json", dict(status="active_scenario_inputs_prepared", manifests=results,
        quality_users=512, cohort_sha256=sha256(output / "cohort.json"), gpu_launched=False,
        model_outcomes_used=False, qualification_performed=False, formal_e5_complete=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    prepare(args.output.resolve(), args.threads)
