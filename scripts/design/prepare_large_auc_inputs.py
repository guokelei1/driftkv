#!/usr/bin/env python3
"""Prepare matched 6L/10L requests and Large-mapped release snapshots on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import duckdb
import numpy as np
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from design.competitor_models import load_model_pair, sha256_file
from design.prepare_expanded_auc_inputs import fixed_bank_panels
from design.prepare_unified_auc_inputs import (
    ARRAY_NAMES, fast_history, reference_check, sql_path, write_group,
)

DEFAULT_ROOT = ROOT / "results/insight/large_unified_auc_10k_20260920"
IDENTITY = "uid,query_timestamp,raw_item_id,is_organic"
EXPECTED_REQUESTS = [54826, 53874, 52309, 51764, 50672]


def prepare_requests(args):
    started = time.perf_counter()
    source = json.loads(args.source_config.read_text())
    evaluation = source["evaluation_uids"]
    pilot = source["canary"]["mature_pilot_uids"][:96]
    assert len(evaluation) == 10000 and len(pilot) == 96
    assert len(set(evaluation + pilot)) == 10096
    medium = ROOT / source["request_manifest"]["requests_fidelity"]["path"]
    assert sha256_file(medium) == source["request_manifest"]["requests_fidelity"]["sha256"]
    large_root = ROOT / "data/manifests/yambda500m_large_hstu_native_d7_d14_v1"
    large_manifest = large_root / "manifest.json"
    manifest = json.loads(large_manifest.read_text())
    large = large_root / "requests_fidelity.parquet"
    assert sha256_file(large) == manifest["artifacts"][large.name]["sha256"]
    if args.request_output.exists():
        raise FileExistsError(args.request_output)
    args.request_output.mkdir(parents=True)
    db = duckdb.connect()
    db.execute(f"SET threads={args.threads}")
    db.register("selected", pa.table({
        "uid": pa.array(evaluation + pilot, type=pa.uint64()),
        "cohort": ["evaluation"] * len(evaluation) + ["pilot"] * len(pilot),
    }))
    for name, path, known in (("medium", medium, "AND r.target_known"), ("large", large, "")):
        db.execute(f"""CREATE TEMP TABLE {name} AS
            SELECT r.*,s.cohort FROM read_parquet('{sql_path(path)}') r
            JOIN selected s USING(uid)
            WHERE r.time_block='matrix_horizon'
              AND r.query_timestamp >= {231 * 86400}
              AND r.query_timestamp < {300 * 86400} {known}""")
        count, unique = db.execute(
            f"SELECT count(*),count(DISTINCT ({IDENTITY})) FROM {name}"
        ).fetchone()
        assert count == unique, f"duplicate {name} request identity"
    db.execute(f"""CREATE TEMP TABLE matched AS
        SELECT l.*,m.request_id AS medium_request_id
        FROM large l JOIN medium m USING({IDENTITY})""")
    matched, unknown, ids = db.execute(
        "SELECT count(*),count(*) FILTER(WHERE NOT target_known),count(DISTINCT request_id) FROM matched"
    ).fetchone()
    assert matched == db.execute("SELECT count(*) FROM medium").fetchone()[0]
    assert unknown == 0 and ids == matched
    fidelity = args.request_output / "requests_fidelity.parquet"
    mapping = args.request_output / "request_identity_map.parquet"
    db.execute(f"""COPY (SELECT * EXCLUDE(cohort,medium_request_id) FROM matched
        ORDER BY uid,query_timestamp,raw_item_id,is_organic)
        TO '{sql_path(fidelity)}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    db.execute(f"""COPY (SELECT {IDENTITY},cohort,medium_request_id,
        request_id AS large_request_id FROM matched
        ORDER BY uid,query_timestamp,raw_item_id,is_organic)
        TO '{sql_path(mapping)}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    counts = []
    for index, day in enumerate([231, 245, 259, 273, 287]):
        edge = f"v{index}_to_v{index + 1}"
        end = min(day + 14, 300)
        row = dict(edge=edge, start_day=day, effective_end_day=end, effective_days=end - day)
        for cohort in ("evaluation", "pilot"):
            requests, users = db.execute(f"""SELECT count(*),count(DISTINCT uid) FROM matched
                WHERE cohort='{cohort}' AND query_timestamp>={day * 86400}
                  AND query_timestamp<{end * 86400}""").fetchone()
            row[cohort] = dict(requests=requests, feedback_users=users)
        assert row["evaluation"]["requests"] == EXPECTED_REQUESTS[index]
        counts.append(row)
    db.close()
    metadata = dict(status="completed", source_configuration=str(args.source_config.resolve()),
        source_config_sha256=sha256_file(args.source_config), source_sha256=sha256_file(__file__),
        medium_fidelity=dict(path=str(medium), sha256=sha256_file(medium)),
        large_manifest=dict(path=str(large_manifest), sha256=sha256_file(large_manifest)),
        large_fidelity=dict(path=str(large), sha256=manifest["artifacts"][large.name]["sha256"]),
        requests_fidelity=dict(path=str(fidelity), sha256=sha256_file(fidelity), rows=matched),
        requests_quality=dict(path=str(large_root / "requests_quality.parquet"),
            sha256=manifest["artifacts"]["requests_quality.parquet"]["sha256"],
            source="unchanged Large manifest; labels not read during preparation"),
        request_identity_map=dict(path=str(mapping), sha256=sha256_file(mapping)),
        identity_columns=IDENTITY.split(","), evaluation_uids=evaluation, pilot_uids=pilot,
        selection="exact Medium-known raw request identities; Large request_id and item_idx; all selected targets Large-known",
        one_to_one_identity_match=True, counts=counts, threads=args.threads,
        labels_read=False, model_weights_loaded=False, elapsed_seconds=time.perf_counter() - started)
    (args.request_output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: metadata[k] for k in ("status", "requests_fidelity", "counts", "elapsed_seconds")}), flush=True)


def prepare_histories(args):
    config = json.loads(args.config.read_text())
    source = json.loads(args.source_config.read_text())
    assert config["scale"] == "large" and config["edges"] == [0, 1, 2, 3, 4]
    assert config["history_length"] == 1024
    fit, expanded = config["calibration_uids"], config["expanded_calibration_uids"]
    pilot, evaluation = config["canary"]["mature_pilot_uids"][:96], config["evaluation_uids"]
    assert fit == expanded[:256] == source["calibration_uids"]
    assert expanded == source["expanded_calibration_uids"] and len(expanded) == 7144
    assert evaluation == source["evaluation_uids"] and len(evaluation) == 10000
    assert pilot == source["canary"]["mature_pilot_uids"][:96] and len(pilot) == 96
    uids = expanded + pilot + evaluation
    assert len(uids) == len(set(uids))
    pairs = [load_model_pair("large", edge, verify_hashes=False) for edge in config["edges"]]
    assert sha256_file(ROOT / config["model_chain"]) == config["model_chain_sha256"]
    data = pairs[0]["dataset"]
    assert sha256_file(data["manifest"]) == data["manifest_sha256"]
    assert sha256_file(data["item_mapping"]) == data["item_mapping_sha256"]
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    groups = dict(fit=fit, expanded_fit=expanded, pilot=pilot, evaluation=evaluation)
    references = dict(fit=fit, added_teachers=expanded[256:], pilot=pilot, evaluation=evaluation)
    reference = reference_check(data, references, pairs, 1024, args.threads)
    (args.output / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    print(f"Old-loader reference passed; load {len(uids)} histories with Large item mapping", flush=True)
    history, timing = fast_history(data, uids, pairs[0]["cutover"] - 1,
                                   pairs[-1]["cutover"] + 1, 1024, args.threads)
    outputs = {}
    for pair in pairs:
        edge = pair["edge"]
        records = {}
        for group, selected in groups.items():
            directory = args.output / edge / group
            record = write_group(directory, history, selected, pair["cutover"], 1024)
            if group in ("fit", "expanded_fit"):
                items = np.load(directory / "items.npy", mmap_mode="r")
                panel, bank = fixed_bank_panels(items, data["known_items"])
                panel = panel[:, config["calibration_candidate_indices"]]
                path = directory / "panel.npy"
                np.save(path, panel, allow_pickle=False)
                record["files"]["panel"] = dict(shape=list(panel.shape), dtype=str(panel.dtype),
                    bytes=path.stat().st_size, sha256=sha256_file(path))
                record.update(bank_users=256, bank_size=len(bank))
            records[group] = record
        for name in (*ARRAY_NAMES, "uids", "panel"):
            a = np.load(args.output / edge / "fit" / f"{name}.npy", mmap_mode="r")
            b = np.load(args.output / edge / "expanded_fit" / f"{name}.npy", mmap_mode="r")
            np.testing.assert_array_equal(a, b[:256])
        outputs[edge] = records
        (args.output / edge / "metadata.json").write_text(json.dumps(dict(status="completed",
            edge=edge, config_sha256=sha256_file(args.config), groups=records), indent=2) + "\n")
        print(f"READY {edge}: fit256/expanded_fit7144/pilot96/evaluation10000", flush=True)
    metadata = dict(status="completed", configuration=str(args.config.resolve()),
        config_sha256=sha256_file(args.config), source_config_sha256=sha256_file(args.source_config),
        source_sha256={str(p.relative_to(ROOT)): sha256_file(p) for p in
            [Path(__file__).resolve(), ROOT / "scripts/design/prepare_unified_auc_inputs.py",
             ROOT / "scripts/design/prepare_expanded_auc_inputs.py", ROOT / "scripts/design/competitor_data.py"]},
        data=data, model_chain_sha256=config["model_chain_sha256"], threads=args.threads,
        history_length=1024, array_order=list(ARRAY_NAMES),
        uid_order="exact configured order; fit is expanded_fit[:256]",
        cutover_days=[p["cutover_day"] for p in pairs],
        original_admission=[dict(edge=p["edge"], **p["admission"]) for p in pairs],
        reference=reference, loading=timing, outputs=outputs,
        all_teachers_in_large_population=True, first256_arrays_and_panels_equal=True,
        labels_read=False, model_weights_loaded=False, elapsed_seconds=time.perf_counter() - started)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Completed in {metadata['elapsed_seconds']:.2f}s: {args.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests-only", action="store_true")
    parser.add_argument("--source-config", type=Path,
        default=ROOT / "configs/insight/unified_auc_v4_v5_preview_20260920.json")
    parser.add_argument("--config", type=Path,
        default=ROOT / "configs/insight/large_unified_auc_10k_20260920.json")
    parser.add_argument("--request-output", type=Path, default=DEFAULT_ROOT / "requests")
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "prepared")
    parser.add_argument("--threads", type=int, default=48)
    args = parser.parse_args()
    prepare_requests(args) if args.requests_only else prepare_histories(args)
