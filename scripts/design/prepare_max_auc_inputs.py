#!/usr/bin/env python3
"""Prepare the fixed independent Max cohorts with their native item mapping."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from design.max_auc_models import load_model_pair
from design.competitor_models import sha256_file
from design.prepare_expanded_auc_inputs import fixed_bank_panels
from design.prepare_unified_auc_inputs import ARRAY_NAMES, fast_history, reference_check, write_group


def run(args):
    config = json.loads(args.config.read_text())
    assert config["scale"] == "max" and config["edges"] == [0, 1]
    assert config["history_length"] == 1024
    fit, expanded = config["calibration_uids"], config["expanded_calibration_uids"]
    pilot, evaluation = config["pilot_uids"], config["evaluation_uids"]
    assert len(fit) == 256 and fit == expanded[:256] and len(expanded) == 7144
    assert len(evaluation) == 10000 and len(pilot) == 96
    assert pilot == config["canary"]["mature_pilot_uids"]
    uids = expanded + pilot + evaluation
    assert len(set(uids)) == len(uids)
    population = config["source_population"]
    counts_path = ROOT / population["eligibility_counts"]
    assert sha256_file(counts_path) == population["eligibility_counts_sha256"]
    counts = pq.read_table(counts_path).to_pandas().set_index("uid")
    assert int(counts.loc[uids, "n_pre217"].min()) >= 1024
    assert sha256_file(ROOT / population["users"]) == population["users_sha256"]
    members = set(pq.read_table(ROOT / population["users"], columns=["uid"])["uid"].to_pylist())
    assert set(uids) <= members
    old_split = ROOT / population["previous_split"]
    old_config = ROOT / population["previous_diagnostic_config"]
    assert sha256_file(old_split) == population["previous_split_sha256"]
    assert sha256_file(old_config) == population["previous_diagnostic_config_sha256"]
    previous, old = json.loads(old_split.read_text()), json.loads(old_config.read_text())
    excluded = set()
    for name in ("development", "calibration", "confirmation", "reserved_legacy_confirmation", "historical_fitted_uids"):
        excluded.update(previous[name])
    for name in ("expanded_calibration_uids", "pilot_uids", "evaluation_uids"):
        excluded.update(old[name])
    assert not set(uids) & excluded
    pairs = [load_model_pair("max", i, verify_hashes=False) for i in config["edges"]]
    assert all(p["admission"]["reuse_eligible"] for p in pairs)
    assert sha256_file(ROOT / config["model_chain"]) == config["model_chain_sha256"]
    data = pairs[0]["dataset"]
    assert sha256_file(data["item_mapping"]) == data["item_mapping_sha256"]
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    groups = dict(fit=fit, expanded_fit=expanded, pilot=pilot, evaluation=evaluation)
    reference = reference_check(data, dict(fit=fit, added=expanded[256:], pilot=pilot,
                                          evaluation=evaluation), pairs, 1024, args.threads)
    (args.output / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    print(f"Reference passed; load {len(uids)} Max histories once", flush=True)
    history, timing = fast_history(data, uids, pairs[0]["cutover"] - 1,
                                  pairs[-1]["cutover"] + 1, 1024, args.threads)
    outputs = {}
    for pair in pairs:
        edge, cutover = pair["edge"], pair["cutover"]
        records = {}
        for group, selected in groups.items():
            directory = args.output / edge / group
            record = write_group(directory, history, selected, cutover, 1024)
            items = np.load(directory / "items.npy", mmap_mode="r")
            timestamps = np.load(directory / "timestamps.npy", mmap_mode="r")
            assert items.shape == (len(selected), 1024) and (items > 0).all()
            assert (timestamps < cutover).all() and (timestamps[:, 1:] >= timestamps[:, :-1]).all()
            if group in ("fit", "expanded_fit"):
                panel, bank = fixed_bank_panels(items, data["known_item_upper_exclusive"])
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
        config_sha256=sha256_file(args.config), model_chain_sha256=config["model_chain_sha256"],
        source_sha256={str(p.relative_to(ROOT)): sha256_file(p) for p in
            [Path(__file__).resolve(), ROOT / "scripts/design/max_auc_models.py",
             ROOT / "scripts/design/prepare_unified_auc_inputs.py",
             ROOT / "scripts/design/prepare_expanded_auc_inputs.py", ROOT / "scripts/design/competitor_data.py"]},
        data=data, source_population=population, selection_provenance=config["selection_provenance"],
        threads=args.threads, history_length=1024, array_order=list(ARRAY_NAMES),
        uid_order="exact configured order; fit is expanded_fit[:256]", cutover_days=config["cutover_days"],
        reference=reference, loading=timing, outputs=outputs, all_groups_full1024_causal=True,
        first256_arrays_and_panels_equal=True, labels_read=False, model_weights_loaded=False,
        elapsed_seconds=time.perf_counter() - started)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Completed in {metadata['elapsed_seconds']:.2f}s: {args.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
        default=ROOT / "configs/insight/max_unified_auc_10k_20260921.json")
    parser.add_argument("--output", type=Path,
        default=ROOT / "results/insight/max_unified_auc_10k_20260921/prepared")
    parser.add_argument("--threads", type=int, default=48)
    run(parser.parse_args())
