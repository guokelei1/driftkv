#!/usr/bin/env python3
"""Prepare the fixed7144-teacher extension with the original256-user bank."""

from __future__ import annotations

import argparse
from itertools import islice
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from design.competitor_data import _recent_unique, make_candidate_panel
from design.competitor_models import load_model_pair, sha256_file
from design.prepare_unified_auc_inputs import ARRAY_NAMES, fast_history, reference_check, write_group


def fixed_bank_panels(items, known_items):
    """Use the exact old panel rule, freezing only its popularity bank."""
    source = np.asarray(items[:256])
    known = source[(source > 0) & (source < known_items)]
    observed, counts = np.unique(known, return_counts=True)
    bank = observed[np.lexsort((observed, -counts))[:16384]].tolist()
    panels = []
    for row in items:
        recent_set = {int(item) for item in row[-256:] if 0 < item < known_items}
        full_set = {int(item) for item in row if 0 < item < known_items}
        recent = _recent_unique(row[-256:], set(), known_items)
        old = _recent_unique(row[:-256], recent_set.copy(), known_items)
        count = 64 - len(recent) - len(old)
        # Same prefix as the original complete-bank comprehension, without
        # scanning the unused remainder for every added teacher user.
        novel = list(islice((int(item) for item in bank if item not in full_set), count))
        if len(novel) != count:
            raise ValueError("fixed original256 bank cannot fill a teacher panel")
        panels.append(recent + old + novel)
    result = np.asarray(panels, dtype=np.int64)
    original, _ = make_candidate_panel(source, known_items)
    np.testing.assert_array_equal(result[:256], original)
    return result, bank


def run(args):
    config = json.loads(args.config.read_text())
    base_path = ROOT / config["base_config"]
    assert sha256_file(base_path) == config["base_config_sha256"]
    base = json.loads(base_path.read_text())
    selected = config["calibration_uids"]
    assert len(selected) == len(set(selected)) == 7144
    assert selected[:256] == base["calibration_uids"] == config["original_calibration_uids"]
    assert config["evaluation_uids"] == base["evaluation_uids"]
    assert config["edges"] == base["edges"] == [0, 1, 2, 3]
    assert config["history_length"] == base["history_length"] == 1024
    split = json.loads((ROOT / config["source_split"]["path"]).read_text())
    forbidden = (set(base["evaluation_uids"]) | set(base["pilot_uids"]) |
                 set(split["confirmation"]) | set(split["reserved_legacy_confirmation"]) |
                 (set(split["historical_fitted_uids"]) - set(selected[:256])))
    assert not set(selected) & forbidden
    source = config["selection_provenance"]
    metadata_path = ROOT / source["metadata"]
    assert sha256_file(metadata_path) == source["metadata_sha256"]
    users = pq.read_table(metadata_path, columns=["uid", "selector_rank", "n_theta0"]).to_pandas().set_index("uid")
    teacher_metadata = users.loc[selected]
    assert int(teacher_metadata.n_theta0.min()) >= 1024
    assert (teacher_metadata.iloc[:3585].selector_rank <= 30000).all()
    assert (teacher_metadata.iloc[3585:].selector_rank > 30000).all()
    eligible = users[(users.n_theta0 >= 1024) & ~users.index.isin(forbidden | set(selected[:256]))]
    medium = eligible[(eligible.selector_rank <= 30000) & eligible.index.isin(split["calibration"])]
    medium = medium.reset_index().sort_values(["selector_rank", "uid"])
    outside = eligible[eligible.selector_rank > 30000].reset_index().sort_values(["selector_rank", "uid"])
    assert selected[256:3585] == medium.uid.astype(int).tolist()
    assert selected[3585:] == outside.uid.head(3559).astype(int).tolist()
    pairs = [load_model_pair("medium", edge, verify_hashes=False) for edge in config["edges"]]
    assert all(pair["admission"]["reuse_eligible"] for pair in pairs)
    assert sha256_file(ROOT / config["model_chain"]) == config["model_chain_sha256"]
    data = pairs[0]["dataset"]
    assert sha256_file(data["manifest"]) == data["manifest_sha256"]
    assert sha256_file(data["item_mapping"]) == data["item_mapping_sha256"]
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    reference_groups = {"original": selected[:256], "unused_medium": selected[256:3585],
                        "outside_medium": selected[3585:]}
    print("Compare original, unused-Medium and outside-Medium references with old loader", flush=True)
    reference = reference_check(data, reference_groups, pairs, 1024, args.threads)
    (args.output / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    print(f"References passed in {reference['elapsed_seconds']:.2f}s; load7144 histories", flush=True)
    history, timing = fast_history(data, selected, min(p["cutover"] for p in pairs) - 1,
                                   max(p["cutover"] for p in pairs) + 1, 1024, args.threads)
    print(f"History ready: {timing}", flush=True)
    outputs = {}
    for pair in pairs:
        edge, directory = pair["edge"], args.output / pair["edge"] / "fit"
        record = write_group(directory, history, selected, pair["cutover"], 1024)
        for name in (*ARRAY_NAMES, "uids"):
            current = np.load(directory / f"{name}.npy", mmap_mode="r")
            previous = np.load(ROOT / config["evaluation_prepared"] / edge / "fit" / f"{name}.npy", mmap_mode="r")
            np.testing.assert_array_equal(current[:256], previous)
        items = np.load(directory / "items.npy", mmap_mode="r")
        panel, bank = fixed_bank_panels(items, data["known_items"])
        panel = panel[:, config["calibration_candidate_indices"]]
        path = directory / "panel.npy"
        np.save(path, panel, allow_pickle=False)
        record["files"]["panel"] = dict(shape=list(panel.shape), dtype=str(panel.dtype),
                                         bytes=path.stat().st_size, sha256=sha256_file(path))
        record["bank_users"] = 256
        record["bank_size"] = len(bank)
        record["original256_arrays_and_panel_exactly_equal"] = True
        outputs[edge] = {"fit": record}
        (directory / "metadata.json").write_text(json.dumps(dict(status="completed", edge=edge,
            config_sha256=sha256_file(args.config), **record), indent=2) + "\n")
        print(f"READY {edge}/fit: 7144users, fixed-bank panel{panel.shape}, first256 exact match", flush=True)
    original_prepared = ROOT / config["evaluation_prepared"] / "metadata.json"
    metadata = dict(status="completed", configuration=str(args.config.resolve()),
        config_sha256=sha256_file(args.config), base_config_sha256=config["base_config_sha256"],
        source_sha256={str(path.relative_to(ROOT)): sha256_file(path) for path in
            [Path(__file__).resolve(), ROOT / "scripts/design/prepare_unified_auc_inputs.py",
             ROOT / "scripts/design/competitor_data.py"]},
        data=data, selection_metadata_sha256=source["metadata_sha256"],
        base_prepared_metadata=str(original_prepared), base_prepared_metadata_sha256=sha256_file(original_prepared),
        array_order=list(ARRAY_NAMES), uid_order="exact config.calibration_uids",
        history_length=1024, teacher_users=7144, medium_teachers=3585, outside_medium_teachers=3559,
        teacher_budgets=config["teacher_budgets"], new_budgets=config["new_budgets"],
        threads=args.threads, reference=reference, loading=timing, outputs=outputs,
        elapsed_seconds=time.perf_counter() - started, labels_read=False, model_weights_loaded=False,
        evaluation_prepared=config["evaluation_prepared"], evaluation_users_unchanged=10000)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Completed in {metadata['elapsed_seconds']:.2f}s: {args.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/insight/unified_auc_10k_teachers7144_20260920.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/insight/unified_auc_10k_teachers7144_20260920/prepared")
    parser.add_argument("--threads", type=int, default=48)
    run(parser.parse_args())
