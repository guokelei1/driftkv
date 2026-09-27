#!/usr/bin/env python3
"""Prepare the added V4→V5 preview using unchanged teacher/evaluation UIDs."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from design.competitor_models import load_model_pair, sha256_file
from design.prepare_expanded_auc_inputs import fixed_bank_panels
from design.prepare_unified_auc_inputs import fast_history, reference_check, write_group


def run(args):
    config = json.loads(args.config.read_text())
    base_path = ROOT / config["base_config"]
    extension_path = ROOT / config["teacher_extension_config"]
    assert sha256_file(base_path) == config["base_config_sha256"]
    assert sha256_file(extension_path) == config["teacher_extension_config_sha256"]
    base = json.loads(base_path.read_text())
    extension = json.loads(extension_path.read_text())
    teachers = config["expanded_calibration_uids"]
    assert teachers == extension["calibration_uids"]
    assert teachers[:256] == config["calibration_uids"] == base["calibration_uids"]
    assert config["evaluation_uids"] == base["evaluation_uids"]
    groups = dict(expanded_fit=teachers, pilot=config["canary"]["mature_pilot_uids"],
                  evaluation=config["evaluation_uids"])
    uids = [uid for group in groups.values() for uid in group]
    assert len(set(uids)) == len(uids)
    assert len(teachers) == 7144 and len(groups["evaluation"]) == 10000
    pair = load_model_pair("medium", 4, verify_hashes=False)
    assert pair["admission"]["reuse_eligible"]
    assert config["edges"] == [4] and pair["cutover_day"] == 287
    assert sha256_file(ROOT / config["model_chain"]) == config["model_chain_sha256"]
    data = pair["dataset"]
    for name in ("manifest", "item_mapping"):
        assert sha256_file(data[name]) == data[name + "_sha256"]
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    began = time.perf_counter()
    reference = reference_check(data, groups, [pair], 1024, args.threads)
    history, timing = fast_history(data, uids, pair["cutover"] - 1,
                                   pair["cutover"] + 1, 1024, args.threads)
    outputs = {}
    for group, selected in dict(fit=teachers[:256], **groups).items():
        directory = args.output / pair["edge"] / group
        outputs[group] = write_group(directory, history, selected, pair["cutover"], 1024)
        if group == "expanded_fit":
            items = np.load(directory / "items.npy", mmap_mode="r")
            panel, bank = fixed_bank_panels(items, data["known_items"])
            panel = panel[:, config["calibration_candidate_indices"]]
            path = directory / "panel.npy"
            np.save(path, panel, allow_pickle=False)
            outputs[group]["files"]["panel"] = dict(shape=list(panel.shape), dtype=str(panel.dtype),
                bytes=path.stat().st_size, sha256=sha256_file(path))
            outputs[group]["bank_users"] = 256
            outputs[group]["bank_size"] = len(bank)
        print(f"READY {pair['edge']}/{group}: {len(selected)} users", flush=True)
    metadata = dict(status="completed", config_sha256=sha256_file(args.config),
        source_sha256={str(path.relative_to(ROOT)): sha256_file(path) for path in
            (Path(__file__).resolve(), ROOT / "scripts/design/prepare_unified_auc_inputs.py",
             ROOT / "scripts/design/prepare_expanded_auc_inputs.py")},
        data=data, threads=args.threads, history_length=1024,
        outputs={pair["edge"]: outputs}, reference=reference, loading=timing,
        elapsed_seconds=time.perf_counter()-began, labels_read=False, model_weights_loaded=False)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared in {metadata['elapsed_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/insight/unified_auc_v4_v5_preview_20260920.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/insight/unified_auc_v4_v5_preview_20260920/prepared")
    parser.add_argument("--threads", type=int, default=48)
    run(parser.parse_args())
