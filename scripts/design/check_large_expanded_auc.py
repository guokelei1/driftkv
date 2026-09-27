#!/usr/bin/env python3
"""Check ten-layer expanded calibration against the original 256-user fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from design.competitor_models import load_model_pair, sha256_file
from design.expanded_read_calibration import calibrate_expanded, compare_reference
from design.run_shared_read_probe import write_json

DEFAULT_ROOT = ROOT / "results/insight/large_unified_auc_10k_20260920"


def source_hashes():
    paths = [Path(__file__), ROOT / "scripts/design/competitor_models.py",
        ROOT / "scripts/design/expanded_read_calibration.py",
        ROOT / "scripts/design/bias_read_probe.py", ROOT / "scripts/design/run_auc_read_probe.py",
        ROOT / "scripts/design/run_shared_read_probe.py",
        ROOT / "scripts/evaluate_yambda500m_foundation_raw.py",
        ROOT / "src/hstu_kvcache/adaptation/reader.py",
        *sorted((ROOT / "src/hstu_kvcache/models").glob("*.py"))]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def main(args):
    config = json.loads(args.config.read_text())
    config_hash = sha256_file(args.config)
    assert config["scale"] == "large" and config["history_length"] == 1024
    prepared_path = args.prepared / "metadata.json"
    prepared = json.loads(prepared_path.read_text())
    assert prepared["status"] == "completed" and prepared["config_sha256"] == config_hash
    pair = load_model_pair("large", 0, verify_hashes=True)
    assert pair["admission"]["reuse_eligible"]
    assert pair["chain_manifest_sha256"] == config["model_chain_sha256"]
    assert pair["config"]["num_layers"] == 10
    directory = args.prepared / pair["edge"] / "expanded_fit"
    panel = directory / "panel.npy"
    uids = np.load(directory / "uids.npy", mmap_mode="r")
    np.testing.assert_array_equal(uids, config["expanded_calibration_uids"])
    assert uids[:256].tolist() == config["calibration_uids"]
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    sources = source_hashes()
    settings = dict(config=str(args.config.resolve()), config_sha256=config_hash,
        prepared_metadata_sha256=sha256_file(prepared_path), model_pair=pair,
        source_sha256=sources, panel_sha256=sha256_file(panel), teacher_budgets=[256, 512],
        batch_size=64, gpus=[0, 1, 2, 3], threads_per_worker=4,
        reference_budget=256, labels_read=False, evaluation_users_read=False)
    write_json(args.output / "configuration.json", settings)
    started = time.perf_counter()
    summary = dict(status="running", **settings)
    try:
        artifact = calibrate_expanded(pair, directory, panel, [256, 512],
            batch_size=64, gpus=(0, 1, 2, 3), threads=4)
        artifact["config_sha256"] = config_hash
        rule_path = args.output / "rules.pt"
        torch.save(artifact, rule_path)
        summary.update(execution=artifact["execution"], rules_sha256=sha256_file(rule_path),
            fitting=artifact["fitting"])
        began = time.perf_counter()
        summary["reference"] = compare_reference(pair, directory, panel, artifact,
            budget=256, batch_size=64, gpu=0, threads=4)
        summary["reference_seconds"] = time.perf_counter() - began
        assert source_hashes() == sources and sha256_file(args.config) == config_hash
        summary.update(status="completed", all_execution_sources_unchanged=True)
    except Exception as exc:
        summary.update(status="failed", error=repr(exc))
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        write_json(args.output / "summary.json", summary)
    print(f"Completed expanded calibration check in {summary['elapsed_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
        default=ROOT / "configs/insight/large_unified_auc_10k_20260920.json")
    parser.add_argument("--prepared", type=Path, default=DEFAULT_ROOT / "prepared")
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "expanded_canary")
    main(parser.parse_args())
