#!/usr/bin/env python3
"""Single-edge V4-to-V5 preview using the frozen AUC execution primitives."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from design.competitor_models import load_model_pair, sha256_file
from design.expanded_read_calibration import calibrate_expanded
from design.run_unified_auc import calibration, join_quality, write_json
from design.run_unified_auc_parallel import (
    ARRAY_NAMES, REQUEST_COLUMNS, batch_spans, concatenate, launch_workers,
    load_pair, prepared_arrays, setup_gpu, source_hashes,
)
from hstu_kvcache.training.foundation import FoundationHistoryIndex

DEFAULT_ROOT = ROOT / "results/insight/unified_auc_v4_v5_preview_20260920"
BUDGETS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 7144]


def verify_inputs(args, config):
    prepared = json.loads((args.prepared / "metadata.json").read_text())
    assert prepared["status"] == "completed"
    assert prepared["config_sha256"] == sha256_file(args.config)
    groups = {"fit": config["calibration_uids"],
              "pilot": config["canary"]["mature_pilot_uids"]}
    if not args.canary:
        groups.update(expanded_fit=config["expanded_calibration_uids"],
                      evaluation=config["evaluation_uids"])
    for group, uids in groups.items():
        directory = args.prepared / "v4_to_v5" / group
        assert np.array_equal(np.load(directory / "uids.npy"), uids)
        for name, details in prepared["outputs"]["v4_to_v5"][group]["files"].items():
            assert sha256_file(directory / f"{name}.npy") == details["sha256"]
    for name in ("requests_fidelity", "requests_quality"):
        record = config["request_manifest"][name]
        assert sha256_file(ROOT / record["path"]) == record["sha256"]
    assert sha256_file(ROOT / config["model_chain"]) == config["model_chain_sha256"]
    return prepared


@torch.inference_mode()
def initial_calibration(args, config, pair):
    """Use the original 256-user candidate bank and unchanged baseline fitting."""
    device = setup_gpu(0, args.threads)
    arrays = prepared_arrays(args.prepared, pair["edge"], "fit", config["calibration_uids"])
    history = FoundationHistoryIndex({uid: (arrays[0][i], arrays[1][i], arrays[2][i])
                                     for i, uid in enumerate(config["calibration_uids"])})
    parent, current = load_pair(pair, device)
    began = time.perf_counter()
    artifact, translators = calibration(parent, current, history, pair, config,
                                        args.batch_size, device, args.canary)
    resource = dict(seconds=time.perf_counter()-began,
                    peak_allocated_gib=torch.cuda.max_memory_allocated(device)/(1 << 30))
    del parent, current, translators, arrays, history
    gc.collect()
    torch.cuda.empty_cache()
    return artifact, resource


def run(args):
    config = json.loads(args.config.read_text())
    assert config["edges"] == [4] and config["edge_names"] == ["v4_to_v5"]
    assert config["teacher_budgets"] == BUDGETS and config["history_length"] == 1024
    assert len(config["calibration_uids"]) == 256 and len(config["expanded_calibration_uids"]) == 7144
    assert config["expanded_calibration_uids"][:256] == config["calibration_uids"]
    assert len(config["evaluation_uids"]) == 10000
    assert not set(config["expanded_calibration_uids"]) & set(config["evaluation_uids"])
    assert not set(config["pilot_uids"]) & set(config["evaluation_uids"])
    assert not set(config["pilot_uids"]) & set(config["expanded_calibration_uids"])
    assert args.batch_size == (16 if args.canary else 64) and args.candidate_chunk == 32
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    began = time.perf_counter()
    try:
        prepared = verify_inputs(args, config)
        files = source_hashes()
        for path in (Path(__file__), ROOT / "scripts/design/expanded_read_calibration.py",
                     ROOT / "scripts/insight_one_locality/common.py"):
            files[str(path.relative_to(ROOT))] = sha256_file(path)
        if not args.canary:
            canary = json.loads(args.canary_record.read_text())
            canary_settings = json.loads((args.canary_record.parent / "configuration.json").read_text())
            assert canary["status"] == "completed" and canary["mode"] == "canary_parallel"
            assert canary_settings["config_sha256"] == sha256_file(args.config)
            assert canary_settings["source_sha256"] == files
        pair = load_model_pair("medium", 4, verify_hashes=True)
        assert pair["admission"]["reuse_eligible"] and pair["cutover"] == 287 * 86400
        uids = config["canary"]["mature_pilot_uids"] if args.canary else config["evaluation_uids"]
        assert len(uids) == (96 if args.canary else 10000)
        settings = dict(mode="canary_parallel" if args.canary else "diagnostic", config=config,
            config_sha256=sha256_file(args.config), evaluation_uids=uids, model_pairs=[pair],
            source_sha256=files, prepared_metadata_sha256=sha256_file(args.prepared / "metadata.json"),
            batch_size=args.batch_size, candidate_chunk=args.candidate_chunk,
            global_batch_spans=batch_spans(len(uids), args.batch_size),
            execution_overlay=dict(gpu_ids=[0, 1, 2, 3], worker_torch_threads=args.threads,
                data_threads=prepared.get("threads"), calibration_cache="CPU memory for budgets512..7144",
                canary_sha256=None if args.canary else sha256_file(args.canary_record)),
            aggregation="concatenate all raw scores, then compute pooled AUC once for this edge",
            purpose="additional single-edge preview; existing results and paper remain unchanged")
        write_json(args.output / "configuration.json", settings)
        print("Calibrate original teacher budgets and baseline families on GPU0", flush=True)
        artifact, resource = initial_calibration(args, config, pair)
        if not args.canary:
            directory = args.prepared / pair["edge"] / "expanded_fit"
            print("Calibrate additional teacher budgets on GPU0/1/2/3", flush=True)
            extension = calibrate_expanded(pair, directory, directory / "panel.npy", BUDGETS[4:],
                batch_size=64, gpus=(0, 1, 2, 3), threads=args.threads)
            artifact["rules"].update(extension["rules"])
            artifact["fitting"].update(extension["fitting"])
            artifact.update(calibration_uids=config["expanded_calibration_uids"], teacher_budgets=BUDGETS,
                extension_execution=extension["execution"], panel_sha256=extension["panel_sha256"])
            assert len(artifact["rules"]) == 18
            del extension
        artifact.update(config_sha256=settings["config_sha256"],
            model_binding={role: pair[role]["checkpoint_sha256"] for role in ("parent", "current")})
        rule_path = args.output / f"{pair['edge']}_calibration.pt"
        torch.save(artifact, rule_path)
        if args.canary:
            print("Canary: single-GPU reference followed by four-GPU evaluation", flush=True)
            launch_workers(args, 0, reference=True)
        launch_workers(args, 0)
        raw, execution = concatenate(args, pair, uids)
        paths = [name for name in raw if name not in (*REQUEST_COLUMNS, "prefix_last_timestamp", "prefix_length")]
        assert len(paths) == (25 if args.canary else 37)
        record = dict(edge=pair["edge"], model_pair=pair, rules_sha256=sha256_file(rule_path),
            teacher_budgets=artifact["teacher_budgets"], profile_users=artifact["profile_users"],
            translate_users=artifact["translate_users"], intervals=artifact["intervals"],
            interval_profiles=artifact["interval_profiles"], numerical=artifact["numerical"],
            calibration_seconds=resource["seconds"], calibration_peak_allocated_gib=resource["peak_allocated_gib"],
            calibration_execution=artifact.get("extension_execution"), **execution)
        if args.canary:
            reference = pd.read_parquet(args.output / pair["edge"] / "reference" / "scores.parquet")
            columns = [*REQUEST_COLUMNS, "prefix_last_timestamp", "prefix_length"]
            assert reference[columns].equals(raw[columns])
            differences = {}
            for name in paths:
                np.testing.assert_allclose(raw[name], reference[name], atol=2e-5, rtol=2e-5)
                differences[name] = float(np.max(np.abs(raw[name].to_numpy()-reference[name].to_numpy())))
            record.update(maximum_absolute_logit_difference=max(differences.values()),
                          per_path_logit_difference=differences, labels_read=False)
        else:
            record.update(**join_quality(raw, args.output, pair["edge"]))
        record["elapsed_seconds"] = time.perf_counter()-began
        write_json(args.output / f"{pair['edge']}_summary.json", record)
        write_json(args.output / "summary.json", dict(status="completed", mode=settings["mode"],
            paths=len(paths), elapsed_seconds=time.perf_counter()-began, edges=[record]))
    except Exception as exc:
        write_json(args.output / "summary.json", dict(status="failed", error=repr(exc),
            elapsed_seconds=time.perf_counter()-began))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/insight/unified_auc_v4_v5_preview_20260920.json")
    parser.add_argument("--prepared", type=Path, default=DEFAULT_ROOT / "prepared")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--canary-record", type=Path, default=DEFAULT_ROOT / "canary/summary.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--candidate-chunk", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())
