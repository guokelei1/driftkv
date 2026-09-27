#!/usr/bin/env python3
"""Parallelize independent initial calibrations, retaining the frozen runner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from design import run_large_auc as main
from design.competitor_models import load_model_pair, sha256_file
from design.large_auc_primitives import compare_translators
from design.run_shared_read_probe import write_json
from hstu_kvcache.baselines.kv_translate import KVTranslator

DEFAULT_ROOT = ROOT / "results/insight/large_unified_auc_10k_20260920"


def initial_worker(args):
    settings = json.loads((args.output / "configuration.json").read_text())
    assert settings["wrapper_sha256"] == sha256_file(Path(__file__))
    assert settings["config_sha256"] == sha256_file(args.config)
    pair = settings["model_pairs"][args.edge]
    directory = args.output / pair["edge"]
    began = time.perf_counter()
    try:
        # The child sees one GPU; frozen initial_calibration still uses cuda:0.
        assert torch.cuda.device_count() == 1
        artifact, resource = main.initial_calibration(args, settings["config"], pair)
        path = directory / "calibration.pt"
        torch.save(artifact, path)
        write_json(directory / "summary.json", dict(status="completed", edge=pair["edge"],
            artifact_sha256=sha256_file(path), resource=resource, model_pair=pair,
            physical_gpu=int(os.environ["CUDA_VISIBLE_DEVICES"]),
            elapsed_seconds=time.perf_counter()-began))
    except Exception as exc:
        write_json(directory / "summary.json", dict(status="failed", error=repr(exc)))
        raise


def launch_wave(args, stage, indices, *, canary=False):
    children = []
    try:
        settings = json.loads((stage / "configuration.json").read_text())
        for index in indices:
            edge = settings["model_pairs"][index]["edge"]
            directory = stage / edge
            directory.mkdir()
            physical_gpu = 1 if canary else index % 4
            log = (directory / "runtime.log").open("w")
            command = [sys.executable, str(Path(__file__)), "--initial-worker", "--edge", str(index),
                "--config", str(args.config), "--prepared", str(args.prepared), "--output", str(stage),
                "--batch-size", str(args.batch_size), "--threads", str(args.threads)]
            if canary:
                command.append("--canary")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(physical_gpu),
                       OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
            children.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env), log))
        for process, _ in children:
            if process.wait() != 0:
                raise RuntimeError(f"Initial calibration failed; see {stage}")
    finally:
        for process, log in children:
            if process.poll() is None:
                process.terminate()
            process.wait()
            log.close()


def compare_original_canary(stage, original):
    path = stage / "v0_to_v1/calibration.pt"
    reference_path = original / "v0_to_v1_calibration.pt"
    actual = torch.load(path, map_location="cpu", weights_only=False)
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    assert actual["rules"].keys() == reference["rules"].keys()
    maximum = 0.0
    for name in actual["rules"]:
        for left, right in zip(actual["rules"][name], reference["rules"][name], strict=True):
            assert left.keys() == right.keys()
            for key in left:
                torch.testing.assert_close(left[key], right[key], atol=2e-5, rtol=2e-5)
                maximum = max(maximum, float((left[key]-right[key]).abs().max()))
    assert actual["intervals"] == reference["intervals"]
    for key in ("calibration_uids", "teacher_budgets", "profile_users", "translate_users"):
        assert actual[key] == reference[key]
    translated = compare_translators(
        {int(k): KVTranslator(**value) for k, value in actual["translators"].items()},
        {int(k): KVTranslator(**value) for k, value in reference["translators"].items()})
    return dict(rule_maximum_parameter_difference=maximum, intervals_identical=True,
        translator_comparison=translated, original_calibration_sha256=sha256_file(reference_path),
        mapped_calibration_sha256=sha256_file(path))


def run(args):
    config = json.loads(args.config.read_text())
    assert config["scale"] == "large" and config["edges"] == [0, 1, 2, 3, 4]
    assert args.batch_size == (16 if args.canary else 64)
    original_settings = json.loads((args.original_canary / "configuration.json").read_text())
    original_summary = json.loads((args.original_canary / "summary.json").read_text())
    assert original_summary["status"] == "completed"
    assert original_settings["config_sha256"] == sha256_file(args.config)
    for source, digest in original_settings["source_sha256"].items():
        assert sha256_file(ROOT / source) == digest
    metadata = json.loads((args.prepared / "metadata.json").read_text())
    assert metadata["status"] == "completed" and metadata["config_sha256"] == sha256_file(args.config)
    stage = args.output if args.canary else args.baselines
    if stage.exists() or (not args.canary and args.output.exists()):
        raise FileExistsError(stage if stage.exists() else args.output)
    stage.mkdir(parents=True)
    began = time.perf_counter()
    try:
        scheduler_check = None
        if not args.canary:
            scheduler_check = json.loads((args.scheduler_canary / "summary.json").read_text())
            assert scheduler_check["status"] == "completed"
            assert scheduler_check["wrapper_sha256"] == sha256_file(Path(__file__))
            assert scheduler_check["config_sha256"] == sha256_file(args.config)
        pairs = [load_model_pair("large", i, verify_hashes=True)
                 for i in (config["edges"][:1] if args.canary else config["edges"])]
        for pair in pairs:
            assert pair["admission"]["reuse_eligible"] or pair["edge"] in config["allowed_unadmitted_edges"]
            directory = args.prepared / pair["edge"] / "fit"
            assert np.array_equal(np.load(directory / "uids.npy"), config["calibration_uids"])
            for name, detail in metadata["outputs"][pair["edge"]]["fit"]["files"].items():
                assert sha256_file(directory / f"{name}.npy") == detail["sha256"]
            panel = args.prepared / pair["edge"] / "expanded_fit/panel.npy"
            assert sha256_file(panel) == metadata["outputs"][pair["edge"]]["expanded_fit"]["files"]["panel"]["sha256"]
        settings = dict(mode="initial_parallel_canary" if args.canary else "initial_parallel",
            config=config, config_sha256=sha256_file(args.config), model_pairs=pairs,
            wrapper_sha256=sha256_file(Path(__file__)), frozen_source_sha256=original_settings["source_sha256"],
            original_canary_sha256=sha256_file(args.original_canary / "summary.json"),
            prepared_metadata_sha256=sha256_file(args.prepared / "metadata.json"),
            scheduler_canary_sha256=None if args.canary else sha256_file(args.scheduler_canary / "summary.json"),
            batch_size=args.batch_size, worker_threads=args.threads,
            execution="unchanged initial_calibration in isolated single-visible-GPU subprocesses; four edges then one")
        write_json(stage / "configuration.json", settings)
        if args.canary:
            launch_wave(args, stage, [0], canary=True)
            write_json(stage / "summary.json", dict(status="completed", mode=settings["mode"],
                config_sha256=settings["config_sha256"], wrapper_sha256=settings["wrapper_sha256"],
                elapsed_seconds=time.perf_counter()-began, **compare_original_canary(stage, args.original_canary)))
            return
        launch_wave(args, stage, range(4))
        launch_wave(args, stage, range(4, 5))
        records = {pair["edge"]: json.loads((stage / pair["edge"] / "summary.json").read_text()) for pair in pairs}
        assert all(record["status"] == "completed" for record in records.values())
        write_json(stage / "summary.json", dict(status="completed", mode=settings["mode"],
            elapsed_seconds=time.perf_counter()-began, edges=list(records.values())))

        def cached_initial(run_args, run_config, pair):
            assert run_config == config and run_args.batch_size == args.batch_size
            record = records[pair["edge"]]
            assert record["model_pair"] == pair
            path = stage / pair["edge"] / "calibration.pt"
            assert sha256_file(path) == record["artifact_sha256"]
            return torch.load(path, map_location="cpu", weights_only=False), record["resource"]

        old_initial, old_write = main.initial_calibration, main.write_json

        def write_with_provenance(path, value):
            if path.name == "configuration.json":
                value["initial_calibration_execution"] = dict(
                    directory=str(stage), wrapper_sha256=settings["wrapper_sha256"],
                    configuration_sha256=sha256_file(stage / "configuration.json"),
                    summary_sha256=sha256_file(stage / "summary.json"),
                    scheduler_canary_sha256=settings["scheduler_canary_sha256"],
                    artifact_sha256={edge: record["artifact_sha256"] for edge, record in records.items()})
            old_write(path, value)

        try:
            main.initial_calibration = cached_initial
            main.write_json = write_with_provenance
            # Preserve every original source/canary/data check in the main runner.
            args.canary_record = args.original_canary / "summary.json"
            main.run(args)
        finally:
            main.initial_calibration, main.write_json = old_initial, old_write
    except Exception as exc:
        write_json(stage / "wrapper_failure.json", dict(status="failed", error=repr(exc),
            elapsed_seconds=time.perf_counter()-began))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/insight/large_unified_auc_10k_20260920.json")
    parser.add_argument("--prepared", type=Path, default=DEFAULT_ROOT / "prepared")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_ROOT / "initial_parallel")
    parser.add_argument("--original-canary", type=Path, default=DEFAULT_ROOT / "canary")
    parser.add_argument("--scheduler-canary", type=Path, default=DEFAULT_ROOT / "initial_parallel_canary")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--candidate-chunk", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--initial-worker", action="store_true")
    parser.add_argument("--edge", type=int, default=0)
    args = parser.parse_args()
    initial_worker(args) if args.initial_worker else run(args)
