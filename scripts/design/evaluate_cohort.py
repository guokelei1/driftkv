#!/usr/bin/env python3
"""Run independent UID chunks through the existing frozen trajectory evaluator."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from design.data import ROOT
from design.run import write_json

from hstu_kvcache.evaluation.binary_metrics import binary_metrics


def main(args):
    out = ROOT / "results/design" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    split_path = (ROOT / args.split_path).resolve()
    split = json.loads(split_path.read_text())
    uids = split["development"][:args.users]
    assert len(uids) == args.users and not set(uids) & set(split["confirmation"])
    reference = ROOT / "results/design" / args.reference
    fitted = json.loads((reference / "configuration.json").read_text())
    args.no_op_targets = sorted(set(args.no_op_targets) | set(fitted.get("no_op_targets",[])))
    assert json.loads((reference / "summary.json").read_text())["status"] == "development_complete"
    assert not set(fitted["calibration_uids"]) & set(uids)
    shards = [dict(offset=offset, users=min(args.chunk_users, args.users-offset),
                   run=f"{args.run_id}/shard_{offset:05d}")
              for offset in range(0, args.users, args.chunk_users)]
    config = dict(vars(args), status="prospective_frozen_trajectory_evaluation", shards=shards,
        development_uids=uids, trajectory_uids=uids, calibration_uids=[], targets=5,
        reserved_confirmation_count=len(split["confirmation"]), confirmation_read=False,
        split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
        calibration_reference=dict(run=args.reference,
            configuration_sha256=hashlib.sha256((reference / "configuration.json").read_bytes()).hexdigest(),
            translator_sha256={str(t):hashlib.sha256((reference / f"translator_v{t}.pt").read_bytes()).hexdigest()
                               for t in range(1, 6) if t not in args.no_op_targets}),
        adaptation_targets=[t for t in range(1,6) if t not in args.no_op_targets],
        no_op_scope="native model/writes/source maintenance continue; no Translator correction at declared targets",
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), ROOT / "scripts/design/run.py", *sorted((ROOT / "src/hstu_kvcache/adaptation").glob("*.py"))]},
        expected_wall_seconds=args.estimate_seconds, expected_gpu_mib_per_worker=22000,
        estimate_basis="512-user control436s includes static diagnostics and compilation; quality-only4-user canary17.69s; conservative per-user allowance plus per-shard IO",
        execution="tmux detached" if args.detached else "direct small canary",
        backbone_seed_repeats=1, cost_target_population=30000, v5_partial_tail=True,
        quality_only=True, write_mode="reuse",
        limitations=["development sample, not final confirmation", "synchronous ready views",
                     "one training seed", "M5 E14_partial diagnostic", "population IO/cost qualification still separate"])
    assert args.estimate_seconds <= 1800 or args.detached
    write_json(out / "configuration.json", config)
    start = time.perf_counter()

    def worker(gpu, assigned):
        completed = []
        for shard in assigned:
            command = [sys.executable, "-u", "scripts/design/run.py", "--run-id", shard["run"],
                "--evaluation-from", args.reference, "--split-path", str(split_path),
                "--dev-offset", str(shard["offset"]), "--dev-users", str(shard["users"]),
                "--trajectory-users", str(shard["users"]), "--quality-only", "--targets", "5",
                "--tail-days", "14", "--replay-chunk", "128", "--release-batch", "16"]
            if args.compile_reads:
                command.append("--compile-reads")
            if args.no_op_targets:
                command.extend(["--no-op-targets", *map(str,args.no_op_targets)])
            if args.detached:
                command.append("--detached")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH="src:scripts")
            log = ROOT / "results/design" / f"{shard['run']}.log"
            with log.open("w") as stream:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            log.with_suffix(".exit").write_text(str(result.returncode) + "\n")
            if result.returncode:
                raise RuntimeError(f"{shard['run']} failed; see {log}")
            completed.append(shard)
            print(json.dumps(dict(phase="shard_complete", gpu=gpu, **shard,
                                  elapsed_seconds=time.perf_counter()-start)), flush=True)
        return completed

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures = [pool.submit(worker, gpu, shards[i::len(args.gpus)]) for i, gpu in enumerate(args.gpus)]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        frames, summaries, ledgers, counts = [], [], defaultdict(float), {}
        for shard in shards:
            path = ROOT / "results/design" / shard["run"]
            child_config = json.loads((path / "configuration.json").read_text())
            assert all(child_config["source_sha256"][key] == value for key, value in config["source_sha256"].items())
            assert child_config["calibration_reference"] == config["calibration_reference"] | {
                "cost_scope": "teacher, fitting and calibration lineage remain in this referenced run; not zero method preparation"}
            expected = uids[shard["offset"]:shard["offset"]+shard["users"]]
            assert child_config["trajectory_uids"] == expected
            raw = pd.read_parquet(path / "quality_raw.parquet")
            assert set(raw.uid) <= set(expected)
            frames.append(raw)
            summary = json.loads((path / "summary.json").read_text())
            summaries.append(summary)
            assert not set(counts) & set(summary["state_counts"])
            counts.update(summary["state_counts"])
            for key, value in summary["ledger_seconds"].items():
                ledgers[key] += value
        raw = pd.concat(frames, ignore_index=True)
        assert not raw.duplicated(["target", "request_id"]).any()
        assert set(map(int, counts)) == set(uids)
        raw.to_parquet(out / "quality_raw.parquet", index=False)
        quality = []
        for target, group in raw.groupby("target", sort=True):
            quality.append(dict(target=int(target), requests=len(group), users=int(group.uid.nunique()),
                metrics={name:binary_metrics(group.label.to_numpy(), group[name].to_numpy())
                         for name in ("exact", "reuse", "learned")}))
        write_json(out / "summary.json", dict(status="development_complete", quality=quality, mechanisms=[],
            configuration_sha256=hashlib.sha256((out / "configuration.json").read_bytes()).hexdigest(),
            elapsed_seconds=time.perf_counter()-start, worker_seconds=sum(s["elapsed_seconds"] for s in summaries),
            peak_allocated_mib=max(s["peak_allocated_mib"] for s in summaries),
            ledger_seconds=dict(ledgers), state_counts=counts, selected_users=len(uids),
            users_with_feedback=int(raw.uid.nunique()), raw_sha256=hashlib.sha256((out / "quality_raw.parquet").read_bytes()).hexdigest(),
            aggregation="concatenate disjoint raw requests, then recompute metrics; never average shard AUC",
            confirmation_read=False, model_seed_repeats=1, limitations=config["limitations"]))
        print(json.dumps(dict(status="development_complete", selected_users=len(uids), requests=len(raw),
                              elapsed_seconds=time.perf_counter()-start)), flush=True)
    except Exception as exc:
        write_json(out / "summary.json", dict(status="failed", error=repr(exc), elapsed_seconds=time.perf_counter()-start))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--split-path", default="data/manifests/evokv_design_medium_6000_v1/split.json")
    parser.add_argument("--users", type=int, default=6000)
    parser.add_argument("--chunk-users", type=int, default=500)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--estimate-seconds", type=int, required=True)
    parser.add_argument("--compile-reads", action="store_true")
    parser.add_argument("--no-op-targets",type=int,nargs="+",default=[],choices=(2,3,4,5))
    parser.add_argument("--detached", action="store_true")
    main(parser.parse_args())
