#!/usr/bin/env python3
"""Compare translation designs on independent, already observed feedback.

Parent-Exact source prefixes are legitimate adjacent calibration scenes, not
the continuous evaluation branch. All feedback predates the target release.
"""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from design.data import DAY, ROOT, frozen_model, histories
from design.run import cache_at_many, new_translator, timed, write_json
from insight_two.common import CUTOVER_DAYS

from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.adaptation.summary import SummaryWriter
from hstu_kvcache.evaluation.binary_metrics import binary_metrics
from hstu_kvcache.models import HSTUKVCache


@torch.no_grad()
def observe(current, previous, history, rows, translators, target, ledger):
    device = next(current.parameters()).device
    outputs = {name:np.zeros(len(rows)) for name in ("reuse", *translators)}
    for offset in range(0, len(rows), 16):
        part = rows.iloc[offset:offset+16]
        requests = list(zip(part.uid.astype(int), part.query_timestamp.astype(int), strict=True))
        caches = timed(lambda requests=requests: cache_at_many(previous, history, requests, 16), ledger, "parent_prefix")
        writers = timed(lambda caches=caches: [SummaryWriter.from_cache(cache, times, target-1,
                        segment_size=1024, slots=1) for cache, times in caches], ledger, "source_summary")
        groups = defaultdict(list)
        for i, (cache, times) in enumerate(caches):
            assert int(times[-1]) < requests[i][1]
            groups[cache.seq_len].append(i)
        features = {}
        for name, mapper in translators.items():
            features[name] = timed(lambda mapper=mapper, writers=writers:
                mapper.writer_features(writers, [0]*len(writers)), ledger, "source_features")
        for length, indices in groups.items():
            cache = HSTUKVCache(torch.cat([caches[i][0].k for i in indices], 1),
                               torch.cat([caches[i][0].v for i in indices], 1), length)
            candidates = torch.tensor(part.iloc[indices].item_idx.to_numpy()[:, None], device=device)
            delta = torch.tensor([requests[i][1]-int(caches[i][1][-1]) for i in indices], device=device, dtype=torch.float32)
            if offset == 0:
                expected = torch.tensor(part.iloc[indices].parent_full.to_numpy()[:, None], device=device)
                torch.testing.assert_close(score(previous, cache, candidates, delta)[0].double(), expected,
                                           atol=2e-5, rtol=2e-5)
            values = timed(lambda cache=cache, candidates=candidates, delta=delta:
                           score(current, cache, candidates, delta)[0], ledger, "reuse_read")
            outputs["reuse"][offset+np.asarray(indices)] = values[:, 0].cpu().numpy()
            for name, mapper in translators.items():
                inputs, counts = features[name]
                inputs, counts = inputs[indices], counts[indices]
                response, temporal = timed(lambda mapper=mapper, inputs=inputs, counts=counts:
                    (mapper.rates(inputs)*counts[:, None, None],
                     mapper.time_coefficients(inputs)*counts[:, None, None, None]), ledger, name+"_translation")
                values = timed(lambda cache=cache, candidates=candidates, delta=delta, response=response, temporal=temporal:
                    score(current, cache, candidates, delta, response_delta=response, response_time_delta=temporal)[0],
                    ledger, name+"_read")
                outputs[name][offset+np.asarray(indices)] = values[:, 0].cpu().numpy()
    return rows.assign(**outputs)


def main(args):
    out = ROOT / "results/design" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    profile = ROOT / "results/design" / "pre_release_auc128_01"
    profile_config = json.loads((profile / "configuration.json").read_text())
    uids = profile_config["calibration_uids"][:args.users]
    references = {"stable":"v9_stratified_stable64_01", "source_time":"v9_stratified_source64_01"}
    configs = {name:json.loads((ROOT / "results/design" / run / "configuration.json").read_text())
               for name, run in references.items()}
    assert all(not set(uids) & set(c["calibration_uids"]) for c in configs.values())
    rows = pd.read_parquet(profile / "pre_release_raw.parquet")
    rows = rows[rows.uid.isin(uids) & (rows.target <= args.targets)]
    config = dict(vars(args), profile=str(profile.relative_to(ROOT)), calibration_uids=uids,
        profile_raw_sha256=hashlib.sha256((profile / "pre_release_raw.parquet").read_bytes()).hexdigest(),
        references={name:dict(run=run, configuration_sha256=hashlib.sha256(
            (ROOT / "results/design" / run / "configuration.json").read_bytes()).hexdigest(),
            translator_sha256={str(t):hashlib.sha256((ROOT / "results/design" / run / f"translator_v{t}.pt").read_bytes()).hexdigest()
                               for t in range(1, args.targets+1)})
            for name, run in references.items()},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="pre-release observed-label adjacent calibration only; not continuous method quality or admission",
        expected_seconds=[30, 120] if args.users > 8 else [10, 40], expected_gpu_mib=5000,
        confirmation_read=False)
    write_json(out / "configuration.json", config)
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    start, ledger = time.perf_counter(), defaultdict(float)
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda:0")
        history = timed(lambda: histories(uids, CUTOVER_DAYS[args.targets-1]), ledger, "history_io")
        previous = timed(lambda: frozen_model(0, device), ledger, "model_io")
        frames, records = [], []
        for target in range(1, args.targets+1):
            current = timed(lambda target=target: frozen_model(target, device), ledger, "model_io")
            part = rows[rows.target == target]
            assert part.query_timestamp.max() < CUTOVER_DAYS[target-1]*DAY
            translators = {}
            for name, run in references.items():
                mapper = new_translator(SimpleNamespace(**configs[name]), target, device)
                saved = torch.load(ROOT / "results/design" / run / f"translator_v{target}.pt", map_location=device, weights_only=False)
                mapper.load_state_dict(saved["state_dict"])
                mapper.supported_producers = set(saved["supported_producers"])
                translators[name] = mapper.eval()
            observed = observe(current, previous, history, part, translators, target, ledger)
            frames.append(observed)
            metrics = {name:binary_metrics(observed.label.to_numpy(), observed[name].to_numpy())
                       for name in ("current_full", "reuse", "stable", "source_time")}
            records.append(dict(target=target, requests=len(part), users=int(part.uid.nunique()), metrics=metrics,
                teacher_logit_mse={name:float(np.square(observed[name]-observed.current_full).mean())
                                  for name in ("reuse", "stable", "source_time")}))
            print(json.dumps(records[-1]), flush=True)
            pd.concat(frames, ignore_index=True).to_parquet(out / "calibration_raw.parquet", index=False)
            previous = current
        write_json(out / "summary.json", dict(status="development_complete", calibration_feedback=records,
            ledger_seconds=dict(ledger), elapsed_seconds=time.perf_counter()-start,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/(1<<20),
            passing_checks="Parent Full reference scores and strict pre-release/prefix timestamps",
            confirmation_read=False))
    except Exception as exc:
        write_json(out / "summary.json", dict(status="failed", error=repr(exc), ledger_seconds=dict(ledger)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--users", type=int, default=128)
    parser.add_argument("--targets", type=int, choices=(1, 2, 3, 4, 5), default=5)
    main(parser.parse_args())
