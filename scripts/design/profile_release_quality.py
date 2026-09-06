#!/usr/bin/env python3
"""Causal release features from already observed, pre-release feedback.

These are Parent/Current Full metrics on separate method-calibration users.
They are not persistent-Reuse quality, serving admission or backbone holdout.
"""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from design.data import DAY, ROOT, frozen_model, histories, quality_requests, stratified_calibration
from design.run import cache_at_many, timed, write_json
from insight_two.common import CUTOVER_DAYS

from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.evaluation.binary_metrics import binary_metrics, sigmoid
from hstu_kvcache.models import HSTUKVCache


@torch.no_grad()
def full_scores(model, history, rows, ledger, key):
    device = next(model.parameters()).device
    result = np.zeros(len(rows))
    for offset in range(0, len(rows), 16):
        part = rows.iloc[offset:offset+16]
        requests = list(zip(part.uid.astype(int), part.query_timestamp.astype(int), strict=True))
        caches = timed(lambda requests=requests: cache_at_many(model, history, requests, 16), ledger, key+"_prefix")
        groups = defaultdict(list)
        for i, (cache, times) in enumerate(caches):
            assert int(times[-1]) < requests[i][1]
            groups[cache.seq_len].append(i)
        for length, indices in groups.items():
            combined = HSTUKVCache(torch.cat([caches[i][0].k for i in indices], 1),
                                  torch.cat([caches[i][0].v for i in indices], 1), length)
            candidates = torch.tensor(part.iloc[indices].item_idx.to_numpy()[:, None], device=device)
            delta = torch.tensor([requests[i][1]-int(caches[i][1][-1]) for i in indices], device=device, dtype=torch.float32)
            observed = timed(lambda combined=combined, candidates=candidates, delta=delta:
                             score(model, combined, candidates, delta)[0], ledger, key+"_read")
            if offset == 0:
                # One real scalar native reference for each length in the first
                # batch checks candidate/prefix association and read semantics.
                i = indices[0]
                expected = model.score_cc_reuse(caches[i][0], candidates[:1], delta[:1])
                torch.testing.assert_close(observed[:1], expected, atol=2e-5, rtol=2e-5)
            result[offset+np.asarray(indices)] = observed[:, 0].cpu().numpy()
    return result


def main(args):
    out = ROOT / "results/design" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    split_path = ROOT / "data/manifests/evokv_design_medium_6000_v1/split.json"
    split = json.loads(split_path.read_text())
    reference = ROOT / "results/design" / args.reference
    fitted = json.loads((reference / "configuration.json").read_text())["calibration_uids"]
    available = dict(split, calibration=[u for u in split["calibration"] if u not in set(fitted)])
    uids = stratified_calibration(available, args.users)
    config = dict(vars(args), calibration_uids=uids, fitted_uids_excluded=fitted,
        split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
        windows=[[day-14, day] for day in CUTOVER_DAYS[:args.targets]],
        causality="feedback timestamp strictly before release; each Full prefix strictly before its own request",
        scope="prospective method features only; not persistent-Reuse evaluation, admission or backbone holdout",
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), ROOT / "scripts/design/data.py", ROOT / "scripts/design/run.py"]},
        expected_seconds=[30, 120] if args.users > 8 else [10, 40], expected_gpu_mib=5000,
        confirmation_read=False)
    write_json(out / "configuration.json", config)
    start, ledger = time.perf_counter(), defaultdict(float)
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda:0")
        history = timed(lambda: histories(uids, CUTOVER_DAYS[args.targets-1]), ledger, "history_io")
        previous = timed(lambda: frozen_model(0, device), ledger, "model_io")
        records, raw = [], []
        for target in range(1, args.targets+1):
            cutover = CUTOVER_DAYS[target-1]*DAY
            current = timed(lambda target=target: frozen_model(target, device), ledger, "model_io")
            grouped = quality_requests(uids, cutover-14*DAY, cutover)
            rows = pd.DataFrame([row for values in grouped.values() for row in values])
            assert len(rows) and rows.query_timestamp.max() < cutover
            parent_scores = full_scores(previous, history, rows, ledger, "parent_full")
            current_scores = full_scores(current, history, rows, ledger, "current_full")
            labels = rows.label.to_numpy()
            parent = binary_metrics(labels, parent_scores)
            child = binary_metrics(labels, current_scores)
            row = dict(target=target, requests=len(rows), users=int(rows.uid.nunique()),
                negatives=int((labels==0).sum()), parent_full=parent, current_full=child,
                full_auc_change=(child["ROC_AUC"]-parent["ROC_AUC"] if child["ROC_AUC"] is not None else None),
                signed_probability_shift=float((sigmoid(current_scores)-sigmoid(parent_scores)).mean()),
                logit_change_rms=float(np.square(current_scores-parent_scores).mean()**.5),
                maximum_feedback_timestamp=int(rows.query_timestamp.max()), release_timestamp=cutover)
            records.append(row)
            raw.extend(rows.assign(target=target, parent_full=parent_scores, current_full=current_scores).to_dict("records"))
            write_json(out / "release_features.json", records)
            print(json.dumps(row), flush=True)
            previous = current
        pd.DataFrame(raw).to_parquet(out / "pre_release_raw.parquet", index=False)
        write_json(out / "summary.json", dict(status="development_complete", release_features=records,
            ledger_seconds=dict(ledger), elapsed_seconds=time.perf_counter()-start,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/(1<<20),
            configuration_sha256=hashlib.sha256((out / "configuration.json").read_bytes()).hexdigest(),
            confirmation_read=False, passing_checks="first-batch native references and strict timestamp separation"))
        print(json.dumps(dict(status="development_complete", elapsed_seconds=time.perf_counter()-start)), flush=True)
    except Exception as exc:
        write_json(out / "summary.json", dict(status="failed", error=repr(exc), ledger_seconds=dict(ledger)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reference", default="v9_stratified_stable64_01")
    parser.add_argument("--users", type=int, default=128)
    parser.add_argument("--targets", type=int, choices=(1, 2, 3, 4, 5), default=5)
    main(parser.parse_args())
