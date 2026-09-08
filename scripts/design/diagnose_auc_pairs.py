#!/usr/bin/env python3
"""Explain existing AUC with exact positive-negative pair accounting; no model run."""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]


def credit(margin):
    return (margin > 0).astype(np.float64) + .5 * (margin == 0)


def accounting(frame, target):
    frame = frame.copy()
    methods = ["scheme9_stable", "scheme9_source", "scheme15"]
    for method in methods:
        residual = frame[method]-frame.exact
        frame[method+"_bias"] = residual.groupby(frame.uid).transform("mean")
        frame[method+"_within"] = residual-frame[method+"_bias"]
    positive = frame[frame.label == 1]
    negative = frame[frame.label == 0]
    columns = ["exact", "reuse", "scheme9_stable", "scheme9_source", "scheme15"]
    totals = defaultdict(lambda: defaultdict(float))
    bins = [0, .01, .1, 1, np.inf]
    for start in range(0, len(positive), 64):
        p = positive.iloc[start:start+64]
        margins = {c: p[c].to_numpy()[:, None] - negative[c].to_numpy()[None] for c in columns}
        credits = {c: credit(margins[c]) for c in columns}
        bias = {c: p[c+"_bias"].to_numpy()[:, None]-negative[c+"_bias"].to_numpy()[None] for c in methods}
        within = {c: p[c+"_within"].to_numpy()[:, None]-negative[c+"_within"].to_numpy()[None] for c in methods}
        for c in methods:
            np.testing.assert_allclose(margins[c]-margins["exact"], bias[c]+within[c], atol=1e-12, rtol=1e-10)
        exact, reuse = credits["exact"], credits["reuse"]
        repaired = exact > reuse
        harmed = exact < reuse
        same = p.uid.to_numpy()[:, None] == negative.uid.to_numpy()[None]
        margin_bin = np.searchsorted(bins[1:-1], np.abs(margins["exact"]), side="right")
        for relation, uid_mask in (("same_uid", same), ("cross_uid", ~same)):
            for bin_index in range(4):
                mask = uid_mask & (margin_bin == bin_index)
                row = totals[relation, bin_index]
                row["pairs"] += int(mask.sum())
                row["exact_repaired_pairs"] += int((mask & repaired).sum())
                row["exact_harmed_pairs"] += int((mask & harmed).sum())
                row["exact_repaired_credit"] += float(np.maximum(exact - reuse, 0)[mask].sum())
                for c in columns:
                    row[c + "_correct_credit"] += float(credits[c][mask].sum())
                    if c not in ("exact", "reuse"):
                        difference = credits[c] - reuse
                        row[c + "_recovered_repaired_credit"] += float(np.maximum(difference, 0)[mask & repaired].sum())
                        row[c + "_net_on_repaired_credit"] += float(difference[mask & repaired].sum())
                        row[c + "_damaged_joint_correct_credit"] += float(
                            np.maximum(reuse - credits[c], 0)[mask & (exact == 1) & (reuse == 1)].sum())
                        row[c + "_gained_credit"] += float(np.maximum(difference, 0)[mask].sum())
                        row[c + "_lost_credit"] += float(np.maximum(-difference, 0)[mask].sum())
                        for event, selected in (("repaired_recovered", repaired & (credits[c] > reuse)),
                                ("repaired_missed", repaired & (credits[c] < exact)),
                                ("joint_correct_damaged", (exact == 1) & (reuse == 1) & (credits[c] < reuse))):
                            selected = selected & mask
                            row[c+"_"+event+"_pairs"] += int(selected.sum())
                            row[c+"_"+event+"_bias_dominant"] += int((selected & (np.abs(bias[c]) > np.abs(within[c]))).sum())
                            row[c+"_"+event+"_bias_against_exact"] += int((selected & (bias[c]*margins["exact"] < 0)).sum())
                            row[c+"_"+event+"_within_against_exact"] += int((selected & (within[c]*margins["exact"] < 0)).sum())
    pair_count = len(positive) * len(negative)
    assert int(sum(r["pairs"] for r in totals.values())) == pair_count
    auc = {}
    for c in columns:
        auc[c] = sum(r[c + "_correct_credit"] for r in totals.values()) / pair_count
        np.testing.assert_allclose(auc[c], roc_auc_score(frame.label, frame[c]), atol=1e-12, rtol=0)
    rows = [dict(target=target, relation=relation, exact_abs_margin=[bins[b], bins[b+1] if b < 3 else None],
                 **row) for (relation, b), row in totals.items()]
    return dict(target=target, requests=len(frame), users=int(frame.uid.nunique()),
                positive_negative_pairs=pair_count, auc=auc, rows=rows)


def main(args):
    out = ROOT / "results/design/analysis" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    runs = dict(scheme9_stable="v9_stratified_stable_quality6000_01",
                scheme9_source="v9_stratified_source_quality6000_01",
                scheme15="v15_query_view_four_edge_quality6000_01")
    keys = ["target", "uid", "timestamp", "request_id"]
    hashes, frame = {}, None
    start = time.perf_counter()
    for name, run_id in runs.items():
        path = ROOT / "results/design" / run_id / "quality_raw.parquet"
        data = pd.read_parquet(path).sort_values(keys).reset_index(drop=True)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if frame is None:
            frame = data[keys + ["label", "exact", "reuse"]].copy()
        else:
            assert frame[keys + ["label", "exact", "reuse"]].equals(data[keys + ["label", "exact", "reuse"]])
        frame[name] = data.learned
    results = []
    for target in (1, 3, 4, 5):
        results.append(accounting(frame[frame.target == target], target))
        print(json.dumps(dict(target=target, auc=results[-1]["auc"])), flush=True)
    report = dict(status="analysis_complete", results=results, raw_sha256=hashes,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  elapsed_seconds=time.perf_counter()-start, confirmation_read=False,
                  scope="existing development requests; exact all-pair decomposition, ties half-credit; no selector or new inference",
                  no_op_targets=[2], m5_scope="E14_partial", frozen_backbone_seeds=[17])
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "source.py").write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    main(parser.parse_args())
