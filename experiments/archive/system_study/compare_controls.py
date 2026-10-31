#!/usr/bin/env python3
"""Match independent GPU runners before combining service and policy evidence."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import pyarrow.parquet as pq

from hstu_kvcache.common.artifacts import sha256, write_json

IDENTITY = ("uid", "version", "label", "item_idx", "query_timestamp")


def matched_control(left, right, *, atol=3e-5, rtol=3e-5):
    def index(rows):
        result = {}
        for row in rows:
            key = row["protocol"], row["request_id"]
            if key in result:
                raise ValueError(f"duplicate request: {key}")
            result[key] = row
        return result

    a, b = index(left), index(right)
    if not a or a.keys() != b.keys():
        raise ValueError("control runners used different or empty request sets")
    errors = {}
    for key, row in a.items():
        other = b[key]
        if any(row[field] != other[field] for field in IDENTITY):
            raise ValueError(f"control identity differs: {key}")
        score, expected = float(row["score"]), float(other["score"])
        if not np.isfinite(score) or not np.isfinite(expected):
            raise ValueError("nonfinite control score")
        error = abs(score - expected)
        if error > atol + rtol * abs(expected):
            raise ValueError(f"independent control score mismatch: {key} / {error}")
        result = errors.setdefault(row["protocol"], dict(candidate_scores=0, max_abs=0.))
        result["candidate_scores"] += 1
        result["max_abs"] = max(result["max_abs"], error)
    return errors


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    summaries = {name: json.loads((folder / "summary.json").read_text())
                 for name, folder in (("service", args.service), ("baseline", args.baseline))}
    if summaries["service"]["status"] != "passed_gpu_global_replay_correctness":
        raise ValueError("service smoke did not complete successfully")
    # The baseline exposes its endpoint checks in each protocol; the source
    # status must also be a successful terminal state, never a partial file.
    if not str(summaries["baseline"]["status"]).startswith(("passed", "complete")):
        raise ValueError("baseline smoke did not complete successfully")
    paths = {name: folder / "scores.parquet"
             for name, folder in (("service", args.service), ("baseline", args.baseline))}
    rows = {name: pq.read_table(path).to_pylist() for name, path in paths.items()}
    comparisons = {}
    for method in ("reuse", "rebuild"):
        left = [row for row in rows["service"] if row["method"] == method]
        right = [row for row in rows["baseline"] if row["method"] == method]
        comparisons[method] = matched_control(left, right)
    # Every evaluated policy must score the same task identities as its own
    # reference; labels and items cannot silently change across methods.
    for name, values in rows.items():
        reference = [row for row in values if row["method"] == "reuse"]
        for method in {row["method"] for row in values}:
            policy = [dict(row, score=0.) for row in values if row["method"] == method]
            matched_control(policy, [dict(row, score=0.) for row in reference])
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="passed_independent_control_match",
        created_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        comparisons=comparisons, all_policy_task_identities_equal=True,
        sources={name: dict(path=str(path), sha256=sha256(path),
            summary_sha256=sha256(path.parent / "summary.json")) for name, path in paths.items()},
        scope="Four-user implementation linkage, not a population quality or loaded-serving conclusion")
    write_json(args.output / "summary.json", report)
    lines = ["# 两个GPU入口的原生对照核对", "",
             "同一任务的请求、用户、版本、标签、候选和时间完全匹配；全部政策保留同一请求集。", "",
             "| 对照 | 协议 | 候选评分 | 最大logit差 |", "| --- | --- | ---: | ---: |"]
    for method, protocols in comparisons.items():
        for protocol, result in protocols.items():
            lines.append(f"| {method} | {protocol} | {result['candidate_scores']} | {result['max_abs']:.9g} |")
    lines += ["", "这是4用户正确性联通证据，不据此选择方案或宣称总体质量优势。", ""]
    (args.output / "summary.md").write_text("\n".join(lines))
    print(json.dumps(dict(status=report["status"], output=str(args.output))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
