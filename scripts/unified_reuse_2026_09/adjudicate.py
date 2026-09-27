#!/usr/bin/env python3
"""Compare sealed Reuse shards with retained Full scores on identical requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / "src")]
from common import DEFAULT_OUTPUT, DAY, binding, sha256, write_json
from hstu_kvcache.evaluation import binary_metrics


def adjudicate(scale: str, edge: int, output_root: Path) -> dict:
    b = binding(scale, edge)
    directory = output_root / scale / b["edge"]
    bound = json.loads((directory / "binding.json").read_text())
    for name in ("full_raw", "labels"):
        if sha256(b[name]) != bound["sources"][name]["sha256"]:
            raise RuntimeError(f"retained {name} changed after Reuse preparation")
    reuse: dict[str, float] = {}
    shards = []
    execution_sources = None
    for rank in range(4):
        rank_dir = directory / f"rank{rank}"
        complete = json.loads((rank_dir / "complete.json").read_text())
        if complete["status"] != "complete" or complete["requests"] != bound["ranks"][rank]["requests"]:
            raise RuntimeError(f"rank {rank} Reuse output is not complete")
        if execution_sources is None:
            execution_sources = complete["execution_source_hashes"]
        elif complete["execution_source_hashes"] != execution_sources:
            raise RuntimeError("four Reuse ranks used different scoring source code")
        for path in sorted(rank_dir.glob("shard_*.parquet")):
            seal = json.loads(path.with_suffix(".seal.json").read_text())
            if seal["sha256"] != sha256(path) or seal["source_signature"] != complete["source_signature"]:
                raise RuntimeError(f"Reuse shard changed: {path}")
            table = pq.read_table(path, columns=["request_id", "hstu_logit"])
            for request_id, score in zip(table["request_id"].to_pylist(), table["hstu_logit"].to_pylist(), strict=True):
                if request_id in reuse:
                    raise RuntimeError(f"duplicate Reuse request: {request_id}")
                reuse[request_id] = float(score)
            shards.append({"path": str(path.relative_to(ROOT)), "sha256": seal["sha256"], "requests": len(table)})
    full = pq.read_table(b["full_raw"], columns=["request_id", "model_name", "hstu_logit"])
    selected = [(request_id, float(score)) for request_id, name, score in zip(
        full["request_id"].to_pylist(), full["model_name"].to_pylist(),
        full["hstu_logit"].to_pylist(), strict=True
    ) if name == b["full_current_name"]]
    full_scores = dict(selected)
    if len(selected) != len(full_scores) or full_scores.keys() != reuse.keys():
        raise RuntimeError("Reuse and retained Current Full request IDs differ")
    start, end = b["days"]
    quality = pq.read_table(b["labels"], filters=[
        ("time_block", "=", "matrix_horizon"), ("target_known", "=", True),
        ("query_timestamp", ">=", start * DAY), ("query_timestamp", "<", end * DAY),
    ], columns=["request_id", "label"])
    labels = dict(zip(quality["request_id"].to_pylist(), quality["label"].to_pylist(), strict=True))
    if labels.keys() != reuse.keys():
        raise RuntimeError("quality labels and evaluated request IDs differ")
    ordered = sorted(reuse)
    target = np.asarray([labels[request_id] for request_id in ordered], dtype=np.int64)
    full_logits = np.asarray([full_scores[request_id] for request_id in ordered], dtype=np.float64)
    reuse_logits = np.asarray([reuse[request_id] for request_id in ordered], dtype=np.float64)
    full_metrics = binary_metrics(target, full_logits)
    reuse_metrics = binary_metrics(target, reuse_logits)
    adjudication = json.loads(b["full_raw"].with_name("adjudication.json").read_text())
    recorded_auc = adjudication["candidates"][b["full_current_name"]]["absolute"]["hstu_native"]["ROC_AUC"]
    if abs(full_metrics["ROC_AUC"] - recorded_auc) > 1e-10:
        raise RuntimeError("retained Full AUC does not reproduce its original adjudication")
    report = {"status": "complete", "scale": scale, "edge": b["edge"], "days": b["days"],
              "requests": len(ordered), "users": bound["users"],
              "current_full": full_metrics, "current_reuse": reuse_metrics,
              "exact_minus_reuse_auc_pp": 100 * (full_metrics["ROC_AUC"]-reuse_metrics["ROC_AUC"]),
              "reuse_minus_exact_log_loss": reuse_metrics["log_loss"]-full_metrics["log_loss"],
              "full_raw_sha256": bound["sources"]["full_raw"]["sha256"],
              "full_adjudication_sha256": sha256(b["full_raw"].with_name("adjudication.json")),
              "reuse_execution_source_hashes": execution_sources, "reuse_shards": shards}
    write_json(directory / "summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", required=True, choices=["medium", "large", "max"])
    parser.add_argument("--edge", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = adjudicate(args.scale, args.edge, args.output_root)
    print(json.dumps({key: result[key] for key in ("scale", "edge", "requests", "exact_minus_reuse_auc_pp")}, ensure_ascii=False))
