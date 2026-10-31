"""Compare the two Max alignment epochs on matched pre-release validation only."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .data_common import json_save, sha256
from .protocol import paired_uid_interval
from .version_chain import read_json


def load_validation(directory):
    summary = read_json(directory / "summary.json")
    if (summary["status"] != "passed" or summary["role"] != 1 or summary["days"] != [24, 25, 26]
            or summary["history_limit"] != 1024 or summary["first_only"] or summary["requests"] != 4096):
        raise ValueError("Expected the fixed pre-release Max validation window/population")
    paths = sorted(directory.glob("rank_*.csv"))
    integers = {"uid", "rid", "day", "candidates", "exposed", "positives", "history_length"}
    rows = {}
    for path in paths:
        with path.open() as handle:
            for raw in csv.DictReader(handle):
                row = {key: (int(value) if key in integers else float(value)) if value else None
                       for key, value in raw.items()}
                key = tuple(row[name] for name in ["uid", "day", "rid"])
                if key in rows:
                    raise ValueError("Duplicate distributed validation request")
                rows[key] = row
    if len(rows) != summary["requests"]:
        raise ValueError("Incomplete validation score population")
    return summary, rows, {str(path): sha256(path) for path in paths}


def compare(first, second):
    if first.keys() != second.keys():
        raise ValueError("Validation checkpoints have different request populations")
    fields = ["candidates", "exposed", "positives", "history_length"]
    if any(first[key][name] != second[key][name] for key in first for name in fields):
        raise ValueError("Validation input contexts or feedback changed between epochs")
    positive = [key for key in first if first[key]["positives"] > 0]
    if not positive:
        raise ValueError("No shared positive requests")
    gap = paired_uid_interval([key[0] for key in positive],
                              [second[key]["current_ndcg@10"] for key in positive],
                              [first[key]["current_ndcg@10"] for key in positive])
    observed = [key for key in first if first[key]["exposed"] > 0]
    return {"epoch2_minus_epoch1_ndcg10": gap,
            "epoch1_ndcg10": float(np.mean([first[key]["current_ndcg@10"] for key in positive])),
            "epoch2_ndcg10": float(np.mean([second[key]["current_ndcg@10"] for key in positive])),
            "epoch1_observed_bce": float(np.mean([first[key]["current_observed_bce"] for key in observed])),
            "epoch2_observed_bce": float(np.mean([second[key]["current_observed_bce"] for key in observed]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run / "native_learning_analysis.json"
    if output.exists():
        raise ValueError("Retain the existing analysis; use a new output for a new analysis")
    summary1, rows1, hashes1 = load_validation(args.run / "alignment_epoch1_validation")
    summary2, rows2, hashes2 = load_validation(args.run / "alignment_epoch2_validation")
    if summary1["source_binding"] != summary2["source_binding"] or summary1["evaluator_sha256"] != summary2["evaluator_sha256"]:
        raise ValueError("Validation sources/backends differ")
    result = compare(rows1, rows2)
    per_day = {str(day): compare({key: row for key, row in rows1.items() if key[1] == day},
                                {key: row for key, row in rows2.items() if key[1] == day}) for day in [24, 25, 26]}
    result.update(status="passed", scope="fixed pre-release validation; no cache scores or confirmation inputs",
                  per_day=per_day, epoch1_checkpoint_sha256=summary1["checkpoint_sha256"],
                  epoch2_checkpoint_sha256=summary2["checkpoint_sha256"], source_csv_sha256=hashes1 | hashes2,
                  analysis_sha256=sha256(__file__), sample_sufficiency_proven=False,
                  interpretation="Two checkpoints can show learning or overfitting; they cannot prove the optimum data amount")
    json_save(output, result)
    print(result["epoch2_minus_epoch1_ndcg10"])


if __name__ == "__main__":
    main()
