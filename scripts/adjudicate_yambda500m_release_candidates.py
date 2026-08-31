#!/usr/bin/env python3
"""Join labels after a Full-only raw seal and report candidate release quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from hstu_kvcache.evaluation import binary_metrics, paired_harm
from hstu_kvcache.evaluation.raw_protocol import sha256_file


def score_views(rows: list[dict], labels: np.ndarray) -> dict:
    logits = np.asarray([row["hstu_logit"] for row in rows])
    return {"hstu_native": binary_metrics(labels, logits)}


def oov_bucket(value: float) -> str:
    if value < 0.01:
        return "lt_1pct"
    if value < 0.05:
        return "1_to_5pct"
    return "gte_5pct"


def prefix_bucket(value: int) -> str:
    if value < 32:
        return "lt_32"
    if value < 128:
        return "32_to_127"
    if value < 512:
        return "128_to_511"
    return "512"


def paired_release(parent_rows: list[dict], current_rows: list[dict], labels: np.ndarray, namespace: str) -> dict:
    if [row["request_id"] for row in parent_rows] != [row["request_id"] for row in current_rows]:
        raise RuntimeError("Parent/Current rows are not aligned")
    uids = np.asarray([row["uid"] for row in parent_rows])
    parent = np.asarray([row["hstu_logit"] for row in parent_rows])
    current = np.asarray([row["hstu_logit"] for row in current_rows])
    # Existing paired_harm implements L(left)-L(right); here it is release gain.
    result = paired_harm(
        uids=uids, labels=labels, reuse_logits=parent, current_logits=current,
        namespace=namespace,
    )
    return {"parent_minus_current_log_loss": result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seal = json.loads(args.seal.read_text())
    if seal["contains_reuse"]:
        raise RuntimeError("release-only adjudicator refuses a Reuse artifact")
    if seal.get("architecture") != "hstu_native_cc":
        raise RuntimeError("release-only adjudicator accepts HSTU-native raw only")
    if seal["raw_sha256"] != sha256_file(args.raw):
        raise RuntimeError("raw no longer matches its seal")
    label_table = pq.read_table(args.labels, columns=["request_id", "label"])
    labels_by_request = dict(zip(label_table["request_id"].to_pylist(), label_table["label"].to_pylist(), strict=True))
    raw = pq.read_table(args.raw).to_pylist()
    parent_name = seal["parent"]
    by_model = {
        name: sorted((row for row in raw if row["model_name"] == name), key=lambda row: row["request_id"])
        for name in [parent_name, *seal["currents"]]
    }
    parent_rows = by_model[parent_name]
    labels = np.asarray([labels_by_request[row["request_id"]] for row in parent_rows], dtype=np.int64)
    if any(row["request_id"] not in labels_by_request for row in raw):
        raise RuntimeError("incomplete label join")

    report = {
        "status": "release_candidate_full_only_adjudicated",
        "raw_sha256": seal["raw_sha256"], "stage": seal["stage"],
        "evaluation_block": seal["evaluation_block"], "training_block": seal["training_block"],
        "parent": parent_name, "parent_absolute": score_views(parent_rows, labels), "candidates": {},
    }
    for candidate in seal["currents"]:
        rows = by_model[candidate]
        report["candidates"][candidate] = {
            "checkpoint_progress": rows[0]["checkpoint_progress"],
            "training_epochs_completed": rows[0].get("training_epochs_completed"),
            "absolute": score_views(rows, labels),
            "paired_release_gain": paired_release(parent_rows, rows, labels, f"release:{seal['stage']}:{candidate}"),
            "daily": {}, "cohorts": {"recurring_user": {}, "history_oov": {}, "prefix_length": {}},
        }
        groupings = {
            "daily": lambda row: str(row["query_timestamp"] // 86_400),
            "recurring_user": lambda row: "recurring" if row["recurring_user"] else "new_to_window",
            "history_oov": lambda row: oov_bucket(row["history_oov_fraction"]),
            "prefix_length": lambda row: prefix_bucket(row["history_length"]),
        }
        for grouping, key_fn in groupings.items():
            for key in sorted({key_fn(row) for row in rows}):
                indices = [index for index, row in enumerate(rows) if key_fn(row) == key]
                candidate_rows = [rows[index] for index in indices]
                selected_parent = [parent_rows[index] for index in indices]
                selected_labels = labels[indices]
                payload = {
                    "requests": len(indices), "users": len({row["uid"] for row in candidate_rows}),
                    "label_rate": float(selected_labels.mean()),
                    "parent": score_views(selected_parent, selected_labels),
                    "current": score_views(candidate_rows, selected_labels),
                }
                if grouping == "daily":
                    report["candidates"][candidate]["daily"][key] = payload
                else:
                    report["candidates"][candidate]["cohorts"][grouping][key] = payload
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "candidates": list(report["candidates"])}, indent=2))


if __name__ == "__main__":
    main()
