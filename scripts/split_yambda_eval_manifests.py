#!/usr/bin/env python3
"""Deterministically split an untrained candidate manifest into dev/qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def uid_bucket(uid: int) -> int:
    return ((uid * 2654435761) & 0xFFFFFFFF) % 16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines()]
    dev = [row for row in rows if uid_bucket(int(row["uid"])) == 0]
    qualification = [row for row in rows if uid_bucket(int(row["uid"])) != 0]
    for path, selected, usage in (
        (args.dev, dev, "theta0_dev_evaluation"),
        (args.qualification, qualification, "theta0_qualification_evaluation"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as stream:
            for row in selected:
                row = dict(row)
                row["manifest_usage"] = usage
                row["paper_eligible"] = True
                row["checkpoint_selection_eligible"] = usage == "theta0_dev_evaluation"
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    metadata = {
        "source": str(args.input),
        "split": "uid_hash_v1",
        "dev_rows": len(dev),
        "qualification_rows": len(qualification),
        "dev_rule": "uid_hash_v1_bucket == 0 of 16",
        "training_reads these manifests": False,
    }
    args.dev.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    args.qualification.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
