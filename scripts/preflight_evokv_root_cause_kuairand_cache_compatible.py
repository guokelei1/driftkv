from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from hstu_kvcache.streaming.kuairand_cache_compatible import (
    validate_compatible_document,
)
from hstu_kvcache.streaming.qk_stream_version import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = json.loads(args.config.read_text())
    validate_compatible_document(document)
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    gpus = []
    for line in output.splitlines():
        index, total, used = [int(value.strip()) for value in line.split(",")]
        if index in (0, 1):
            gpus.append({"index": index, "total_mib": total, "used_mib": used})
    if [value["index"] for value in gpus] != [0, 1] or any(
        value["used_mib"] > 1024 for value in gpus
    ):
        raise RuntimeError("KuaiRand cache-compatible GPU preflight failed")
    print(
        json.dumps(
            {
                "status": "ready",
                "config_sha256": file_sha256(args.config),
                "gpus": gpus,
                "qualification_consumed": False,
                "final_consumed": False
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
