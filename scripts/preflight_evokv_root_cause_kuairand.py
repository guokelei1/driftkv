from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from hstu_kvcache.streaming.kuairand_root_cause import load_plan, validate_document
from hstu_kvcache.streaming.qk_stream_version import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("training", "evaluation"), required=True)
    return parser.parse_args()


def gpu_memory() -> list[dict[str, int]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [
        {"index": int(parts[0]), "total_mib": int(parts[1]), "used_mib": int(parts[2])}
        for line in output.splitlines()
        if (parts := [value.strip() for value in line.split(",")])
    ]


def main() -> None:
    args = parse_args()
    document = json.loads(args.config.read_text())
    validate_document(document)
    plan, metadata = load_plan(document)
    expected = [0] if args.phase == "training" else [0, 1]
    memory = gpu_memory()
    selected = [value for value in memory if value["index"] in expected]
    if [value["index"] for value in selected] != expected:
        raise RuntimeError("required KuaiRand GPUs are absent")
    if any(value["used_mib"] > 1024 for value in selected):
        raise RuntimeError("required KuaiRand GPU is not free")
    free = shutil.disk_usage(Path.cwd()).free
    if free < 20 * 1024**3:
        raise RuntimeError("KuaiRand root-cause disk preflight failed")
    print(
        json.dumps(
            {
                "status": "ready",
                "phase": args.phase,
                "config_sha256": file_sha256(args.config),
                "campaign_sha256": document["campaign"]["sha256"],
                "gpus": selected,
                "disk_free_bytes": free,
                "data": metadata,
                "model_context_length": plan.max_seq_len,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
