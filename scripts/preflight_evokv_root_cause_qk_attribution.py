from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

from hstu_kvcache.streaming.qk_stream_runner import _atomic_json
from hstu_kvcache.streaming.qk_stream_version import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _memory_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("host memory availability is absent")


def main() -> None:
    config_path = parse_args().config
    document = json.loads(config_path.read_text())
    execution = document["execution"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != execution["cuda_visible_devices"]:
        raise RuntimeError("QK attribution CUDA visibility differs")
    if torch.cuda.device_count() != execution["world_size"]:
        raise RuntimeError("QK attribution visible GPU count differs")
    campaign = document["campaign"]
    checks = {
        "campaign": {
            "path": campaign["path"],
            "expected_sha256": campaign["sha256"],
            "observed_sha256": file_sha256(Path(campaign["path"])),
        },
        "corpus": {
            "path": document["data"]["corpus"],
            "expected_sha256": document["data"]["corpus_sha256"],
            "observed_sha256": file_sha256(Path(document["data"]["corpus"])),
        },
    }
    for name, checkpoint in document["checkpoints"].items():
        manifest = Path(checkpoint["root"]) / f"theta_{checkpoint['version']}" / "manifest.json"
        checks[name] = {
            "path": str(manifest),
            "expected_sha256": checkpoint["manifest_sha256"],
            "observed_sha256": file_sha256(manifest),
        }
    if "canary" in document:
        for name in ("result", "decision"):
            path = Path(document["canary"][name])
            checks[f"canary_{name}"] = {
                "path": str(path),
                "expected_sha256": document["canary"][f"{name}_sha256"],
                "observed_sha256": file_sha256(path),
            }
    if any(value["expected_sha256"] != value["observed_sha256"] for value in checks.values()):
        raise RuntimeError("QK attribution preflight artifact hash differs")
    hbm = []
    for device in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(device)
        properties = torch.cuda.get_device_properties(device)
        hbm.append(
            {
                "visible_device": device,
                "name": properties.name,
                "uuid": str(properties.uuid),
                "free_bytes": int(free),
                "total_bytes": int(total),
            }
        )
    if any(value["free_bytes"] < execution["minimum_free_hbm_bytes_per_rank"] for value in hbm):
        raise RuntimeError("QK attribution preflight HBM is insufficient")
    round_root = Path(document["outputs"]["round_root"])
    round_root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(round_root)
    if disk.free < execution["minimum_free_disk_bytes"]:
        raise RuntimeError("QK attribution preflight disk is insufficient")
    result = {
        "protocol": document["protocol"],
        "status": "preflight_passed",
        "round_id": document["round_id"],
        "scope": document["scope"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hbm": hbm,
        "host_memory_available_bytes": _memory_available_bytes(),
        "disk": {"free_bytes": disk.free, "total_bytes": disk.total},
        "artifact_checks": checks,
        "qualification_consumed": False,
        "final_consumed": False,
    }
    _atomic_json(Path(document["outputs"]["preflight"]), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
