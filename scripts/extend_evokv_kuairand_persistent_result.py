from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.kuairand_projected_persistent import PROTOCOL
from hstu_kvcache.streaming.kuairand_query_transition import (
    _atomic_json,
    file_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-result", required=True)
    parser.add_argument("--accepted", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base_path = Path(args.base_result)
    accepted_path = Path(args.accepted)
    config_path = Path(args.config)
    output_path = Path(args.output)
    base = json.loads(base_path.read_text())
    accepted = json.loads(accepted_path.read_text())
    config = json.loads(config_path.read_text())
    version = int(accepted.get("version", -1))
    checkpoints = list(base.get("checkpoints", []))
    targets = list(base.get("targets", []))
    if (
        base.get("status") not in ("complete", "complete_extension_manifest")
        or base.get("protocol") != PROTOCOL
        or [int(value["version"]) for value in checkpoints] != list(range(1, version))
        or len(targets) != version - 1
        or accepted.get("status") != "accepted"
        or accepted.get("source_version") != version - 1
        or config.get("checkpoint", {}).get("versions") < version
        or config.get("transitions", [])[version - 1].get("target_version") != version
    ):
        raise ValueError("KuaiRand persistent extension inputs differ")
    manifest_path = Path(accepted["checkpoint"]["path"])
    if (
        not manifest_path.is_file()
        or file_sha256(manifest_path) != accepted["checkpoint"]["sha256"]
    ):
        raise ValueError("KuaiRand persistent extension checkpoint differs")
    targets.append(
        {
            "target_version": version,
            "source_version": version - 1,
            "transition": config["transitions"][version - 1],
            "lineage": [
                {
                    "source_version": version - 1,
                    "cache_age": 1,
                    "summary": accepted["candidate"]["summary"],
                }
            ],
        }
    )
    checkpoints.append(
        {
            "version": version,
            "path": str(manifest_path),
            "sha256": accepted["checkpoint"]["sha256"],
            "bytes": int(accepted["checkpoint"]["bytes"]),
        }
    )
    result = {
        **base,
        "status": "complete_extension_manifest",
        "scientific_result": False,
        "formal_result": False,
        "targets": targets,
        "checkpoints": checkpoints,
        "extension": {
            "base_result": {"path": str(base_path), "sha256": file_sha256(base_path)},
            "accepted": {
                "path": str(accepted_path),
                "sha256": file_sha256(accepted_path),
            },
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "version": version,
        },
    }
    _atomic_json(output_path, result)
    print(json.dumps(result["extension"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
