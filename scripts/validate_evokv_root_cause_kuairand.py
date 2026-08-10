from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.kuairand_root_cause import PROTOCOL, validate_document
from hstu_kvcache.streaming.qk_stream_version import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact", choices=("training", "evaluation"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = json.loads(args.config.read_text())
    validate_document(document)
    path = Path(document["outputs"][f"{args.artifact}_result"])
    result = json.loads(path.read_text())
    expected_status = (
        "complete_development_training"
        if args.artifact == "training"
        else "complete_development_measurement"
    )
    if (
        result.get("protocol") != PROTOCOL
        or result.get("status") != expected_status
        or result.get("scope") != document["scope"]
        or result.get("scientific_result") is not False
        or result.get("formal_result") is not False
        or result.get("config", {}).get("sha256") != file_sha256(args.config)
        or result.get("execution", {}).get("qualification_consumed") is not False
        or result.get("execution", {}).get("final_consumed") is not False
    ):
        raise ValueError(f"KuaiRand {args.artifact} artifact differs")
    if args.artifact == "training":
        if [value.get("version") for value in result.get("versions", [])] != [0, 1, 2]:
            raise ValueError("KuaiRand training versions differ")
    else:
        edges = result.get("edges", [])
        if (
            [value.get("edge") for value in edges] != [1, 2]
            or not result.get("sanity", {}).get("implementation_passed")
            or any(value.get("positive_targets", 0) < 1 for value in edges)
        ):
            raise ValueError("KuaiRand evaluation validation failed")
    print(
        json.dumps(
            {
                "status": "valid",
                "artifact": args.artifact,
                "path": str(path),
                "sha256": file_sha256(path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
