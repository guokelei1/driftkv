from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.kuairand_path_attribution import (
    PROTOCOL,
    validate_attribution_document,
)
from hstu_kvcache.streaming.qk_stream_version import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = json.loads(args.config.read_text())
    validate_attribution_document(document)
    path = Path(document["outputs"]["result"])
    result = json.loads(path.read_text())
    if (
        result.get("protocol") != PROTOCOL
        or result.get("status") != "complete_development_attribution"
        or result.get("scope") != "development_attribution"
        or result.get("scientific_result") is not False
        or result.get("formal_result") is not False
        or result.get("config", {}).get("sha256") != file_sha256(args.config)
        or [value.get("edge") for value in result.get("edges", [])] != [1, 2]
        or not result.get("sanity", {}).get("implementation_passed")
        or result.get("execution", {}).get("qualification_consumed") is not False
        or result.get("execution", {}).get("final_consumed") is not False
    ):
        raise ValueError("KuaiRand path-attribution result differs")
    print(
        json.dumps(
            {"status": "valid", "path": str(path), "sha256": file_sha256(path)},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
