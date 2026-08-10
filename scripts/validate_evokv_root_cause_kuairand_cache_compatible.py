from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.streaming.kuairand_cache_compatible import (
    PROTOCOL,
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
    training_path = Path(document["outputs"]["training_result"])
    evaluation_path = Path(document["outputs"]["evaluation_result"])
    training = json.loads(training_path.read_text())
    result = json.loads(evaluation_path.read_text())
    if (
        training.get("protocol") != PROTOCOL
        or training.get("status") != "complete_development_training"
        or training.get("config", {}).get("sha256") != file_sha256(args.config)
        or [value.get("version") for value in training.get("versions", [])] != [1, 2]
        or result.get("protocol") != PROTOCOL
        or result.get("status") != "complete_development_model_revision"
        or result.get("scope") != "development_model_revision"
        or result.get("scientific_result") is not False
        or result.get("formal_result") is not False
        or result.get("config", {}).get("sha256") != file_sha256(args.config)
        or [value.get("edge") for value in result.get("edges", [])] != [1, 2]
        or not result.get("sanity", {}).get("implementation_passed")
        or result.get("decision", {}).get("classification")
        not in (
            "primary_cache_compatible_update_candidate",
            "hybrid_cache_compatible_update_candidate",
            "pure_output_only_revision_rejected",
        )
        or result.get("execution", {}).get("qualification_consumed") is not False
        or result.get("execution", {}).get("final_consumed") is not False
    ):
        raise ValueError("KuaiRand cache-compatible result differs")
    print(
        json.dumps(
            {
                "status": "valid",
                "training": {
                    "path": str(training_path),
                    "sha256": file_sha256(training_path),
                },
                "evaluation": {
                    "path": str(evaluation_path),
                    "sha256": file_sha256(evaluation_path),
                },
                "classification": result["decision"]["classification"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
