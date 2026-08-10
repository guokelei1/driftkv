from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.data.qk_stream_chain import load_corpus
from hstu_kvcache.streaming.qk_sensitivity_runner import PROTOCOL
from hstu_kvcache.streaming.qk_stream_version import (
    file_sha256,
    prequential_evaluation_role_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    corpus = load_corpus(config["data"]["corpus"])
    source_manifest = (
        Path(config["source_checkpoint"]["root"])
        / "theta_0"
        / "manifest.json"
    )
    current_manifest = (
        Path(config["current_checkpoint"]["root"])
        / "theta_1"
        / "manifest.json"
    )
    audit = prequential_evaluation_role_audit(corpus, 1)
    quality = config["quality"]
    if (
        config.get("protocol") != PROTOCOL
        or corpus.file_sha256 != config["data"]["corpus_sha256"]
        or file_sha256(source_manifest)
        != config["source_checkpoint"]["manifest_sha256"]
        or file_sha256(current_manifest)
        != config["current_checkpoint"]["manifest_sha256"]
        or audit["primary_role_users"] != quality["primary_users"]
        or audit["optimizer_participant_users"]
        != quality["optimizer_participant_users"]
        or audit["supplemental_users"] != quality["supplemental_users"]
        or audit["user_overlap"] != 0
        or audit["training_window"] != 1
        or audit["evaluation_window"] != 2
    ):
        raise ValueError("QK sensitivity input binding differs")
    report: dict[str, object] = {
        "status": "pass",
        "protocol": PROTOCOL,
        "config_sha256": file_sha256(args.config),
        "corpus_sha256": corpus.file_sha256,
        "source_manifest_sha256": file_sha256(source_manifest),
        "current_manifest_sha256": file_sha256(current_manifest),
        "role_audit": audit,
    }
    if not args.inputs_only:
        result_path = Path(config["outputs"]["result"])
        result = json.loads(result_path.read_text())
        primary = result.get("quality", {}).get("primary_update_local", {})
        supplemental = result.get("quality", {}).get(
            "supplemental_disjoint_user", {}
        )
        participant = primary.get("cohorts", {}).get(
            "optimizer_participants", {}
        )
        if (
            result.get("protocol") != PROTOCOL
            or result.get("status") != "complete_development_measurement"
            or result.get("scientific_result") is not False
            or result.get("formal_result") is not False
            or result.get("config", {}).get("sha256")
            != file_sha256(args.config)
            or result.get("role_audit") != audit
            or primary.get("records") != quality["primary_users"]
            or supplemental.get("records") != quality["supplemental_users"]
            or participant.get("full_catalog_next_item", {}).get("records")
            != quality["optimizer_participant_users"]
            or result.get("quality", {}).get("qualification_consumed")
            is not False
            or result.get("quality", {}).get("final_consumed") is not False
        ):
            raise ValueError("QK sensitivity result differs")
        report["result"] = {
            "path": str(result_path),
            "sha256": file_sha256(result_path),
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
