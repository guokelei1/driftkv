from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.data.qk_stream_chain import load_corpus
from hstu_kvcache.streaming.qk_alignment_runner import (
    COHORTS,
    MODES,
    PROTOCOL,
    _validate_document,
)
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
    _validate_document(config)
    corpus = load_corpus(config["data"]["corpus"])
    edge = config["edge"]
    source_version = int(edge["source_version"])
    target_version = int(edge["target_version"])
    edge_index = int(edge["edge"])
    source_manifest = (
        Path(config["source_checkpoint"]["root"])
        / f"theta_{source_version}"
        / "manifest.json"
    )
    current_manifest = (
        Path(config["current_checkpoint"]["root"])
        / f"theta_{target_version}"
        / "manifest.json"
    )
    audit = prequential_evaluation_role_audit(corpus, edge_index)
    quality = config["quality"]
    if (
        corpus.file_sha256 != config["data"]["corpus_sha256"]
        or file_sha256(source_manifest)
        != config["source_checkpoint"]["manifest_sha256"]
        or file_sha256(current_manifest)
        != config["current_checkpoint"]["manifest_sha256"]
        or audit["primary_role_users"] != quality["primary_users"]
        or audit["optimizer_participant_users"]
        != quality["optimizer_participant_users"]
        or audit["training_window"] != edge_index
        or audit["evaluation_window"] != edge_index + 1
    ):
        raise ValueError("QK alignment input binding differs")
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
        evaluation = result.get("quality", {}).get("evaluation", {})
        summary = evaluation.get("summary", {})
        matrix_valid = set(summary) == set(MODES)
        if matrix_valid:
            matrix_valid = all(
                set(summary[mode].get("cohorts", {})) == set(COHORTS)
                for mode in MODES
            )
        gate = evaluation.get("alignment_gate", {})
        if (
            result.get("protocol") != PROTOCOL
            or result.get("status") != "complete_development_measurement"
            or result.get("scientific_result") is not False
            or result.get("formal_result") is not False
            or result.get("config", {}).get("sha256") != file_sha256(args.config)
            or result.get("role_audit") != audit
            or evaluation.get("records") != quality["optimizer_participant_users"]
            or not matrix_valid
            or gate.get("status")
            not in {"aligned_protocol_found", "no_aligned_protocol"}
            or result.get("quality", {}).get(
                "evaluation_window_labels_used_for_cohort_definition"
            )
            is not False
            or result.get("quality", {}).get("qualification_consumed") is not False
            or result.get("quality", {}).get("final_consumed") is not False
        ):
            raise ValueError("QK alignment result differs")
        report["result"] = {
            "path": str(result_path),
            "sha256": file_sha256(result_path),
            "alignment_gate": gate["status"],
            "admitted": gate["admitted"],
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
