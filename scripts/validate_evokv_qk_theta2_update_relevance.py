from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.data.qk_stream_chain import load_corpus
from hstu_kvcache.streaming.qk_stream_version import file_sha256
from hstu_kvcache.streaming.qk_update_relevance_runner import (
    COHORTS,
    MODES,
    PROTOCOL,
    _validate_document,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs-only", action="store_true")
    return parser.parse_args()


def validate(config_path: Path, *, inputs_only: bool) -> dict[str, object]:
    document = json.loads(config_path.read_text())
    _validate_document(document)
    corpus = load_corpus(document["data"]["corpus"])
    source = document["source_checkpoint"]
    current = document["current_checkpoint"]
    source_manifest = Path(source["root"]) / "theta_1" / "manifest.json"
    current_manifest = Path(current["root"]) / "theta_2" / "manifest.json"
    if (
        corpus.file_sha256 != document["data"]["corpus_sha256"]
        or file_sha256(source_manifest) != source["manifest_sha256"]
        or file_sha256(current_manifest) != current["manifest_sha256"]
    ):
        raise ValueError("QK update relevance frozen inputs differ")
    output = Path(document["outputs"]["result"])
    if inputs_only:
        return {
            "status": "pass",
            "scope": "inputs_only",
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "candidate": document["candidate"],
        }
    result = json.loads(output.read_text())
    evaluation = result.get("quality", {}).get("evaluation", {})
    summary = evaluation.get("summary", {})
    full = summary.get("full_catalog", {})
    candidate_protocols = summary.get("candidate_protocols", {})
    limit = int(document["execution"]["record_limit_per_rank"])
    if (
        result.get("protocol") != PROTOCOL
        or result.get("status") != "complete_development_measurement"
        or result.get("scientific_result") is not False
        or result.get("formal_result") is not False
        or result.get("config", {}).get("sha256") != file_sha256(config_path)
        or result.get("candidate") != document["candidate"]
        or result.get("source_checkpoint") != source
        or result.get("current_checkpoint") != current
        or result.get("training_relations", {}).get("evaluation_window_events_used") is not False
        or result.get("quality", {}).get("evaluation_targets_used_for_training") is not False
        or result.get("quality", {}).get("evaluation_targets_used_for_relation_construction") is not False
        or set(full) != set(MODES)
        or set(candidate_protocols) != set(MODES)
        or not isinstance(evaluation.get("primary_admission_gate"), dict)
        or (limit == 0 and int(evaluation.get("records", 0)) != 14068)
    ):
        raise ValueError("QK update relevance result differs")
    for mode in MODES:
        cohorts = full[mode].get("cohorts", {})
        candidate_cohorts = candidate_protocols[mode].get("cohorts", {})
        if set(cohorts) != set(COHORTS) or not candidate_cohorts:
            raise ValueError("QK update relevance result cohorts differ")
    if limit == 0:
        rolling = full["rolling_next_item"]["cohorts"]
        expected = {
            "all": 77479,
            "context_h16_support_ge1": 12740,
            "context_h32_support_ge1": 16741,
            "context_h32_support_ge2": 8040,
            "copositive_support_ge1": 13179,
        }
        observed = {
            name: int(rolling[name]["positive_targets"]) for name in expected
        }
        if observed != expected:
            raise ValueError(
                f"QK update relevance target ledger differs: {observed}"
            )
    return {
        "status": "pass",
        "scope": "complete_result",
        "candidate": document["candidate"],
        "result": {"path": str(output), "sha256": file_sha256(output)},
        "primary_admission_status": evaluation["primary_admission_gate"]["status"],
    }


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            validate(args.config, inputs_only=args.inputs_only),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
