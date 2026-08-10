from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.data.qk_stream_chain import load_corpus
from hstu_kvcache.streaming.qk_stream_version import PROTOCOL, file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--corpus-only", action="store_true")
    return parser.parse_args()


def _descriptor(directory: Path, value: object) -> Path:
    if not isinstance(value, dict):
        raise ValueError("QK stream checkpoint descriptor is absent")
    path = directory / str(value.get("path", ""))
    if (
        not path.is_file()
        or path.stat().st_size != int(value.get("bytes", -1))
        or file_sha256(path) != value.get("sha256")
    ):
        raise ValueError(f"QK stream checkpoint artifact differs: {path}")
    return path


def validate(config_path: Path, corpus_only: bool) -> dict[str, object]:
    document = json.loads(config_path.read_text())
    if document.get("protocol") != PROTOCOL:
        raise ValueError("QK stream edge config protocol differs")
    data_config = Path(document["data"]["config"])
    if file_sha256(data_config) != document["data"]["config_sha256"]:
        raise ValueError("QK stream data config hash differs")
    corpus = load_corpus(document["data"]["corpus"])
    summary_path = Path(document["data"]["summary"])
    roles_path = Path(document["data"]["roles"])
    summary = json.loads(summary_path.read_text())
    roles = json.loads(roles_path.read_text())
    if (
        summary.get("status") != "pass"
        or summary.get("artifact", {}).get("sha256") != corpus.file_sha256
        or summary.get("content_sha256") != corpus.content_sha256
        or summary.get("roles", {}).get("sha256") != file_sha256(roles_path)
        or roles.get("roles_pairwise_disjoint") is not True
        or roles.get("post_base_selection_uses_labels") is not False
    ):
        raise ValueError("QK stream data artifacts differ")
    report: dict[str, object] = {
        "protocol": PROTOCOL,
        "status": "pass",
        "config_sha256": file_sha256(config_path),
        "corpus": {
            "path": str(corpus.path),
            "sha256": corpus.file_sha256,
            "content_sha256": corpus.content_sha256,
            "records": len(corpus.arrays["record_user_ids"]),
            "rows": len(corpus.arrays["item_idx"]),
        },
        "roles_sha256": file_sha256(roles_path),
        "summary_sha256": file_sha256(summary_path),
    }
    if corpus_only:
        return report
    result_path = Path(document["outputs"]["result"])
    result = json.loads(result_path.read_text())
    allowed = {
        "complete_development_qualified",
        "complete_tuning_gate_failed",
        "complete_qualification_gate_failed",
    }
    if (
        result.get("protocol") != PROTOCOL
        or result.get("status") not in allowed
        or result.get("config", {}).get("sha256")
        != file_sha256(config_path)
        or result.get("data", {}).get("corpus_sha256")
        != corpus.file_sha256
        or result.get("training", {}).get("checkpoint_admission_passed")
        is not True
        or result.get("quality", {}).get("final_consumed") is not False
        or result.get("quality", {}).get("labels_used_for_routing") is not False
    ):
        raise ValueError("QK stream edge result differs")
    tuning = result["quality"]["tuning"]
    qualification = result["quality"]["qualification"]
    committed = result["checkpoint"]["committed"]
    if result["status"] == "complete_tuning_gate_failed":
        if tuning["practical_gate_passed"] or qualification is not None or committed:
            raise ValueError("QK stream tuning failure semantics differ")
    elif result["status"] == "complete_qualification_gate_failed":
        if (
            not tuning["practical_gate_passed"]
            or qualification is None
            or qualification["practical_gate_passed"]
            or committed
        ):
            raise ValueError("QK stream qualification failure semantics differ")
    else:
        if (
            not tuning["practical_gate_passed"]
            or qualification is None
            or not qualification["practical_gate_passed"]
            or committed is not True
        ):
            raise ValueError("QK stream qualification pass semantics differ")
        version = int(document["edge"]["target_version"])
        directory = Path(document["outputs"]["checkpoint_root"]) / f"theta_{version}"
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if (
            file_sha256(manifest_path)
            != result["checkpoint"]["manifest_sha256"]
            or manifest.get("version") != version
            or manifest.get("world_size") != 2
            or manifest.get("provenance", {}).get("config_sha256")
            != file_sha256(config_path)
            or manifest.get("provenance", {}).get("corpus", {}).get("sha256")
            != corpus.file_sha256
        ):
            raise ValueError("QK stream committed manifest differs")
        _descriptor(directory, manifest.get("dense"))
        _descriptor(directory, manifest.get("projection"))
        for value in manifest.get("embedding_shards", []):
            _descriptor(directory, value)
        state = json.loads((directory / "training_state.json").read_text())
        if (
            state.get("complete") is not True
            or state.get("config_sha256") != file_sha256(config_path)
            or state.get("corpus_file_sha256") != corpus.file_sha256
        ):
            raise ValueError("QK stream committed training state differs")
        _descriptor(directory, state.get("optimizer_resume"))
        report["checkpoint"] = {
            "path": str(directory),
            "manifest_sha256": file_sha256(manifest_path),
        }
    report["result"] = {
        "path": str(result_path),
        "sha256": file_sha256(result_path),
        "status": result["status"],
    }
    return report


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            validate(args.config, args.corpus_only),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
