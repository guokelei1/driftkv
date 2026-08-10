from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.data.qk_theta0 import file_sha256, load_qk_theta0_corpus
from hstu_kvcache.streaming.xp_projected_edge import XPProjectedModelSpec

PROTOCOL = "evokv_qk_theta0_next_item_training_v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--corpus-only", action="store_true")
    return parser.parse_args()


def _spec(document: dict[str, object]) -> XPProjectedModelSpec:
    model = document["model"]
    return XPProjectedModelSpec(
        num_embeddings=int(model["num_embeddings"]),
        embedding_width=int(model["embedding_width"]),
        hidden_size=int(model["hidden_size"]),
        num_prediction_items=int(model["num_prediction_items"]),
        num_behaviors=int(model["num_behaviors"]),
        num_layers=int(model["num_layers"]),
        num_heads=int(model["num_heads"]),
        head_dim=int(model["head_dim"]),
        max_seq_len=int(model["max_seq_len"]),
    )


def _verify_descriptor(directory: Path, value: object) -> Path:
    if not isinstance(value, dict):
        raise ValueError("QK theta0 artifact descriptor is absent")
    path = directory / str(value.get("path", ""))
    if (
        not path.is_file()
        or path.stat().st_size != int(value.get("bytes", -1))
        or file_sha256(path) != value.get("sha256")
    ):
        raise ValueError(f"QK theta0 artifact differs: {path}")
    return path


def validate(
    config: Path,
    corpus_only: bool,
    expected_protocol: str = PROTOCOL,
) -> dict[str, object]:
    document = json.loads(config.read_text())
    if document.get("protocol") != expected_protocol:
        raise ValueError("QK theta0 config protocol differs")
    spec = _spec(document)
    corpus = load_qk_theta0_corpus(
        document["data"]["corpus"],
        num_embeddings=spec.num_embeddings,
        num_prediction_items=spec.num_prediction_items,
    )
    report: dict[str, object] = {
        "protocol": expected_protocol,
        "status": "pass",
        "config": {
            "path": str(config),
            "sha256": file_sha256(config),
        },
        "corpus": {
            "path": str(corpus.path),
            "file_sha256": corpus.file_sha256,
            "content_sha256": corpus.content_sha256,
            "records": corpus.records,
            "tokens": corpus.tokens,
        },
    }
    summary_path = Path(document["data"]["corpus_summary"])
    summary = json.loads(summary_path.read_text())
    artifact = summary.get("artifact")
    if (
        summary.get("status") != "pass"
        or not isinstance(artifact, dict)
        or artifact.get("file_sha256") != corpus.file_sha256
        or artifact.get("content_sha256") != corpus.content_sha256
    ):
        raise ValueError("QK theta0 corpus summary differs")
    report["corpus_summary"] = {
        "path": str(summary_path),
        "sha256": file_sha256(summary_path),
    }
    if corpus_only:
        return report
    result_path = Path(document["outputs"]["result"])
    result = json.loads(result_path.read_text())
    commit = bool(document["execution"]["commit_checkpoint"])
    expected_status = "complete" if commit else "canary_complete"
    if (
        result.get("protocol") != expected_protocol
        or result.get("status") != expected_status
        or result.get("objective") != "sampled next-item cross entropy"
        or result.get("config", {}).get("sha256") != file_sha256(config)
        or result.get("corpus", {}).get("file_sha256") != corpus.file_sha256
        or result.get("gates", {}).get("dense_probe_changed") is not True
        or result.get("gates", {}).get("projection_probe_changed") is not True
    ):
        raise ValueError("QK theta0 result semantics differ")
    if commit:
        minimum = int(document["data"]["minimum_optimizer_active_rows"])
        observed = int(result["gates"]["observed_optimizer_active_rows"])
        admission = str(
            document["execution"].get(
                "optimizer_active_admission", "required"
            )
        )
        threshold_passed = observed >= minimum
        if (
            admission not in ("required", "report_only")
            or (admission == "required" and not threshold_passed)
            or result["gates"].get("optimizer_active_gate_passed")
            is not threshold_passed
            or result["gates"].get("optimizer_active_admission")
            != admission
            or result["gates"].get("checkpoint_admission_passed")
            is not True
            or result.get("checkpoint", {}).get("committed") is not True
        ):
            raise ValueError("QK theta0 optimizer-active gate differs")
        directory = Path(document["outputs"]["checkpoint_root"]) / "theta_0"
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("version") != 0
            or manifest.get("world_size") != 2
            or manifest.get("spec") != result["model"]["spec"]
            or int(manifest["optimizer_active_rows"]["global_active_rows"])
            != observed
            or manifest.get("provenance", {}).get("protocol")
            != expected_protocol
        ):
            raise ValueError("QK theta0 checkpoint manifest differs")
        _verify_descriptor(directory, manifest.get("dense"))
        _verify_descriptor(directory, manifest.get("projection"))
        for value in manifest.get("embedding_shards", []):
            _verify_descriptor(directory, value)
        for value in manifest.get("optimizer_active_rows", {}).get(
            "bitmap_shards", []
        ):
            _verify_descriptor(directory, value)
        state_path = directory / "training_state.json"
        state = json.loads(state_path.read_text())
        if (
            state.get("complete") is not True
            or state.get("config_sha256") != file_sha256(config)
            or state.get("corpus_file_sha256") != corpus.file_sha256
            or int(state.get("optimizer_active_rows", -1)) != observed
        ):
            raise ValueError("QK theta0 training state differs")
        _verify_descriptor(directory, state.get("optimizer_resume"))
        report["checkpoint"] = {
            "path": str(directory),
            "manifest_sha256": file_sha256(manifest_path),
            "optimizer_active_rows": observed,
        }
    report["result"] = {
        "path": str(result_path),
        "sha256": file_sha256(result_path),
        "status": result["status"],
    }
    return report


def main() -> None:
    args = parse_args()
    print(json.dumps(validate(args.config, args.corpus_only), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
