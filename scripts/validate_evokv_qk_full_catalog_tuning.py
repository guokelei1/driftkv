from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from hstu_kvcache.data.qk_stream_chain import load_corpus
from hstu_kvcache.streaming.qk_stream_version import (
    FULL_CATALOG_PROTOCOL,
    file_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--corpus-only", action="store_true")
    return parser.parse_args()


def _descriptor(directory: Path, value: object) -> Path:
    if not isinstance(value, dict):
        raise ValueError("QK full-catalog checkpoint descriptor is absent")
    path = directory / str(value.get("path", ""))
    if (
        not path.is_file()
        or path.stat().st_size != int(value.get("bytes", -1))
        or file_sha256(path) != value.get("sha256")
    ):
        raise ValueError(
            f"QK full-catalog checkpoint artifact differs: {path}"
        )
    return path


def validate(config_path: Path, corpus_only: bool) -> dict[str, object]:
    document = json.loads(config_path.read_text())
    if document.get("protocol") != FULL_CATALOG_PROTOCOL:
        raise ValueError("QK full-catalog config protocol differs")
    data_config = Path(document["data"]["config"])
    if file_sha256(data_config) != document["data"]["config_sha256"]:
        raise ValueError("QK full-catalog data config hash differs")
    corpus = load_corpus(document["data"]["corpus"])
    summary_path = Path(document["data"]["summary"])
    roles_path = Path(document["data"]["roles"])
    summary = json.loads(summary_path.read_text())
    roles = json.loads(roles_path.read_text())
    if (
        summary.get("status") != "pass"
        or summary.get("artifact", {}).get("sha256") != corpus.file_sha256
        or summary.get("content_sha256") != corpus.content_sha256
        or summary.get("roles", {}).get("sha256")
        != file_sha256(roles_path)
        or roles.get("roles_pairwise_disjoint") is not True
        or roles.get("post_base_selection_uses_labels") is not False
    ):
        raise ValueError("QK full-catalog data artifacts differ")
    report: dict[str, object] = {
        "protocol": FULL_CATALOG_PROTOCOL,
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
    if (
        result.get("protocol") != FULL_CATALOG_PROTOCOL
        or result.get("status") != "complete_tuning_measurement"
        or result.get("config", {}).get("sha256")
        != file_sha256(config_path)
        or result.get("data", {}).get("corpus_sha256")
        != corpus.file_sha256
        or result.get("training", {}).get("checkpoint_admission_passed")
        is not True
        or result.get("training", {}).get("evaluation_ready") is not True
        or result.get("quality", {}).get("methods")
        != ["reuse", "recompute"]
        or result.get("quality", {}).get("qualification") is not None
        or result.get("quality", {}).get("qualification_consumed") is not False
        or result.get("quality", {}).get("final_consumed") is not False
        or result.get("quality", {}).get("labels_used_for_routing") is not False
        or result.get("quality", {}).get("decision_boundary")
        != "manual_after_tuning"
        or result.get("checkpoint", {}).get("committed") is not False
        or result.get("checkpoint", {}).get("provisional_retained") is not True
    ):
        raise ValueError("QK full-catalog tuning result differs")
    tuning = result["quality"]["tuning"]
    metric_names = (
        "cross_entropy",
        "perplexity",
        "ndcg_at_10",
        "mrr",
        "hit_rate_at_10",
        "hit_rate_at_50",
        "hit_rate_at_200",
    )
    if (
        tuning.get("protocol")
        != "evokv_qk_full_catalog_reuse_recompute_metrics_v1"
        or tuning.get("role") != "fit_tuning"
        or tuning.get("candidate_set")
        != "all prediction item ids [1, num_prediction_items]"
        or tuning.get("num_prediction_items")
        != document["quality"]["num_prediction_items"]
        or tuning.get("decision") != "manual_after_tuning"
        or tuning.get("positive_targets", 0) < 1
    ):
        raise ValueError("QK full-catalog tuning summary differs")
    for method in ("reuse", "recompute"):
        values = tuning.get(method)
        if not isinstance(values, dict) or any(
            metric not in values
            or not math.isfinite(float(values[metric]))
            for metric in metric_names
        ):
            raise ValueError("QK full-catalog endpoint metric differs")
    for metric in metric_names:
        if metric == "perplexity":
            continue
        gap = tuning.get("gaps", {}).get(metric)
        interval = (
            gap.get("record_cluster_bootstrap_95")
            if isinstance(gap, dict)
            else None
        )
        if (
            not isinstance(interval, dict)
            or not math.isfinite(float(gap.get("absolute", math.nan)))
            or not math.isfinite(float(interval.get("lower", math.nan)))
            or not math.isfinite(float(interval.get("upper", math.nan)))
        ):
            raise ValueError("QK full-catalog paired gap differs")
    version = int(document["edge"]["target_version"])
    directory = (
        Path(document["outputs"]["work_checkpoint_root"])
        / f"theta_{version}"
    )
    if str(directory) != result["checkpoint"]["path"]:
        raise ValueError("QK full-catalog provisional checkpoint path differs")
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
        raise ValueError("QK full-catalog provisional manifest differs")
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
        raise ValueError("QK full-catalog training state differs")
    _descriptor(directory, state.get("optimizer_resume"))
    for key in (
        "metric_table_json",
        "metric_table_csv",
        "metric_table_markdown",
    ):
        path = Path(document["outputs"][key])
        if not path.is_file() or path.stat().st_size < 1:
            raise ValueError(f"QK full-catalog metric table is absent: {path}")
    report["checkpoint"] = {
        "path": str(directory),
        "manifest_sha256": file_sha256(manifest_path),
        "provisional": True,
    }
    report["result"] = {
        "path": str(result_path),
        "sha256": file_sha256(result_path),
        "status": result["status"],
    }
    report["metric_tables"] = {
        key: {
            "path": document["outputs"][key],
            "sha256": file_sha256(document["outputs"][key]),
        }
        for key in (
            "metric_table_json",
            "metric_table_csv",
            "metric_table_markdown",
        )
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
