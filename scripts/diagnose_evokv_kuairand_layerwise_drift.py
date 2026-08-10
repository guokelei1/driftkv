import argparse
import json
import time
from pathlib import Path

import torch

from hstu_kvcache.streaming.kuairand_query_transition import (
    _atomic_json,
    _evaluate,
    _summary,
    build_workload,
    file_sha256,
    load_config,
    make_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = load_config(args.config)
    source_path = Path(document["outputs"]["root"]) / "summary.json"
    source = json.loads(source_path.read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("KuaiRand layerwise drift diagnosis requires CUDA")
    workload = build_workload(document)
    started = time.monotonic()
    seed_results = []
    for seed_record in source["seed_results"]:
        seed = int(seed_record["seed"])
        result = json.loads(Path(seed_record["result_path"]).read_text())
        previous = make_model(
            document, int(workload["metadata"]["embedding_rows"]), device
        )
        current = make_model(
            document, int(workload["metadata"]["embedding_rows"]), device
        )
        theta0_path = Path(result["checkpoints"]["theta0"]["path"])
        theta1_path = Path(result["checkpoints"]["theta1"]["path"])
        theta0 = torch.load(theta0_path, map_location=device, weights_only=True)
        theta1 = torch.load(theta1_path, map_location=device, weights_only=True)
        previous.load_state_dict(theta0["state_dict"])
        current.load_state_dict(theta1["state_dict"])
        previous.eval()
        current.eval()
        evaluation = _evaluate(previous, current, workload, document)
        compact = _summary(evaluation, document)
        seed_results.append(
            {
                "seed": seed,
                "checkpoints": {
                    "theta0": {
                        "path": str(theta0_path),
                        "sha256": file_sha256(theta0_path),
                    },
                    "theta1": {
                        "path": str(theta1_path),
                        "sha256": file_sha256(theta1_path),
                    },
                },
                "summary": compact,
                "records": evaluation["records"],
            }
        )
        del previous, current, theta0, theta1
        torch.cuda.empty_cache()
    output = {
        "protocol": "evokv_kuairand_layerwise_cache_drift_v0",
        "status": "complete",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": args.config, "sha256": file_sha256(args.config)},
        "source": {"path": str(source_path), "sha256": file_sha256(source_path)},
        "seed_results": seed_results,
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(Path(args.output), output)


if __name__ == "__main__":
    main()
