from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .ml1m_candidate_robustness import _strict_ranking_gate
from .ml1m_opportunity import (
    _atomic_json,
    _candidate_ids,
    _evaluate,
    _score_vectors,
    file_sha256,
    load_causal_records,
    load_config,
    make_model,
    summarize_evaluation,
)

PROTOCOL = "evokv_ml1m_frozen_hard_candidate_v0"


def load_hard_candidate_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    parent = document.get("parent")
    evaluation = document.get("evaluation")
    expected_strategies = [
        {"candidate_count": 20, "id": "benchmark_hard_20", "source": "benchmark"},
        {"candidate_count": 50, "id": "random_unseen_50", "source": "random_unseen"},
        {"candidate_count": 50, "id": "popular_unseen_50", "source": "train_popularity"},
        {"candidate_count": 50, "id": "old_model_hard_50", "source": "old_model_hard"},
        {"candidate_count": 100, "id": "random_unseen_100", "source": "random_unseen"},
        {"candidate_count": 100, "id": "popular_unseen_100", "source": "train_popularity"},
        {"candidate_count": 100, "id": "old_model_hard_100", "source": "old_model_hard"},
    ]
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or not isinstance(parent, dict)
        or not isinstance(evaluation, dict)
        or evaluation.get("strategies") != expected_strategies
        or evaluation.get("candidate_seed") != 917341
    ):
        raise ValueError("ML1m hard-candidate config differs")
    for path_key, hash_key in (
        ("base_config", "base_config_sha256"),
        ("replication_config", "replication_config_sha256"),
        ("replication_summary", "replication_summary_sha256"),
    ):
        if file_sha256(parent[path_key]) != parent[hash_key]:
            raise ValueError("ML1m hard-candidate parent binding differs")
    return document


def _base_candidates(record: dict[str, Any]) -> list[int]:
    positive = int(record["positive_items"][0])
    values = [positive]
    values.extend(int(value) for value in record["candidates"] if int(value) != positive)
    if len(values) != 20 or len(set(values)) != 20:
        raise ValueError("ML1m benchmark candidate set differs")
    return values


def _candidate_map_hash(candidate_map: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for user_id in sorted(candidate_map):
        digest.update(user_id.encode())
        digest.update(b"\0")
        digest.update(candidate_map[user_id].astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def _truncate_map(candidate_map: dict[str, np.ndarray], count: int):
    output = {user_id: values[:count].copy() for user_id, values in candidate_map.items()}
    if any(len(values) != count or len(np.unique(values)) != count for values in output.values()):
        raise ValueError("ML1m hard candidate truncation differs")
    return output


def _random_map(
    records: dict[str, Any],
    selected: list[str],
    num_items: int,
    count: int,
    seed: int,
):
    return {
        user_id: _candidate_ids(
            records[user_id],
            user_id,
            num_items,
            count,
            seed,
            filter_seen=True,
        )
        for user_id in selected
    }


def _popularity_order(train_records: dict[str, Any], num_items: int) -> np.ndarray:
    counts = np.zeros(num_items + 1, dtype=np.int64)
    for record in train_records.values():
        counts += np.bincount(record["history"], minlength=num_items + 1)
        counts[int(record["positive_items"][0])] += 1
    item_ids = np.arange(1, num_items + 1, dtype=np.int64)
    return item_ids[np.lexsort((item_ids, -counts[1:]))]


def _popular_map(
    train_records: dict[str, Any],
    test_records: dict[str, Any],
    selected: list[str],
    num_items: int,
    count: int,
):
    order = _popularity_order(train_records, num_items)
    output = {}
    for user_id in selected:
        record = test_records[user_id]
        values = _base_candidates(record)
        excluded = set(values)
        excluded.update(int(value) for value in record["history"])
        for item_id in order:
            value = int(item_id)
            if value not in excluded:
                values.append(value)
                excluded.add(value)
                if len(values) == count:
                    break
        output[user_id] = np.asarray(values, dtype=np.int64)
    return output


@torch.no_grad()
def _old_model_hard_map(
    model,
    records: dict[str, Any],
    selected: list[str],
    document: dict[str, Any],
    count: int,
    normalized: bool,
):
    max_seq_len = int(document["model"]["max_seq_len"])
    num_items = int(document["model"]["num_items"])
    groups: dict[int, list[str]] = defaultdict(list)
    for user_id in selected:
        groups[min(len(records[user_id]["history"]), max_seq_len)].append(user_id)
    device = next(model.parameters()).device
    catalog = torch.arange(1, num_items + 1, dtype=torch.long, device=device)
    batch_size = int(document["evaluation"]["batch_size"])
    output = {}
    model.eval()
    for _, user_ids in sorted(groups.items()):
        for start in range(0, len(user_ids), batch_size):
            batch_users = user_ids[start : start + batch_size]
            histories = np.stack(
                [records[user_id]["history"][-max_seq_len:] for user_id in batch_users]
            )
            items = torch.as_tensor(histories, dtype=torch.long, device=device)
            behaviors = torch.ones_like(items)
            deltas = torch.zeros_like(items, dtype=torch.float32)
            hidden, _ = model(items, behaviors, deltas)
            candidates = catalog.unsqueeze(0).expand(len(batch_users), -1)
            scores = _score_vectors(
                model,
                hidden[:, -1],
                candidates,
                normalized,
                float(document["training"]["temperature"] if normalized else 1.0),
            )
            for row, user_id in enumerate(batch_users):
                record = records[user_id]
                values = _base_candidates(record)
                excluded = set(values)
                excluded.update(int(value) for value in record["history"])
                excluded_tensor = torch.tensor(
                    sorted(excluded),
                    dtype=torch.long,
                    device=device,
                )
                scores[row, excluded_tensor - 1] = -torch.inf
                needed = count - len(values)
                mined = torch.topk(scores[row], k=needed, largest=True, sorted=True).indices + 1
                values.extend(int(value) for value in mined.cpu().tolist())
                output[user_id] = np.asarray(values, dtype=np.int64)
    if set(output) != set(selected):
        raise ValueError("ML1m old-model hard candidate coverage differs")
    return output


def _load_seed_models(
    seed_result: dict[str, Any],
    variant: dict[str, Any],
    base_document: dict[str, Any],
    device: torch.device,
):
    models = []
    for version in ("theta0", "theta1"):
        binding = seed_result["checkpoints"][version]
        path = Path(binding["path"])
        if file_sha256(path) != binding["sha256"]:
            raise ValueError("ML1m hard-candidate checkpoint differs")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = make_model(base_document, variant["architecture"], device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models.append(model)
    return models[0], models[1]


def run_hard_candidate(config_path: str | Path) -> dict[str, Any]:
    document = load_hard_candidate_config(config_path)
    output_root = Path(document["outputs"]["root"])
    summary_path = output_root / "summary.json"
    if summary_path.is_file():
        result = json.loads(summary_path.read_text())
        validate_hard_candidate_summary(result, document)
        return result
    output_root.mkdir(parents=True, exist_ok=True)
    base_document = load_config(document["parent"]["base_config"])
    replication_config = json.loads(Path(document["parent"]["replication_config"]).read_text())
    base_document["data"]["user_limit"] = int(replication_config["replication"]["user_limit"])
    replication = json.loads(Path(document["parent"]["replication_summary"]).read_text())
    causal = load_causal_records(base_document)
    selected = causal["selected_users"]
    num_items = int(base_document["model"]["num_items"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for ML1m hard-candidate evaluation")
    replication_by_id = {value["id"]: value for value in replication["variants"]}
    variant_by_id = {value["id"]: value for value in replication_config["variants"]}
    variants = []
    started = time.monotonic()
    for variant_id, replication_variant in replication_by_id.items():
        variant = variant_by_id[variant_id]
        seed_outputs = []
        for seed_result in replication_variant["seed_results"]:
            seed = int(seed_result["seed"])
            print(
                f"phase=ml1m_hard_candidate_start variant={variant_id} seed={seed}",
                flush=True,
            )
            previous, current = _load_seed_models(
                seed_result,
                variant,
                base_document,
                device,
            )
            benchmark_map = {
                user_id: np.asarray(_base_candidates(causal["splits"]["test"][user_id]), dtype=np.int64)
                for user_id in selected
            }
            random_100 = _random_map(
                causal["splits"]["test"],
                selected,
                num_items,
                100,
                int(document["evaluation"]["candidate_seed"]),
            )
            popular_100 = _popular_map(
                causal["splits"]["train"],
                causal["splits"]["test"],
                selected,
                num_items,
                100,
            )
            model_hard_100 = _old_model_hard_map(
                previous,
                causal["splits"]["test"],
                selected,
                base_document,
                100,
                bool(variant["normalized_scoring"]),
            )
            maps = {
                "benchmark_hard_20": benchmark_map,
                "random_unseen_50": _truncate_map(random_100, 50),
                "popular_unseen_50": _truncate_map(popular_100, 50),
                "old_model_hard_50": _truncate_map(model_hard_100, 50),
                "random_unseen_100": random_100,
                "popular_unseen_100": popular_100,
                "old_model_hard_100": model_hard_100,
            }
            strategy_results = []
            for strategy in document["evaluation"]["strategies"]:
                candidate_map = maps[strategy["id"]]
                evaluation = _evaluate(
                    previous,
                    current,
                    causal["splits"]["test"],
                    selected,
                    base_document,
                    bool(variant["normalized_scoring"]),
                    candidate_map=candidate_map,
                    candidate_protocol_override=strategy["id"],
                )
                compact = summarize_evaluation(evaluation, base_document)
                gate = _strict_ranking_gate(compact)
                result_path = (
                    output_root
                    / "variants"
                    / variant_id
                    / f"seed_{seed}"
                    / f"{strategy['id']}.json"
                )
                _atomic_json(
                    result_path,
                    {
                        "variant": variant,
                        "seed": seed,
                        "strategy": strategy,
                        "candidate_map_sha256": _candidate_map_hash(candidate_map),
                        "summary": compact,
                        "strict_ranking_gate": gate,
                        "records": evaluation["records"],
                    },
                )
                strategy_results.append(
                    {
                        "strategy": strategy,
                        "candidate_map_sha256": _candidate_map_hash(candidate_map),
                        "result_path": str(result_path),
                        "result_sha256": file_sha256(result_path),
                        "summary": compact,
                        "strict_ranking_gate": gate,
                    }
                )
                stale = compact["comparisons"]["recompute_over_reuse"]
                print(
                    f"phase=ml1m_hard_candidate_result variant={variant_id} seed={seed} "
                    f"strategy={strategy['id']} stale_ce={stale['candidate_cross_entropy']['absolute']:.6f} "
                    f"stale_mrr={stale['mrr']['absolute']:.6f} "
                    f"stale_ndcg10={stale['ndcg_at_10']['absolute']:.6f} passed={gate['passed']}",
                    flush=True,
                )
            seed_outputs.append({"seed": seed, "strategy_results": strategy_results})
            del previous, current
            torch.cuda.empty_cache()
        variants.append({"id": variant_id, "seed_results": seed_outputs})
    stable = []
    for variant in variants:
        strategy_ids = [value["id"] for value in document["evaluation"]["strategies"]]
        for strategy_id in strategy_ids:
            if all(
                next(
                    result
                    for result in seed["strategy_results"]
                    if result["strategy"]["id"] == strategy_id
                )["strict_ranking_gate"]["passed"]
                for seed in variant["seed_results"]
            ):
                stable.append({"variant": variant["id"], "strategy": strategy_id})
    summary = {
        "protocol": PROTOCOL,
        "round_id": document["round_id"],
        "status": "complete",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "parent": document["parent"],
        "variants": variants,
        "decision": {
            "stable_positive_cells": stable,
            "positive_candidate_found": bool(stable),
            "next": "holdout_seed_confirmation" if stable else "controlled_update_sweep",
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    validate_hard_candidate_summary(summary, document)
    _atomic_json(summary_path, summary)
    return summary


def validate_hard_candidate_summary(summary: dict[str, Any], document: dict[str, Any]) -> None:
    variants = summary.get("variants")
    if (
        summary.get("protocol") != PROTOCOL
        or summary.get("round_id") != document["round_id"]
        or summary.get("status") != "complete"
        or summary.get("scientific_result") is not False
        or not isinstance(variants, list)
    ):
        raise ValueError("ML1m hard-candidate summary differs")
    for variant in variants:
        for seed in variant.get("seed_results", []):
            results = seed.get("strategy_results")
            if not isinstance(results, list) or [value.get("strategy") for value in results] != document[
                "evaluation"
            ]["strategies"]:
                raise ValueError("ML1m hard-candidate strategy coverage differs")
            for result in results:
                path = Path(result["result_path"])
                if not path.is_file() or file_sha256(path) != result["result_sha256"]:
                    raise ValueError("ML1m hard-candidate result binding differs")
                if not result.get("summary", {}).get("sanity", {}).get("passed"):
                    raise ValueError("ML1m hard-candidate sanity failed")
