from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .ml1m_candidate_robustness import _strict_ranking_gate
from .ml1m_hard_candidate import _base_candidates, _popular_map
from .ml1m_opportunity import (
    _atomic_json,
    _candidate_ids,
    _evaluate,
    _sample_candidates,
    _save_checkpoint,
    _score_vectors,
    _seed_everything,
    _state_delta,
    file_sha256,
    load_causal_records,
    load_config,
    make_model,
    summarize_evaluation,
)

PROTOCOL = "evokv_ml1m_prediction_query_objective_v0"


def load_query_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    parent = document.get("parent")
    training = document.get("training")
    evaluation = document.get("evaluation")
    expected_variants = [
        {
            "architecture": "legacy",
            "id": "q1_legacy_normalized_query",
            "normalized_scoring": True,
        },
        {
            "architecture": "dense_hstu_v2",
            "id": "q2_dense_normalized_query",
            "normalized_scoring": True,
        },
    ]
    expected_strategies = [
        {"candidate_count": 20, "id": "benchmark_hard_20", "source": "benchmark"},
        {"candidate_count": 50, "id": "random_unseen_50", "source": "random_unseen"},
        {"candidate_count": 50, "id": "popular_unseen_50", "source": "train_popularity"},
        {"candidate_count": 100, "id": "random_unseen_100", "source": "random_unseen"},
        {"candidate_count": 100, "id": "popular_unseen_100", "source": "train_popularity"},
    ]
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or not isinstance(parent, dict)
        or not isinstance(training, dict)
        or not isinstance(evaluation, dict)
        or document.get("variants") != expected_variants
        or evaluation.get("strategies") != expected_strategies
        or training.get("seeds") != [4217, 14929]
        or training.get("base_epochs") != 3
        or training.get("update_epochs") != 3
        or training.get("query_loss_weight") != 16.0
    ):
        raise ValueError("ML1m prediction-query config differs")
    if file_sha256(parent["base_config"]) != parent["base_config_sha256"]:
        raise ValueError("ML1m prediction-query parent binding differs")
    return document


def _base_examples(records: dict[str, Any], selected: list[str], max_seq_len: int):
    output = []
    for user_id in selected:
        record = records[user_id]
        full = np.concatenate((record["history"], record["positive_items"]))[-max_seq_len:]
        inputs = np.concatenate((full[:-1], np.asarray([0], dtype=np.int64)))
        targets = np.concatenate((full[1:], full[-1:]))
        output.append((user_id, inputs, targets))
    return output


def _update_examples(records: dict[str, Any], selected: list[str], max_seq_len: int):
    output = []
    for user_id in selected:
        record = records[user_id]
        history = record["history"][-(max_seq_len - 1) :]
        inputs = np.concatenate((history, np.asarray([0], dtype=np.int64)))
        output.append((user_id, inputs, int(record["positive_items"][0])))
    return output


def _collate_base(batch, device: torch.device):
    lengths = torch.tensor([len(value[1]) for value in batch], dtype=torch.long, device=device)
    width = int(lengths.max().item())
    items = torch.zeros(len(batch), width, dtype=torch.long, device=device)
    targets = torch.zeros_like(items)
    for row, (_, sequence, sequence_targets) in enumerate(batch):
        items[row, : len(sequence)] = torch.as_tensor(sequence, dtype=torch.long, device=device)
        targets[row, : len(sequence_targets)] = torch.as_tensor(
            sequence_targets,
            dtype=torch.long,
            device=device,
        )
    valid = torch.arange(width, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    behaviors = valid.long()
    deltas = torch.zeros_like(items, dtype=torch.float32)
    return items, behaviors, deltas, targets, lengths, valid


def _collate_update(batch, device: torch.device):
    lengths = torch.tensor([len(value[1]) for value in batch], dtype=torch.long, device=device)
    width = int(lengths.max().item())
    items = torch.zeros(len(batch), width, dtype=torch.long, device=device)
    for row, (_, sequence, _) in enumerate(batch):
        items[row, : len(sequence)] = torch.as_tensor(sequence, dtype=torch.long, device=device)
    valid = torch.arange(width, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    behaviors = valid.long()
    deltas = torch.zeros_like(items, dtype=torch.float32)
    targets = torch.tensor([value[2] for value in batch], dtype=torch.long, device=device)
    return items, behaviors, deltas, targets, lengths


def _train_query(
    model,
    examples,
    document: dict[str, Any],
    phase: str,
    seed: int,
):
    training = document["training"]
    epochs = int(training["base_epochs"] if phase == "base" else training["update_epochs"])
    learning_rate = float(training["base_lr"] if phase == "base" else training["update_lr"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(training["weight_decay"]),
    )
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed + 9173)
    rng = np.random.default_rng(seed)
    epoch_results = []
    started = time.monotonic()
    model.train()
    for epoch in range(epochs):
        order = rng.permutation(len(examples))
        loss_sum = 0.0
        weight_sum = 0.0
        for start in range(0, len(order), int(training["batch_size"])):
            batch = [examples[int(index)] for index in order[start : start + int(training["batch_size"])]]
            optimizer.zero_grad(set_to_none=True)
            if phase == "base":
                items, behaviors, deltas, targets, lengths, valid = _collate_base(batch, device)
                hidden, _ = model(items, behaviors, deltas, lengths=lengths)
                selected_hidden = hidden[valid]
                selected_targets = targets[valid]
                weights = torch.ones_like(targets, dtype=torch.float32)
                rows = torch.arange(len(batch), device=device)
                weights[rows, lengths - 1] = float(training["query_loss_weight"])
                selected_weights = weights[valid]
            else:
                items, behaviors, deltas, selected_targets, lengths = _collate_update(batch, device)
                hidden, _ = model(items, behaviors, deltas, lengths=lengths)
                selected_hidden = model.last_hidden(hidden, lengths)
                selected_weights = torch.ones_like(selected_targets, dtype=torch.float32)
            candidates = _sample_candidates(
                selected_targets,
                int(document["model"]["num_items"]),
                int(training["negative_samples"]),
                generator,
            )
            scores = _score_vectors(
                model,
                selected_hidden,
                candidates,
                True,
                float(training["temperature"]),
            )
            losses = F.cross_entropy(
                scores,
                torch.zeros(scores.shape[0], dtype=torch.long, device=device),
                reduction="none",
            )
            loss = (losses * selected_weights).sum() / selected_weights.sum()
            if not torch.isfinite(loss):
                raise RuntimeError("ML1m prediction-query training produced non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step()
            loss_sum += float((losses.detach() * selected_weights).sum().item())
            weight_sum += float(selected_weights.sum().item())
        epoch_results.append(
            {
                "epoch": epoch + 1,
                "weighted_sampled_cross_entropy": loss_sum / weight_sum,
                "effective_weight": weight_sum,
            }
        )
        print(
            f"phase=ml1m_query_{phase} epoch={epoch + 1}/{epochs} "
            f"loss={loss_sum / weight_sum:.6f} effective_weight={weight_sum:.0f}",
            flush=True,
        )
    model.eval()
    return {
        "phase": phase,
        "epochs": epoch_results,
        "elapsed_seconds": time.monotonic() - started,
    }


def _candidate_maps(
    causal: dict[str, Any],
    selected: list[str],
    num_items: int,
    seed: int,
):
    test = causal["splits"]["test"]
    benchmark = {
        user_id: np.asarray(_base_candidates(test[user_id]), dtype=np.int64)
        for user_id in selected
    }
    random_50 = {
        user_id: _candidate_ids(test[user_id], user_id, num_items, 50, seed, filter_seen=True)
        for user_id in selected
    }
    random_100 = {
        user_id: _candidate_ids(test[user_id], user_id, num_items, 100, seed, filter_seen=True)
        for user_id in selected
    }
    popular_100 = _popular_map(
        causal["splits"]["train"],
        test,
        selected,
        num_items,
        100,
    )
    return {
        "benchmark_hard_20": benchmark,
        "random_unseen_50": random_50,
        "popular_unseen_50": {
            user_id: values[:50].copy() for user_id, values in popular_100.items()
        },
        "random_unseen_100": random_100,
        "popular_unseen_100": popular_100,
    }


def run_query_objective(config_path: str | Path) -> dict[str, Any]:
    document = load_query_config(config_path)
    output_root = Path(document["outputs"]["root"])
    summary_path = output_root / "summary.json"
    if summary_path.is_file():
        result = json.loads(summary_path.read_text())
        validate_query_summary(result, document)
        return result
    output_root.mkdir(parents=True, exist_ok=True)
    base_document = load_config(document["parent"]["base_config"])
    base_document["data"]["user_limit"] = int(document["data"]["user_limit"])
    base_document["training"].update(document["training"])
    causal = load_causal_records(base_document)
    selected = causal["selected_users"]
    max_seq_len = int(base_document["model"]["max_seq_len"])
    base_examples = _base_examples(causal["splits"]["train"], selected, max_seq_len)
    update_examples = _update_examples(causal["splits"]["dev"], selected, max_seq_len)
    candidate_maps = _candidate_maps(
        causal,
        selected,
        int(base_document["model"]["num_items"]),
        int(document["evaluation"]["candidate_seed"]),
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for ML1m prediction-query training")
    variants = []
    started = time.monotonic()
    for variant_index, variant in enumerate(document["variants"]):
        seed_results = []
        for seed_index, seed in enumerate(document["training"]["seeds"]):
            print(
                f"phase=ml1m_query_start variant={variant['id']} seed={seed}",
                flush=True,
            )
            _seed_everything(int(seed))
            previous = make_model(base_document, variant["architecture"], device)
            base_training = _train_query(
                previous,
                base_examples,
                base_document,
                "base",
                int(seed) + 1009,
            )
            previous_state = {
                name: value.detach().cpu().clone()
                for name, value in previous.state_dict().items()
            }
            theta0 = _save_checkpoint(
                output_root / "checkpoints" / variant["id"] / f"seed_{seed}" / "theta0.pt",
                previous,
                variant["architecture"],
                {"training": base_training, "prediction_protocol": "learned_query_after_history"},
            )
            current = deepcopy(previous)
            update_training = _train_query(
                current,
                update_examples,
                base_document,
                "update",
                int(seed) + 2003 + variant_index * 10007 + seed_index * 1000003,
            )
            current_state = {
                name: value.detach().cpu().clone()
                for name, value in current.state_dict().items()
            }
            parameter_delta = _state_delta(previous_state, current_state)
            theta1 = _save_checkpoint(
                output_root / "checkpoints" / variant["id"] / f"seed_{seed}" / "theta1.pt",
                current,
                variant["architecture"],
                {
                    "training": update_training,
                    "prediction_protocol": "learned_query_after_history",
                    "parameter_delta": parameter_delta,
                },
            )
            strategy_results = []
            for strategy in document["evaluation"]["strategies"]:
                evaluation = _evaluate(
                    previous,
                    current,
                    causal["splits"]["test"],
                    selected,
                    base_document,
                    True,
                    candidate_map=candidate_maps[strategy["id"]],
                    candidate_protocol_override=strategy["id"],
                    prediction_query=True,
                )
                compact = summarize_evaluation(evaluation, base_document)
                gate = _strict_ranking_gate(compact)
                result_path = (
                    output_root
                    / "variants"
                    / variant["id"]
                    / f"seed_{seed}"
                    / f"{strategy['id']}.json"
                )
                _atomic_json(
                    result_path,
                    {
                        "variant": variant,
                        "seed": seed,
                        "strategy": strategy,
                        "summary": compact,
                        "strict_ranking_gate": gate,
                        "records": evaluation["records"],
                    },
                )
                strategy_results.append(
                    {
                        "strategy": strategy,
                        "result_path": str(result_path),
                        "result_sha256": file_sha256(result_path),
                        "summary": compact,
                        "strict_ranking_gate": gate,
                    }
                )
                stale = compact["comparisons"]["recompute_over_reuse"]
                print(
                    f"phase=ml1m_query_result variant={variant['id']} seed={seed} "
                    f"strategy={strategy['id']} stale_ce={stale['candidate_cross_entropy']['absolute']:.6f} "
                    f"stale_mrr={stale['mrr']['absolute']:.6f} "
                    f"stale_ndcg10={stale['ndcg_at_10']['absolute']:.6f} passed={gate['passed']}",
                    flush=True,
                )
            seed_results.append(
                {
                    "seed": seed,
                    "base_training": base_training,
                    "update_training": update_training,
                    "parameter_delta": parameter_delta,
                    "checkpoints": {"theta0": theta0, "theta1": theta1},
                    "strategy_results": strategy_results,
                }
            )
            del previous, current
            torch.cuda.empty_cache()
        variants.append({"id": variant["id"], "seed_results": seed_results})
    stable = []
    for variant in variants:
        for strategy in document["evaluation"]["strategies"]:
            if strategy["candidate_count"] >= 50 and all(
                next(
                    value
                    for value in seed["strategy_results"]
                    if value["strategy"]["id"] == strategy["id"]
                )["strict_ranking_gate"]["passed"]
                for seed in variant["seed_results"]
            ):
                stable.append({"variant": variant["id"], "strategy": strategy["id"]})
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
            "prediction_bypass_root_cause_supported": bool(stable),
            "next": "holdout_seed_and_balance_curve" if stable else "target_aware_or_data_semantics",
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    validate_query_summary(summary, document)
    _atomic_json(summary_path, summary)
    return summary


def validate_query_summary(summary: dict[str, Any], document: dict[str, Any]) -> None:
    variants = summary.get("variants")
    if (
        summary.get("protocol") != PROTOCOL
        or summary.get("round_id") != document["round_id"]
        or summary.get("status") != "complete"
        or summary.get("scientific_result") is not False
        or not isinstance(variants, list)
        or [value.get("id") for value in variants]
        != [value["id"] for value in document["variants"]]
    ):
        raise ValueError("ML1m prediction-query summary differs")
    for variant in variants:
        seeds = variant.get("seed_results")
        if not isinstance(seeds, list) or [value.get("seed") for value in seeds] != document[
            "training"
        ]["seeds"]:
            raise ValueError("ML1m prediction-query seed coverage differs")
        for seed in seeds:
            results = seed.get("strategy_results")
            if not isinstance(results, list) or [value.get("strategy") for value in results] != document[
                "evaluation"
            ]["strategies"]:
                raise ValueError("ML1m prediction-query strategy coverage differs")
            for result in results:
                path = Path(result["result_path"])
                if not path.is_file() or file_sha256(path) != result["result_sha256"]:
                    raise ValueError("ML1m prediction-query result binding differs")
                if not result.get("summary", {}).get("sanity", {}).get("passed"):
                    raise ValueError("ML1m prediction-query sanity failed")
