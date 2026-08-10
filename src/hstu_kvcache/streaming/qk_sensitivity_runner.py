from __future__ import annotations

import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ..data.qk_stream_chain import load_corpus
from .qk_stream_runner import (
    _atomic_json,
    _dense_state,
    _load_model,
    _runtime,
)
from .qk_stream_version import (
    cache_relative_error,
    distributed_full_catalog_metrics,
    distributed_full_catalog_topk,
    distributed_projected_candidate_scores,
    eligible_training_records,
    evaluation_suffix,
    file_sha256,
    fp16_storage_fp32_consumption,
    local_role_records,
    paired_full_catalog_summary,
    prefix_inputs,
    prequential_evaluation_role_audit,
    snapshot_source_prefixes,
    summarize_full_catalog_record,
    summarize_logged_window_record,
    summarize_rank_sensitivity_record,
    summarize_window_topk_record,
)
from .sharded_edge import ExternalEmbeddingHSTU
from .xp_projected_edge import TrainableProjectedModuloEmbedding

PROTOCOL = "evokv_qk_stream_sensitivity_evaluation_v0"


def _validate_document(document: dict[str, object]) -> None:
    edge = document.get("edge")
    quality = document.get("quality")
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_user_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or not isinstance(edge, dict)
        or edge.get("source_version") != 0
        or edge.get("target_version") != 1
        or edge.get("edge") != 1
        or not isinstance(quality, dict)
        or quality.get("primary_role") != "stream_train"
        or quality.get("supplemental_role") != "fit_tuning"
        or quality.get("candidate_set") != "all_prediction_items"
        or quality.get("cutoffs") != [10, 50, 200]
        or int(quality.get("bootstrap_samples", 0)) < 1
        or int(quality.get("full_catalog_item_chunk", 0)) < 1
    ):
        raise ValueError("QK sensitivity config differs")


def _edge_delta_bitmap(
    source_counts: torch.Tensor,
    current_counts: torch.Tensor,
    *,
    num_embeddings: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> np.ndarray:
    if source_counts.shape != current_counts.shape:
        raise ValueError("QK sensitivity active ledgers differ")
    local = current_counts > source_counts
    bitmap = torch.zeros(
        num_embeddings,
        dtype=torch.uint8,
        device=device,
    )
    rows = torch.nonzero(local, as_tuple=False).flatten().to(device)
    global_rows = rows * world_size + rank
    bitmap[global_rows] = 1
    dist.all_reduce(bitmap, op=dist.ReduceOp.MAX)
    return bitmap.cpu().numpy().astype(np.bool_, copy=False)


def _logged_candidates(
    targets: torch.Tensor,
    labels: torch.Tensor,
    num_prediction_items: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    values: dict[int, bool] = {}
    for item, label in zip(targets.tolist(), labels.tolist(), strict=True):
        item = int(item)
        if 1 <= item <= num_prediction_items:
            values[item] = values.get(item, False) or bool(label)
    ids = torch.tensor(list(values), dtype=torch.int64)
    outcomes = torch.tensor(list(values.values()), dtype=torch.int64)
    return ids, outcomes


def _aggregate_sensitivity(
    records: list[dict[str, object]],
) -> dict[str, object]:
    targets = sum(int(value["targets"]) for value in records)
    if targets < 1:
        return {"records": len(records), "targets": 0}
    result: dict[str, object] = {
        "records": len(records),
        "targets": targets,
    }
    count_names = (
        "recompute_nll_wins",
        "reuse_nll_wins",
        "nll_ties",
        "recompute_rank_wins",
        "reuse_rank_wins",
        "rank_ties",
    )
    for name in count_names:
        count = sum(int(value[name]) for value in records)
        result[name] = count
        result[f"{name}_percent"] = 100.0 * count / targets
    for name in (
        "rank_improvement_sum",
        "log_rank_improvement_sum",
        "absolute_log_rank_shift_sum",
    ):
        result[name.removesuffix("_sum") + "_mean"] = sum(
            float(value[name]) for value in records
        ) / targets
    for cutoff in (10, 50, 200):
        rescues = sum(
            int(value[f"recompute_rescues_at_{cutoff}"])
            for value in records
        )
        regressions = sum(
            int(value[f"recompute_regressions_at_{cutoff}"])
            for value in records
        )
        flips = sum(
            int(value[f"decision_flips_at_{cutoff}"])
            for value in records
        )
        result[f"top{cutoff}"] = {
            "recompute_rescues": rescues,
            "recompute_regressions": regressions,
            "decision_flips": flips,
            "net_rescues": rescues - regressions,
            "net_rescues_per_10000_targets": 10_000.0
            * (rescues - regressions)
            / targets,
            "decision_flip_percent": 100.0 * flips / targets,
        }
    return result


def _paired_macro(
    records: list[dict[str, object]],
    metric_pairs: tuple[tuple[str, str, str], ...],
) -> dict[str, object]:
    result: dict[str, object] = {"records": len(records)}
    for metric, reuse_name, recompute_name in metric_pairs:
        reuse = np.asarray(
            [value[reuse_name] for value in records], dtype=np.float64
        )
        recompute = np.asarray(
            [value[recompute_name] for value in records],
            dtype=np.float64,
        )
        finite = np.isfinite(reuse) & np.isfinite(recompute)
        if not np.any(finite):
            result[metric] = {"records": 0}
            continue
        reuse_mean = float(reuse[finite].mean())
        recompute_mean = float(recompute[finite].mean())
        gap = recompute_mean - reuse_mean
        result[metric] = {
            "records": int(np.count_nonzero(finite)),
            "reuse": reuse_mean,
            "recompute": recompute_mean,
            "recompute_minus_reuse": gap,
            "relative_to_reuse_percent": (
                100.0 * gap / reuse_mean if reuse_mean else None
            ),
        }
    return result


def _window_summary(records: list[dict[str, object]]) -> dict[str, object]:
    eligible = [value for value in records if int(value["positive_items"]) > 0]
    pairs = []
    for cutoff in (10, 50, 200):
        for metric in ("hit", "recall", "ndcg"):
            pairs.append(
                (
                    f"{metric}_at_{cutoff}",
                    f"reuse_{metric}_at_{cutoff}",
                    f"recompute_{metric}_at_{cutoff}",
                )
            )
    return {
        "records": len(records),
        "records_with_positive_items": len(eligible),
        "positive_items": sum(
            int(value["positive_items"]) for value in eligible
        ),
        "user_macro": _paired_macro(eligible, tuple(pairs)),
    }


def _logged_summary(records: list[dict[str, object]]) -> dict[str, object]:
    return _paired_macro(
        records,
        (
            ("auc", "reuse_auc", "recompute_auc"),
            (
                "average_precision",
                "reuse_average_precision",
                "recompute_average_precision",
            ),
        ),
    )


def _cohort_name(
    history_length: int,
    update_overlap: float,
) -> tuple[str, str]:
    if history_length < 80:
        history = "history_64_79"
    elif history_length < 96:
        history = "history_80_95"
    else:
        history = "history_96_plus"
    if update_overlap < 0.9:
        overlap = "updated_fraction_lt_0.90"
    elif update_overlap < 0.99:
        overlap = "updated_fraction_0.90_0.99"
    elif update_overlap < 1.0:
        overlap = "updated_fraction_0.99_lt_1.00"
    else:
        overlap = "updated_fraction_1.00"
    return history, overlap


def _summarize_records(
    records: list[dict[str, object]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    if not any(int(value["targets"]) > 0 for value in records):
        return {
            "records": len(records),
            "positive_targets": 0,
        }
    full = paired_full_catalog_summary(
        records,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    sensitivity = _aggregate_sensitivity(
        [value["sensitivity"] for value in records]
    )
    window = _window_summary([value["window"] for value in records])
    logged = _logged_summary(
        [value["logged"] for value in records if value["logged"] is not None]
    )
    return {
        "full_catalog_next_item": full,
        "paired_target_sensitivity": sensitivity,
        "full_catalog_next_window": window,
        "logged_ordinal_window": logged,
    }


@torch.no_grad()
def _evaluate_role(
    document: dict[str, object],
    role: str,
    corpus,
    spec,
    current_dense: ExternalEmbeddingHSTU,
    current_embedding: TrainableProjectedModuloEmbedding,
    source_dense: ExternalEmbeddingHSTU,
    source_vectors: list[torch.Tensor],
    edge_delta: np.ndarray,
    training_participants: set[int],
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, object]:
    quality = document["quality"]
    records = local_role_records(corpus, role, rank, world_size)
    if len(records) != len(source_vectors):
        raise ValueError("QK sensitivity source coverage differs")
    target_chunk = int(quality["target_chunk"])
    item_chunk = int(quality["full_catalog_item_chunk"])
    progress_every = int(document["execution"]["progress_every_records"])
    local_results = []
    started = time.perf_counter()
    for ordinal, (raw_record, old_vectors) in enumerate(
        zip(records, source_vectors, strict=True)
    ):
        record = int(raw_record)
        prefix_items, prefix_behaviors, prefix_deltas, prefix_length = (
            prefix_inputs(corpus, record, int(document["edge"]["edge"]))
        )
        lengths = torch.tensor([prefix_length], dtype=torch.int64, device=device)
        current_vectors = current_embedding(
            prefix_items.unsqueeze(0).to(device), lengths
        )
        old_cache = source_dense.core.compute_kv_from_item_embeddings(
            old_vectors.unsqueeze(0).to(device),
            prefix_behaviors.unsqueeze(0).to(device),
            prefix_deltas.unsqueeze(0).to(device),
            lengths,
        )
        recompute_cache = current_dense.core.compute_kv_from_item_embeddings(
            current_vectors,
            prefix_behaviors.unsqueeze(0).to(device),
            prefix_deltas.unsqueeze(0).to(device),
            lengths,
        )
        old_cache = fp16_storage_fp32_consumption(old_cache)
        recompute_cache = fp16_storage_fp32_consumption(recompute_cache)
        kv_error = cache_relative_error(old_cache, recompute_cache)
        suffix_items, suffix_behaviors, suffix_deltas, targets, labels = (
            evaluation_suffix(corpus, record, int(document["edge"]["edge"]))
        )
        suffix_lengths = torch.tensor(
            [len(suffix_items)], dtype=torch.int64, device=device
        )
        suffix_vectors = current_embedding(
            suffix_items.unsqueeze(0).to(device), suffix_lengths
        )
        reuse_hidden, _ = current_dense.core.forward_with_cache_from_item_embeddings(
            old_cache,
            suffix_vectors,
            suffix_behaviors.unsqueeze(0).to(device),
            suffix_deltas.unsqueeze(0).to(device),
        )
        recompute_hidden, _ = (
            current_dense.core.forward_with_cache_from_item_embeddings(
                recompute_cache,
                suffix_vectors,
                suffix_behaviors.unsqueeze(0).to(device),
                suffix_deltas.unsqueeze(0).to(device),
            )
        )
        mask = labels.to(device)
        positive_ids = targets.to(device)[mask]
        reuse_positive = reuse_hidden[0][mask]
        recompute_positive = recompute_hidden[0][mask]
        maximum_targets = torch.tensor(
            len(positive_ids), dtype=torch.int64, device=device
        )
        dist.all_reduce(maximum_targets, op=dist.ReduceOp.MAX)
        nll_values = []
        rank_values = []
        for step in range(math.ceil(int(maximum_targets.item()) / target_chunk)):
            start = step * target_chunk
            real = min(target_chunk, max(0, len(positive_ids) - start))
            hidden = torch.zeros(
                (2, target_chunk, spec.hidden_size),
                dtype=torch.float32,
                device=device,
            )
            candidates = torch.zeros(
                target_chunk, dtype=torch.int64, device=device
            )
            if real:
                hidden[0, :real] = reuse_positive[start : start + real]
                hidden[1, :real] = recompute_positive[start : start + real]
                candidates[:real] = positive_ids[start : start + real]
            nll, ranks = distributed_full_catalog_metrics(
                current_embedding,
                hidden,
                candidates,
                real,
                num_prediction_items=spec.num_prediction_items,
                item_chunk=item_chunk,
            )
            nll_values.append(nll)
            rank_values.append(ranks)
        record_nll = (
            torch.cat(nll_values, dim=1)
            if nll_values
            else torch.empty((2, 0), device=device)
        )
        record_ranks = (
            torch.cat(rank_values, dim=1)
            if rank_values
            else torch.empty((2, 0), dtype=torch.int64, device=device)
        )
        boundary_hidden = torch.stack(
            (reuse_hidden[0, 0], recompute_hidden[0, 0])
        )
        _, topk_ids = distributed_full_catalog_topk(
            current_embedding,
            boundary_hidden,
            num_prediction_items=spec.num_prediction_items,
            maximum_k=200,
            item_chunk=item_chunk,
        )
        window = summarize_window_topk_record(
            topk_ids,
            positive_ids,
        )
        logged_ids, logged_labels = _logged_candidates(
            targets,
            labels,
            spec.num_prediction_items,
        )
        local_candidates = len(logged_ids)
        maximum_candidates = torch.tensor(
            local_candidates, dtype=torch.int64, device=device
        )
        dist.all_reduce(maximum_candidates, op=dist.ReduceOp.MAX)
        padded_width = max(2, int(maximum_candidates.item()))
        padded = torch.zeros((1, padded_width), dtype=torch.int64, device=device)
        if local_candidates:
            padded[0, :local_candidates] = logged_ids.to(device)
        logged_scores = distributed_projected_candidate_scores(
            current_embedding,
            boundary_hidden[:, None, :],
            padded,
            1,
        )[:, 0, :local_candidates]
        logged = (
            summarize_logged_window_record(
                logged_scores,
                logged_labels.to(device),
            )
            if local_candidates
            else None
        )
        full = summarize_full_catalog_record(record_nll, record_ranks)
        sensitivity = summarize_rank_sensitivity_record(
            record_nll,
            record_ranks,
        )
        prefix_array = prefix_items.numpy()
        overlap = float(edge_delta[prefix_array].mean())
        history_cohort, overlap_cohort = _cohort_name(
            prefix_length, overlap
        )
        local_results.append(
            {
                **full,
                "record": record,
                "user_id": int(corpus.arrays["record_user_ids"][record]),
                "training_participant": record in training_participants,
                "history_length": prefix_length,
                "update_overlap_fraction": overlap,
                "history_cohort": history_cohort,
                "update_overlap_cohort": overlap_cohort,
                "prefix_cache_relative_error": kv_error,
                "sensitivity": sensitivity,
                "window": window,
                "logged": logged,
            }
        )
        if (
            ordinal == 0
            or (ordinal + 1) % progress_every == 0
            or ordinal + 1 == len(records)
        ):
            print(
                f"phase=qk_sensitivity role={role} rank={rank} "
                f"record={ordinal + 1}/{len(records)}",
                flush=True,
            )
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, local_results)
    if rank != 0:
        return {}
    combined = [value for piece in gathered for value in piece]
    combined.sort(key=lambda value: int(value["record"]))
    samples = int(quality["bootstrap_samples"])
    seed = int(quality["bootstrap_seed"])
    main = _summarize_records(
        combined,
        bootstrap_samples=samples,
        bootstrap_seed=seed + (0 if role == "stream_train" else 10_000_019),
    )
    cohorts: dict[str, object] = {}
    if role == "stream_train":
        participant = [value for value in combined if value["training_participant"]]
        nonparticipant = [
            value for value in combined if not value["training_participant"]
        ]
        cohorts["optimizer_participants"] = _summarize_records(
            participant,
            bootstrap_samples=samples,
            bootstrap_seed=seed + 20_000_033,
        )
        cohorts["nonparticipants"] = _summarize_records(
            nonparticipant,
            bootstrap_samples=samples,
            bootstrap_seed=seed + 30_000_047,
        )
    for name in (
        "history_64_79",
        "history_80_95",
        "history_96_plus",
        "updated_fraction_lt_0.90",
        "updated_fraction_0.90_0.99",
        "updated_fraction_0.99_lt_1.00",
        "updated_fraction_1.00",
    ):
        selected = [
            value
            for value in combined
            if value["history_cohort"] == name
            or value["update_overlap_cohort"] == name
        ]
        if selected:
            cohorts[name] = {
                "records": len(selected),
                "full_catalog_next_item": paired_full_catalog_summary(
                    selected,
                    bootstrap_samples=max(200, samples // 5),
                    bootstrap_seed=seed + len(cohorts) * 1_000_003,
                ),
                "paired_target_sensitivity": _aggregate_sensitivity(
                    [value["sensitivity"] for value in selected]
                ),
                "full_catalog_next_window": _window_summary(
                    [value["window"] for value in selected]
                ),
            }
    return {
        "role": role,
        "records": len(combined),
        "runtime_seconds": time.perf_counter() - started,
        "all": main,
        "cohorts": cohorts,
    }


def run_qk_sensitivity_evaluation(
    config_path: Path,
) -> dict[str, object] | None:
    document = json.loads(config_path.read_text())
    _validate_document(document)
    rank, world_size, local_rank, device = _runtime()
    started = time.perf_counter()
    try:
        output = Path(document["outputs"]["result"])
        if output.exists():
            raise FileExistsError("QK sensitivity result already exists")
        corpus = load_corpus(document["data"]["corpus"])
        if corpus.file_sha256 != document["data"]["corpus_sha256"]:
            raise ValueError("QK sensitivity corpus hash differs")
        audit = prequential_evaluation_role_audit(
            corpus,
            int(document["edge"]["edge"]),
        )
        source = document["source_checkpoint"]
        current = document["current_checkpoint"]
        source_root = Path(source["root"])
        current_root = Path(current["root"])
        source_manifest_path = source_root / "theta_0" / "manifest.json"
        current_manifest_path = current_root / "theta_1" / "manifest.json"
        if (
            file_sha256(source_manifest_path) != source["manifest_sha256"]
            or file_sha256(current_manifest_path) != current["manifest_sha256"]
        ):
            raise ValueError("QK sensitivity checkpoint hash differs")
        spec, source_dense, source_embedding, source_tracker, _ = _load_model(
            source_root,
            0,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        old_dense_state = _dense_state(source_dense)
        source_counts = source_tracker.local_update_counts.clone()
        snapshots = snapshot_source_prefixes(
            corpus,
            source_embedding,
            ("stream_train", "fit_tuning"),
            1,
            rank,
            world_size,
            device,
            int(document["execution"]["snapshot_batch_size_per_rank"]),
        )
        del source_dense, source_embedding, source_tracker
        gc.collect()
        torch.cuda.empty_cache()
        spec, current_dense, current_embedding, current_tracker, _ = _load_model(
            current_root,
            1,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        edge_delta = _edge_delta_bitmap(
            source_counts,
            current_tracker.local_update_counts,
            num_embeddings=spec.num_embeddings,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        source_dense = ExternalEmbeddingHSTU(spec.hstu_config()).to(device)
        source_dense.load_state_dict(old_dense_state)
        source_dense.eval()
        current_dense.eval()
        current_embedding.eval()
        participants = set(eligible_training_records(corpus, 1).tolist())
        primary = _evaluate_role(
            document,
            "stream_train",
            corpus,
            spec,
            current_dense,
            current_embedding,
            source_dense,
            snapshots["stream_train"],
            edge_delta,
            participants,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        supplemental = _evaluate_role(
            document,
            "fit_tuning",
            corpus,
            spec,
            current_dense,
            current_embedding,
            source_dense,
            snapshots["fit_tuning"],
            edge_delta,
            participants,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        result = {
            "protocol": PROTOCOL,
            "status": "complete_development_measurement",
            "scientific_result": False,
            "formal_result": False,
            "dataset": "tenrec-qk",
            "config": {
                "path": str(config_path),
                "sha256": file_sha256(config_path),
            },
            "edge": document["edge"],
            "role_audit": audit,
            "source_checkpoint": source,
            "current_checkpoint": current,
            "quality": {
                "same_current_model": 1,
                "reuse_source_version": 0,
                "recompute_version": 1,
                "candidate_set": "all prediction item ids [1, 250000]",
                "primary_update_local": primary,
                "supplemental_disjoint_user": supplemental,
                "qualification_consumed": False,
                "final_consumed": False,
                "evaluation_labels_used_for_role_or_cohort_selection": False,
                "training_window_labels_define_optimizer_participant": True,
            },
            "execution": {
                "world_size": world_size,
                "local_rank": local_rank,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "runtime_seconds": time.perf_counter() - started,
                "hbm": {
                    "allocated_bytes": torch.cuda.memory_allocated(device),
                    "reserved_bytes": torch.cuda.memory_reserved(device),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                },
            },
        }
        if rank == 0:
            _atomic_json(output, result)
        return result if rank == 0 else None
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
