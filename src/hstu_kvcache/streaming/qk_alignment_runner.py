from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ..data.qk_stream_chain import QKStreamChainCorpus, load_corpus
from .qk_stream_runner import _atomic_json, _dense_state, _load_model, _runtime
from .qk_stream_version import (
    distributed_full_catalog_metrics,
    eligible_training_records,
    evaluation_suffix,
    file_sha256,
    fp16_storage_fp32_consumption,
    local_role_records,
    paired_full_catalog_summary,
    prefix_inputs,
    prequential_evaluation_role_audit,
    record_window,
    snapshot_source_prefixes,
    summarize_full_catalog_record,
)
from .sharded_edge import ExternalEmbeddingHSTU
from .xp_projected_edge import TrainableProjectedModuloEmbedding

PROTOCOL = "evokv_qk_stream_alignment_diagnostic_v0"
MODES = ("rolling_next_item", "boundary_multi_positive")
COHORTS = (
    "all",
    "first_positive",
    "offset_0",
    "offset_lt_2",
    "offset_lt_4",
    "offset_lt_8",
    "offset_lt_16",
    "train_exposure_supported",
    "train_positive_supported",
    "early_8_train_positive_supported",
    "first_positive_train_positive_supported",
    "prefix_positive_mass_q4",
    "early_8_positive_supported_prefix_q4",
)


def _validate_document(document: dict[str, object]) -> None:
    edge = document.get("edge")
    quality = document.get("quality")
    execution = document.get("execution")
    if (
        not isinstance(edge, dict)
        or not isinstance(quality, dict)
        or not isinstance(execution, dict)
    ):
        raise ValueError("QK alignment config differs")
    source_version = edge.get("source_version")
    target_version = edge.get("target_version")
    edge_index = edge.get("edge")
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_user_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or not isinstance(source_version, int)
        or not isinstance(target_version, int)
        or not isinstance(edge_index, int)
        or source_version < 0
        or target_version != source_version + 1
        or edge_index != target_version
        or quality.get("primary_role") != "stream_train"
        or quality.get("evaluation_users") != "optimizer_participants"
        or quality.get("candidate_set") != "all_prediction_items"
        or quality.get("modes") != list(MODES)
        or quality.get("cohorts") != list(COHORTS)
        or quality.get("early_offsets") != [2, 4, 8, 16]
        or int(quality.get("target_chunk", 0)) < 1
        or int(quality.get("full_catalog_item_chunk", 0)) < 1
        or int(quality.get("bootstrap_samples", 0)) < 1
        or quality.get("preferred_relative_gap_percent_range") != [5.0, 10.0]
        or int(quality.get("minimum_cohort_targets", 0)) < 1
        or int(execution.get("world_size", 0)) != 2
    ):
        raise ValueError("QK alignment config differs")


def training_alignment_state(
    corpus: QKStreamChainCorpus,
    *,
    edge: int,
    num_embeddings: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    participants = eligible_training_records(corpus, edge)
    offsets = corpus.arrays["record_offsets"]
    items = corpus.arrays["item_idx"]
    labels = corpus.arrays["label"]
    exposure_counts = np.zeros(num_embeddings, dtype=np.int64)
    positive_counts = np.zeros(num_embeddings, dtype=np.int64)
    for raw_record in participants:
        record = int(raw_record)
        previous, current, _ = record_window(corpus, record, edge)
        start = int(offsets[record])
        values = items[start + previous + 1 : start + current + 1].astype(
            np.int64, copy=False
        )
        outcomes = labels[
            start + previous + 1 : start + current + 1
        ].astype(np.bool_, copy=False)
        np.add.at(exposure_counts, values, 1)
        np.add.at(positive_counts, values[outcomes], 1)
    prefix_positive_mass = np.full(
        len(corpus.arrays["record_user_ids"]), np.nan, dtype=np.float64
    )
    prefix_positive_fraction = np.full_like(prefix_positive_mass, np.nan)
    for raw_record in participants:
        record = int(raw_record)
        _, current, _ = record_window(corpus, record, edge)
        start = int(offsets[record])
        prefix = items[start : start + current].astype(np.int64, copy=False)
        counts = positive_counts[prefix]
        prefix_positive_mass[record] = float(np.log1p(counts).mean())
        prefix_positive_fraction[record] = float(np.mean(counts > 0))
    values = prefix_positive_mass[participants]
    quartile_boundaries = np.quantile(values, [0.25, 0.5, 0.75])
    prefix_quartile = np.zeros(len(prefix_positive_mass), dtype=np.uint8)
    prefix_quartile[participants] = 1 + np.searchsorted(
        quartile_boundaries, values, side="right"
    ).astype(np.uint8)
    state = {
        "exposure_counts": exposure_counts,
        "positive_counts": positive_counts,
        "prefix_positive_mass": prefix_positive_mass,
        "prefix_positive_fraction": prefix_positive_fraction,
        "prefix_quartile": prefix_quartile,
    }
    metadata = {
        "participants": len(participants),
        "training_window": edge,
        "source": f"stream_train window{edge} only",
        "evaluation_window_labels_used": False,
        "exposure_events": int(exposure_counts.sum()),
        "positive_events": int(positive_counts.sum()),
        "exposure_supported_rows": int(np.count_nonzero(exposure_counts)),
        "positive_supported_rows": int(np.count_nonzero(positive_counts)),
        "prefix_positive_mass_quartile_boundaries": quartile_boundaries.tolist(),
        "exposure_counts_sha256": hashlib.sha256(
            exposure_counts.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "positive_counts_sha256": hashlib.sha256(
            positive_counts.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
    }
    return state, metadata


def alignment_cohort_masks(
    target_offsets: np.ndarray,
    target_orders: np.ndarray,
    exposure_supported: np.ndarray,
    positive_supported: np.ndarray,
    prefix_quartile: int,
) -> dict[str, np.ndarray]:
    arrays = (
        target_offsets,
        target_orders,
        exposure_supported,
        positive_supported,
    )
    if (
        any(value.ndim != 1 for value in arrays)
        or len({len(value) for value in arrays}) != 1
        or prefix_quartile not in {1, 2, 3, 4}
    ):
        raise ValueError("QK alignment cohort inputs differ")
    target_offsets = target_offsets.astype(np.int64, copy=False)
    target_orders = target_orders.astype(np.int64, copy=False)
    exposure_supported = exposure_supported.astype(np.bool_, copy=False)
    positive_supported = positive_supported.astype(np.bool_, copy=False)
    all_targets = np.ones(len(target_offsets), dtype=np.bool_)
    first = target_orders == 0
    early_8 = target_offsets < 8
    q4 = np.full(len(target_offsets), prefix_quartile == 4, dtype=np.bool_)
    return {
        "all": all_targets,
        "first_positive": first,
        "offset_0": target_offsets == 0,
        "offset_lt_2": target_offsets < 2,
        "offset_lt_4": target_offsets < 4,
        "offset_lt_8": early_8,
        "offset_lt_16": target_offsets < 16,
        "train_exposure_supported": exposure_supported,
        "train_positive_supported": positive_supported,
        "early_8_train_positive_supported": early_8 & positive_supported,
        "first_positive_train_positive_supported": first & positive_supported,
        "prefix_positive_mass_q4": q4,
        "early_8_positive_supported_prefix_q4": (
            early_8 & positive_supported & q4
        ),
    }


def _summarize_mode_cohort(
    payloads: list[dict[str, object]],
    *,
    mode: str,
    cohort: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    if mode not in MODES or cohort not in COHORTS:
        raise ValueError("QK alignment summary request differs")
    method_indices = (0, 1) if mode == "rolling_next_item" else (2, 3)
    records = []
    for payload in payloads:
        masks = alignment_cohort_masks(
            payload["target_offsets"],
            payload["target_orders"],
            payload["exposure_supported"],
            payload["positive_supported"],
            int(payload["prefix_quartile"]),
        )
        mask = masks[cohort]
        if not np.any(mask):
            continue
        nll = torch.from_numpy(payload["nll_by_method"])[
            list(method_indices)
        ][:, mask]
        ranks = torch.from_numpy(payload["ranks_by_method"])[
            list(method_indices)
        ][:, mask]
        value = summarize_full_catalog_record(nll, ranks)
        value["record"] = int(payload["record"])
        records.append(value)
    if not records:
        return {
            "records": 0,
            "positive_targets": 0,
        }
    return paired_full_catalog_summary(
        records,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


def summarize_alignment(
    payloads: list[dict[str, object]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    result: dict[str, object] = {}
    ordinal = 0
    for mode in MODES:
        cohorts = {}
        for cohort in COHORTS:
            cohorts[cohort] = _summarize_mode_cohort(
                payloads,
                mode=mode,
                cohort=cohort,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + ordinal * 1_000_003,
            )
            ordinal += 1
        result[mode] = {"cohorts": cohorts}
    return result


def _alignment_gate(
    summary: dict[str, object],
    *,
    candidates: list[dict[str, str]],
    minimum_targets: int,
    relative_range: list[float],
) -> dict[str, object]:
    rows = []
    for candidate in candidates:
        mode = candidate["mode"]
        cohort = candidate["cohort"]
        value = summary[mode]["cohorts"][cohort]
        targets = int(value.get("positive_targets", 0))
        if targets:
            ndcg = value["gaps"]["ndcg_at_10"]
            mrr = value["gaps"]["mrr"]
            relative = ndcg["relative_percent"]
            passed = bool(
                targets >= minimum_targets
                and relative_range[0] <= relative <= relative_range[1]
                and ndcg["positive_direction_with_ci"]
                and mrr["positive_direction_with_ci"]
            )
        else:
            ndcg = None
            mrr = None
            relative = None
            passed = False
        rows.append(
            {
                "mode": mode,
                "cohort": cohort,
                "positive_targets": targets,
                "ndcg_at_10_relative_percent": relative,
                "ndcg_at_10_positive_ci": (
                    None if ndcg is None else ndcg["positive_direction_with_ci"]
                ),
                "mrr_positive_ci": (
                    None if mrr is None else mrr["positive_direction_with_ci"]
                ),
                "preferred_gap_passed": passed,
            }
        )
    admitted = [value for value in rows if value["preferred_gap_passed"]]
    return {
        "criterion": {
            "minimum_targets": minimum_targets,
            "ndcg_at_10_relative_gap_percent_range": relative_range,
            "ndcg_at_10_and_mrr_cluster_ci_lower_above_zero": True,
            "cohorts_defined_without_evaluation_quality": True,
        },
        "status": "aligned_protocol_found" if admitted else "no_aligned_protocol",
        "admitted": admitted,
        "all_checked": rows,
    }


@torch.no_grad()
def _evaluate(
    document: dict[str, object],
    corpus: QKStreamChainCorpus,
    spec,
    current_dense: ExternalEmbeddingHSTU,
    current_embedding: TrainableProjectedModuloEmbedding,
    source_dense: ExternalEmbeddingHSTU,
    source_vectors: list[torch.Tensor],
    alignment_state: dict[str, np.ndarray],
    participants: set[int],
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, object]:
    quality = document["quality"]
    records = local_role_records(corpus, "stream_train", rank, world_size)
    if len(records) != len(source_vectors):
        raise ValueError("QK alignment source coverage differs")
    target_chunk = int(quality["target_chunk"])
    item_chunk = int(quality["full_catalog_item_chunk"])
    progress_every = int(document["execution"]["progress_every_records"])
    local_payloads = []
    started = time.perf_counter()
    for ordinal, (raw_record, old_vectors) in enumerate(
        zip(records, source_vectors, strict=True)
    ):
        record = int(raw_record)
        prefix_items, prefix_behaviors, prefix_deltas, prefix_length = prefix_inputs(
            corpus, record, int(document["edge"]["edge"])
        )
        lengths = torch.tensor([prefix_length], dtype=torch.int64, device=device)
        current_vectors = current_embedding(prefix_items.unsqueeze(0).to(device), lengths)
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
        suffix_items, suffix_behaviors, suffix_deltas, targets, labels = evaluation_suffix(
            corpus, record, int(document["edge"]["edge"])
        )
        suffix_lengths = torch.tensor([len(suffix_items)], dtype=torch.int64, device=device)
        suffix_vectors = current_embedding(suffix_items.unsqueeze(0).to(device), suffix_lengths)
        reuse_hidden, _ = current_dense.core.forward_with_cache_from_item_embeddings(
            old_cache,
            suffix_vectors,
            suffix_behaviors.unsqueeze(0).to(device),
            suffix_deltas.unsqueeze(0).to(device),
        )
        recompute_hidden, _ = current_dense.core.forward_with_cache_from_item_embeddings(
            recompute_cache,
            suffix_vectors,
            suffix_behaviors.unsqueeze(0).to(device),
            suffix_deltas.unsqueeze(0).to(device),
        )
        participant = record in participants
        mask = labels.to(device) if participant else torch.zeros_like(labels, device=device)
        positive_ids = targets.to(device)[mask]
        reuse_positive = reuse_hidden[0][mask]
        recompute_positive = recompute_hidden[0][mask]
        maximum_targets = torch.tensor(len(positive_ids), dtype=torch.int64, device=device)
        dist.all_reduce(maximum_targets, op=dist.ReduceOp.MAX)
        nll_values = []
        rank_values = []
        steps = math.ceil(int(maximum_targets.item()) / target_chunk)
        for step in range(steps):
            start = step * target_chunk
            real = min(target_chunk, max(0, len(positive_ids) - start))
            rolling_hidden = torch.zeros(
                (2, target_chunk, spec.hidden_size),
                dtype=torch.float32,
                device=device,
            )
            boundary_hidden = torch.zeros_like(rolling_hidden)
            candidates = torch.zeros(target_chunk, dtype=torch.int64, device=device)
            if real:
                rolling_hidden[0, :real] = reuse_positive[start : start + real]
                rolling_hidden[1, :real] = recompute_positive[start : start + real]
                boundary_hidden[0, :real] = reuse_hidden[0, 0]
                boundary_hidden[1, :real] = recompute_hidden[0, 0]
                candidates[:real] = positive_ids[start : start + real]
            rolling_nll, rolling_ranks = distributed_full_catalog_metrics(
                current_embedding,
                rolling_hidden,
                candidates,
                real,
                num_prediction_items=spec.num_prediction_items,
                item_chunk=item_chunk,
            )
            boundary_nll, boundary_ranks = distributed_full_catalog_metrics(
                current_embedding,
                boundary_hidden,
                candidates,
                real,
                num_prediction_items=spec.num_prediction_items,
                item_chunk=item_chunk,
            )
            nll_values.append(torch.cat((rolling_nll, boundary_nll), dim=0))
            rank_values.append(torch.cat((rolling_ranks, boundary_ranks), dim=0))
        if participant:
            nll_by_method = (
                torch.cat(nll_values, dim=1).cpu().numpy()
                if nll_values
                else np.empty((4, 0), dtype=np.float32)
            )
            ranks_by_method = (
                torch.cat(rank_values, dim=1).cpu().numpy()
                if rank_values
                else np.empty((4, 0), dtype=np.int64)
            )
            target_offsets = torch.nonzero(labels, as_tuple=False).flatten().numpy()
            selected_targets = targets[labels].numpy()
            local_payloads.append(
                {
                    "record": record,
                    "target_offsets": target_offsets.astype(np.int16, copy=False),
                    "target_orders": np.arange(len(target_offsets), dtype=np.int16),
                    "exposure_supported": (
                        alignment_state["exposure_counts"][selected_targets] > 0
                    ),
                    "positive_supported": (
                        alignment_state["positive_counts"][selected_targets] > 0
                    ),
                    "prefix_quartile": int(
                        alignment_state["prefix_quartile"][record]
                    ),
                    "nll_by_method": nll_by_method,
                    "ranks_by_method": ranks_by_method,
                }
            )
        del (
            current_vectors,
            old_cache,
            recompute_cache,
            suffix_vectors,
            reuse_hidden,
            recompute_hidden,
        )
        if (
            ordinal == 0
            or (ordinal + 1) % progress_every == 0
            or ordinal + 1 == len(records)
        ):
            print(
                f"phase=qk_alignment rank={rank} "
                f"record={ordinal + 1}/{len(records)}",
                flush=True,
            )
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, local_payloads)
    if rank != 0:
        return {}
    combined = [value for piece in gathered for value in piece]
    combined.sort(key=lambda value: int(value["record"]))
    summary = summarize_alignment(
        combined,
        bootstrap_samples=int(quality["bootstrap_samples"]),
        bootstrap_seed=int(quality["bootstrap_seed"]),
    )
    gate = _alignment_gate(
        summary,
        candidates=list(quality["gate_candidates"]),
        minimum_targets=int(quality["minimum_cohort_targets"]),
        relative_range=list(quality["preferred_relative_gap_percent_range"]),
    )
    return {
        "records": len(combined),
        "runtime_seconds": time.perf_counter() - started,
        "summary": summary,
        "alignment_gate": gate,
    }


def run_qk_stream_alignment_diagnostic(
    config_path: Path,
) -> dict[str, object] | None:
    document = json.loads(config_path.read_text())
    _validate_document(document)
    rank, world_size, local_rank, device = _runtime()
    started = time.perf_counter()
    try:
        output = Path(document["outputs"]["result"])
        if output.exists():
            raise FileExistsError("QK alignment result already exists")
        corpus = load_corpus(document["data"]["corpus"])
        if corpus.file_sha256 != document["data"]["corpus_sha256"]:
            raise ValueError("QK alignment corpus hash differs")
        edge = int(document["edge"]["edge"])
        source_version = int(document["edge"]["source_version"])
        target_version = int(document["edge"]["target_version"])
        audit = prequential_evaluation_role_audit(corpus, edge)
        source = document["source_checkpoint"]
        current = document["current_checkpoint"]
        source_root = Path(source["root"])
        current_root = Path(current["root"])
        source_manifest = (
            source_root / f"theta_{source_version}" / "manifest.json"
        )
        current_manifest = (
            current_root / f"theta_{target_version}" / "manifest.json"
        )
        if (
            file_sha256(source_manifest) != source["manifest_sha256"]
            or file_sha256(current_manifest) != current["manifest_sha256"]
        ):
            raise ValueError("QK alignment checkpoint hash differs")
        spec, source_dense, source_embedding, _, _ = _load_model(
            source_root,
            source_version,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        source_dense_state = _dense_state(source_dense)
        snapshots = snapshot_source_prefixes(
            corpus,
            source_embedding,
            ("stream_train",),
            edge,
            rank,
            world_size,
            device,
            int(document["execution"]["snapshot_batch_size_per_rank"]),
        )
        del source_dense, source_embedding
        gc.collect()
        torch.cuda.empty_cache()
        spec, current_dense, current_embedding, _, _ = _load_model(
            current_root,
            target_version,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        source_dense = ExternalEmbeddingHSTU(spec.hstu_config()).to(device)
        source_dense.load_state_dict(source_dense_state)
        source_dense.eval()
        current_dense.eval()
        current_embedding.eval()
        alignment_state, alignment_metadata = training_alignment_state(
            corpus,
            edge=edge,
            num_embeddings=spec.num_embeddings,
        )
        participants = set(eligible_training_records(corpus, edge).tolist())
        evaluation = _evaluate(
            document,
            corpus,
            spec,
            current_dense,
            current_embedding,
            source_dense,
            snapshots["stream_train"],
            alignment_state,
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
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "edge": document["edge"],
            "role_audit": audit,
            "source_checkpoint": source,
            "current_checkpoint": current,
            "training_alignment": alignment_metadata,
            "quality": {
                "candidate_set": "all prediction item ids [1, 250000]",
                "evaluation": evaluation,
                "qualification_consumed": False,
                "final_consumed": False,
                "evaluation_window_labels_used_for_cohort_definition": False,
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
