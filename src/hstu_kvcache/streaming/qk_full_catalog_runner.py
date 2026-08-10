from __future__ import annotations

import gc
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ..data.qk_stream_chain import load_corpus
from .qk_stream_runner import (
    _atomic_json,
    _cleanup_work,
    _dense_state,
    _load_candidate_training,
    _load_model,
    _runtime,
    _save_candidate,
    _seed_everything,
    _train_edge,
)
from .qk_stream_version import (
    QKStreamFullCatalogBinding,
    cache_relative_error,
    distributed_full_catalog_metrics,
    evaluation_suffix,
    file_sha256,
    fp16_storage_fp32_consumption,
    local_role_records,
    paired_full_catalog_summary,
    prefix_inputs,
    snapshot_source_prefixes,
    summarize_full_catalog_record,
    validate_full_catalog_binding_document,
)
from .sharded_edge import ExternalEmbeddingHSTU
from .xp_projected_edge import (
    TrainableProjectedModuloEmbedding,
    XPProjectedModelSpec,
)


@torch.no_grad()
def _evaluate_full_catalog_role(
    document: dict[str, object],
    binding: QKStreamFullCatalogBinding,
    role: str,
    corpus,
    spec: XPProjectedModelSpec,
    current_dense: ExternalEmbeddingHSTU,
    current_embedding: TrainableProjectedModuloEmbedding,
    source_dense: ExternalEmbeddingHSTU,
    source_vectors: list[torch.Tensor],
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, object]:
    records = local_role_records(corpus, role, rank, world_size)
    if len(records) != len(source_vectors):
        raise ValueError("QK full-catalog source snapshot coverage differs")
    target_chunk = int(document["quality"]["target_chunk"])
    progress_every = int(
        document["execution"]["quality_progress_every_records"]
    )
    local_results = []
    started = time.perf_counter()
    for ordinal, (raw_record, old_vectors) in enumerate(
        zip(records, source_vectors, strict=True)
    ):
        record = int(raw_record)
        prefix_items, prefix_behaviors, prefix_deltas, prefix_length = (
            prefix_inputs(corpus, record, binding.edge)
        )
        lengths = torch.tensor(
            [prefix_length], dtype=torch.int64, device=device
        )
        current_vectors = current_embedding(
            prefix_items.unsqueeze(0).to(device), lengths
        )
        old_cache = source_dense.core.compute_kv_from_item_embeddings(
            old_vectors.unsqueeze(0).to(device),
            prefix_behaviors.unsqueeze(0).to(device),
            prefix_deltas.unsqueeze(0).to(device),
            lengths,
        )
        recompute_cache = (
            current_dense.core.compute_kv_from_item_embeddings(
                current_vectors,
                prefix_behaviors.unsqueeze(0).to(device),
                prefix_deltas.unsqueeze(0).to(device),
                lengths,
            )
        )
        old_cache = fp16_storage_fp32_consumption(old_cache)
        recompute_cache = fp16_storage_fp32_consumption(recompute_cache)
        kv_error = cache_relative_error(old_cache, recompute_cache)
        suffix_items, suffix_behaviors, suffix_deltas, targets, labels = (
            evaluation_suffix(corpus, record, binding.edge)
        )
        suffix_lengths = torch.tensor(
            [len(suffix_items)], dtype=torch.int64, device=device
        )
        suffix_vectors = current_embedding(
            suffix_items.unsqueeze(0).to(device), suffix_lengths
        )
        reuse_hidden, _ = (
            current_dense.core.forward_with_cache_from_item_embeddings(
                old_cache,
                suffix_vectors,
                suffix_behaviors.unsqueeze(0).to(device),
                suffix_deltas.unsqueeze(0).to(device),
            )
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
        steps = math.ceil(int(maximum_targets.item()) / target_chunk)
        for step in range(steps):
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
                item_chunk=binding.full_catalog_item_chunk,
            )
            nll_values.append(nll)
            rank_values.append(ranks)
        if nll_values:
            record_nll = torch.cat(nll_values, dim=1)
            record_ranks = torch.cat(rank_values, dim=1)
        else:
            record_nll = torch.empty((2, 0), device=device)
            record_ranks = torch.empty(
                (2, 0), dtype=torch.int64, device=device
            )
        metrics = summarize_full_catalog_record(record_nll, record_ranks)
        metrics.update(
            {
                "record": record,
                "user_id": int(corpus.arrays["record_user_ids"][record]),
                "history_length": prefix_length,
                "append_length": len(suffix_items),
                "prefix_cache_relative_error": kv_error,
            }
        )
        local_results.append(metrics)
        del (
            current_vectors,
            old_cache,
            recompute_cache,
            suffix_vectors,
            reuse_hidden,
            recompute_hidden,
            record_nll,
            record_ranks,
        )
        if (
            ordinal == 0
            or (ordinal + 1) % progress_every == 0
            or ordinal + 1 == len(records)
        ):
            print(
                f"phase=qk_theta{binding.target_version}_{role}_full_catalog "
                f"rank={rank} record={ordinal + 1}/{len(records)}",
                flush=True,
            )
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, local_results)
    if rank != 0:
        return {}
    combined = [value for rank_values in gathered for value in rank_values]
    combined.sort(key=lambda value: int(value["record"]))
    summary = paired_full_catalog_summary(
        combined,
        bootstrap_samples=binding.bootstrap_samples,
        bootstrap_seed=binding.bootstrap_seed + binding.edge * 1_000_033,
    )
    errors = np.asarray(
        [value["prefix_cache_relative_error"] for value in combined],
        dtype=np.float64,
    )
    histories = np.asarray(
        [value["history_length"] for value in combined], dtype=np.int64
    )
    appends = np.asarray(
        [value["append_length"] for value in combined], dtype=np.int64
    )
    summary.update(
        {
            "role": role,
            "num_prediction_items": spec.num_prediction_items,
            "endpoint": "FP16 cache storage followed by FP32 consumption",
            "scorer": "exact owner-sharded full-catalog projected scorer",
            "ranking_tie_rule": "pessimistic rank using score >= positive score",
            "natural_unpadded_het": True,
            "prefix_cache_relative_error": {
                "mean": float(errors.mean()),
                "median": float(np.median(errors)),
                "p95": float(np.quantile(errors, 0.95)),
            },
            "history_length": {
                "minimum": int(histories.min()),
                "median": float(np.median(histories)),
                "p95": float(np.quantile(histories, 0.95)),
                "maximum": int(histories.max()),
                "unique": int(len(np.unique(histories))),
            },
            "append_length": {
                "minimum": int(appends.min()),
                "median": float(np.median(appends)),
                "p95": float(np.quantile(appends, 0.95)),
                "maximum": int(appends.max()),
                "unique": int(len(np.unique(appends))),
            },
            "runtime_seconds": time.perf_counter() - started,
        }
    )
    return summary


def run_full_catalog_tuning_edge(
    config_path: Path,
    binding: QKStreamFullCatalogBinding,
) -> dict[str, object] | None:
    document = json.loads(config_path.read_text())
    validate_full_catalog_binding_document(document, binding)
    rank, world_size, local_rank, device = _runtime()
    started = time.perf_counter()
    try:
        _seed_everything(binding.training_seed)
        torch.set_float32_matmul_precision("high")
        output_path = Path(document["outputs"]["result"])
        final_root = Path(document["outputs"]["checkpoint_root"])
        work_root = Path(document["outputs"]["work_checkpoint_root"])
        final_checkpoint = final_root / f"theta_{binding.target_version}"
        work_checkpoint = work_root / f"theta_{binding.target_version}"
        if output_path.exists() or final_checkpoint.exists():
            raise FileExistsError("QK full-catalog target already exists")
        if work_checkpoint.exists() and not (
            work_checkpoint / "manifest.json"
        ).is_file():
            _cleanup_work(work_root, rank)
        resumed_candidate = (work_checkpoint / "manifest.json").is_file()
        source = document["source_checkpoint"]
        source_root = Path(source["root"])
        source_manifest_path = (
            source_root
            / f"theta_{binding.source_version}"
            / "manifest.json"
        )
        if file_sha256(source_manifest_path) != source["manifest_sha256"]:
            raise ValueError("QK full-catalog source checkpoint hash differs")
        data = document["data"]
        data_config = Path(data["config"])
        if file_sha256(data_config) != data["config_sha256"]:
            raise ValueError("QK full-catalog stream data config hash differs")
        corpus = load_corpus(data["corpus"])
        summary_path = Path(data["summary"])
        summary = json.loads(summary_path.read_text())
        if (
            summary.get("status") != "pass"
            or summary.get("content_sha256") != corpus.content_sha256
            or summary.get("artifact", {}).get("sha256")
            != corpus.file_sha256
        ):
            raise ValueError("QK full-catalog stream data summary differs")
        spec, source_dense, source_embedding, source_tracker, source_manifest = (
            _load_model(
                source_root,
                binding.source_version,
                rank=rank,
                world_size=world_size,
                device=device,
            )
        )
        if source_manifest != json.loads(source_manifest_path.read_text()):
            raise ValueError("QK full-catalog source changed during load")
        old_dense_state = _dense_state(source_dense)
        snapshots = snapshot_source_prefixes(
            corpus,
            source_embedding,
            ("fit_tuning",),
            binding.edge,
            rank,
            world_size,
            device,
            int(document["execution"]["snapshot_batch_size_per_rank"]),
        )
        if resumed_candidate:
            training = _load_candidate_training(
                work_root, binding.target_version
            )
            del source_dense, source_embedding, source_tracker
            gc.collect()
            torch.cuda.empty_cache()
            spec, dense, embedding, tracker, candidate_manifest = _load_model(
                work_root,
                binding.target_version,
                rank=rank,
                world_size=world_size,
                device=device,
            )
        else:
            dense = source_dense
            embedding = source_embedding
            tracker = source_tracker
            training, optimizers = _train_edge(
                document,
                binding,
                corpus,
                spec,
                dense,
                embedding,
                tracker,
                rank=rank,
                world_size=world_size,
                device=device,
            )
            if not training["checkpoint_admission_passed"]:
                raise RuntimeError(
                    "QK full-catalog candidate is not evaluation-ready: "
                    + json.dumps(
                        training["evaluation_readiness"], sort_keys=True
                    )
                )
            candidate_manifest = _save_candidate(
                document,
                config_path,
                binding,
                corpus,
                spec,
                dense,
                embedding,
                tracker,
                optimizers,
                training,
                rank=rank,
            )
            del optimizers
            gc.collect()
            torch.cuda.empty_cache()
        source_dense_for_quality = ExternalEmbeddingHSTU(
            spec.hstu_config()
        ).to(device)
        source_dense_for_quality.load_state_dict(old_dense_state)
        source_dense_for_quality.eval()
        dense.eval()
        embedding.eval()
        tuning = _evaluate_full_catalog_role(
            document,
            binding,
            "fit_tuning",
            corpus,
            spec,
            dense,
            embedding,
            source_dense_for_quality,
            snapshots["fit_tuning"],
            rank=rank,
            world_size=world_size,
            device=device,
        )
        hbm = {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        result = {
            "protocol": document["protocol"],
            "status": "complete_tuning_measurement",
            "scientific_result": False,
            "formal_result": False,
            "dataset": "tenrec-qk",
            "config": {
                "path": str(config_path),
                "sha256": file_sha256(config_path),
            },
            "edge": asdict(binding),
            "source_checkpoint": {
                **source,
                "training_state_sha256": file_sha256(
                    source_root
                    / f"theta_{binding.source_version}"
                    / "training_state.json"
                ),
            },
            "data": {
                "config": data,
                "corpus_sha256": corpus.file_sha256,
                "corpus_content_sha256": corpus.content_sha256,
                "summary_sha256": file_sha256(summary_path),
            },
            "training": training,
            "quality": {
                "same_current_model": binding.target_version,
                "methods": ["reuse", "recompute"],
                "reuse_source_version": binding.source_version,
                "recompute_version": binding.target_version,
                "tuning": tuning,
                "qualification": None,
                "qualification_consumed": False,
                "final_consumed": False,
                "labels_used_for_routing": False,
                "decision_boundary": "manual_after_tuning",
            },
            "checkpoint": {
                "committed": False,
                "provisional_retained": True,
                "path": str(work_checkpoint),
                "version": binding.target_version,
                "manifest_sha256": file_sha256(
                    work_checkpoint / "manifest.json"
                ),
                "optimizer_resume_retained": True,
                "candidate_manifest": candidate_manifest,
            },
            "execution": {
                "world_size": world_size,
                "local_rank": local_rank,
                "visible_devices": os.environ.get(
                    "CUDA_VISIBLE_DEVICES", ""
                ),
                "hbm": hbm,
                "runtime_seconds": time.perf_counter() - started,
                "resumed_candidate": resumed_candidate,
            },
        }
        if rank == 0:
            _atomic_json(output_path, result)
        return result if rank == 0 else None
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
