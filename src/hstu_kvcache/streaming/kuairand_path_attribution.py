from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .distributed import close_distributed_runtime, init_distributed_runtime
from .kuairand_root_cause import (
    METRICS,
    _atomic_json,
    _empty_cache,
    _evaluation_sequence,
    _full_catalog_pair,
    _hash_int_array,
    _load_checkpoint,
    _metric_comparison,
    _metric_sums,
    _run_suffix,
    _seed_everything,
    _selected_users,
    _stored_cache,
    _valid_checkpoint,
    load_plan,
    make_model,
    validate_document,
)
from .qk_stream_version import file_sha256

PROTOCOL = "evokv_root_cause_kuairand_path_attribution_v0"


def validate_attribution_document(document: dict[str, object]) -> dict[str, object]:
    parent = document.get("parent")
    quality = document.get("quality")
    execution = document.get("execution")
    methods = document.get("methods")
    outputs = document.get("outputs")
    checkpoints = document.get("checkpoints")
    current_methods = () if not isinstance(methods, dict) else methods.get("current_scorer")
    previous_methods = () if not isinstance(methods, dict) else methods.get("previous_scorer")
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scope") != "development_attribution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or not all(
            isinstance(value, dict)
            for value in (parent, quality, execution, methods, outputs)
        )
        or not isinstance(checkpoints, list)
        or [value.get("version") for value in checkpoints] != [0, 1, 2]
        or current_methods
        != [
            "current_fresh",
            "current_fresh_duplicate",
            "stale_previous",
            "no_prefix",
            "previous_hidden_current_scorer",
            "embedding_update_only",
        ]
        or previous_methods
        != [
            "previous_fresh",
            "previous_fresh_duplicate",
            "current_hidden_previous_scorer",
            "dense_update_only",
        ]
        or quality.get("exposure_position_boundaries") != [0, 4, 16, 64, 256]
        or quality.get("positive_ordinal_boundaries") != [0, 1, 4, 16, 64]
        or int(quality.get("record_limit_per_rank", 0)) != 128
        or int(quality.get("target_chunk", 0)) < 1
        or int(quality.get("full_catalog_item_chunk", 0)) < 1
        or int(quality.get("suffix_chunk", 0)) < 1
        or int(quality.get("bootstrap_samples", 0)) < 1
        or execution.get("world_size") != 2
        or execution.get("cuda_visible_devices") != "0,1"
    ):
        raise ValueError("KuaiRand path-attribution config differs")
    parent_hashes = {
        "config": "config_sha256",
        "training_result": "training_sha256",
        "evaluation_result": "evaluation_sha256",
        "runner": "runner_sha256",
    }
    for name, hash_name in parent_hashes.items():
        if file_sha256(Path(parent[name])) != parent[hash_name]:
            raise ValueError(f"KuaiRand path-attribution parent {name} hash differs")
    parent_document = json.loads(Path(parent["config"]).read_text())
    validate_document(parent_document)
    for checkpoint in checkpoints:
        path = (
            Path(parent["checkpoint_root"])
            / f"theta_{checkpoint['version']}"
            / "manifest.json"
        )
        if file_sha256(path) != checkpoint["manifest_sha256"]:
            raise ValueError("KuaiRand path-attribution checkpoint hash differs")
    return parent_document


def _hybrid_model(parent_document, plan, previous, current, predicate, device):
    model = make_model(parent_document, plan, device)
    previous_state = previous.state_dict()
    current_state = current.state_dict()
    model.load_state_dict(
        {
            name: current_state[name] if predicate(name) else value
            for name, value in previous_state.items()
        }
    )
    model.eval()
    return model


def _strata(values: np.ndarray, boundaries: list[int]) -> dict[str, np.ndarray]:
    output = {}
    for index, lower in enumerate(boundaries):
        upper = boundaries[index + 1] if index + 1 < len(boundaries) else None
        name = f"{lower}_{upper}" if upper is not None else f"{lower}_plus"
        output[name] = values >= lower if upper is None else (values >= lower) & (values < upper)
    return output


def _compact_records(records, stratum: str | None = None, bin_name: str | None = None):
    if stratum is None:
        return records
    output = []
    for record in records:
        value = record["strata"][stratum][bin_name]
        if value["targets"]:
            output.append(
                {
                    "user_id": record["user_id"],
                    "targets": value["targets"],
                    "metric_sums": value["metric_sums"],
                }
            )
    return output


def _endpoint(records, methods):
    denominator = float(sum(value["targets"] for value in records))
    return {
        method: {
            metric: float(
                sum(value["metric_sums"][method][metric] for value in records)
                / denominator
            )
            for metric in METRICS
        }
        for method in methods
    }


def _summary(records, methods, bootstrap_samples, bootstrap_seed):
    endpoints = _endpoint(records, methods)
    comparisons = {
        "update_value": _metric_comparison(
            records,
            "previous_fresh",
            "current_fresh",
            lambda value: value["metric_sums"]["previous_fresh"],
            lambda value: value["metric_sums"]["current_fresh"],
            bootstrap_samples,
            bootstrap_seed,
        ),
        "stale_tax": _metric_comparison(
            records,
            "stale_previous",
            "current_fresh",
            lambda value: value["metric_sums"]["stale_previous"],
            lambda value: value["metric_sums"]["current_fresh"],
            bootstrap_samples,
            bootstrap_seed + 101,
        ),
        "prior_prefix_value": _metric_comparison(
            records,
            "no_prefix",
            "current_fresh",
            lambda value: value["metric_sums"]["no_prefix"],
            lambda value: value["metric_sums"]["current_fresh"],
            bootstrap_samples,
            bootstrap_seed + 202,
        ),
    }
    route_methods = (
        "previous_hidden_current_scorer",
        "embedding_update_only",
        "current_hidden_previous_scorer",
        "dense_update_only",
    )
    routes = {}
    for index, method in enumerate(route_methods):
        routes[method] = _metric_comparison(
            records,
            "previous_fresh",
            method,
            lambda value: value["metric_sums"]["previous_fresh"],
            lambda value, selected=method: value["metric_sums"][selected],
            bootstrap_samples,
            bootstrap_seed + 1000 + index * 101,
        )
        for metric in METRICS:
            full = comparisons["update_value"][metric][
                "current_fresh_advantage_absolute"
            ]
            route = routes[method][metric][f"{method}_advantage_absolute"]
            routes[method][metric]["fraction_of_full_update_percent"] = (
                100.0 * route / full if abs(full) > 1e-12 else None
            )
    return {
        "users_with_targets": len(records),
        "positive_targets": int(sum(value["targets"] for value in records)),
        "endpoints": endpoints,
        "comparisons": comparisons,
        "parameter_routes": routes,
    }


def _aggregate(records, methods, quality):
    records = sorted(records, key=lambda value: int(value["user_id"]))
    full = _summary(
        records,
        methods,
        int(quality["bootstrap_samples"]),
        int(quality["bootstrap_seed"]),
    )
    stratified = {}
    for stratum_index, stratum in enumerate(("exposure_position", "positive_ordinal")):
        names = list(records[0]["strata"][stratum])
        stratified[stratum] = {}
        for bin_index, name in enumerate(names):
            selected = _compact_records(records, stratum, name)
            stratified[stratum][name] = _summary(
                selected,
                methods,
                int(quality["bootstrap_samples"]),
                int(quality["bootstrap_seed"])
                + (stratum_index + 1) * 1000003
                + bin_index * 1009,
            )
    duplicate_nll = max(
        value["sanity"]["current_duplicate_nll_maximum_absolute_error"]
        for value in records
    )
    duplicate_rank = all(
        value["sanity"]["current_duplicate_ranks_equal"] for value in records
    )
    previous_duplicate_nll = max(
        value["sanity"]["previous_duplicate_nll_maximum_absolute_error"]
        for value in records
    )
    previous_duplicate_rank = all(
        value["sanity"]["previous_duplicate_ranks_equal"] for value in records
    )
    return {
        "full_horizon": full,
        "stratified": stratified,
        "sanity": {
            "current_duplicate_nll_maximum_absolute_error": duplicate_nll,
            "current_duplicate_ranks_equal": duplicate_rank,
            "previous_duplicate_nll_maximum_absolute_error": previous_duplicate_nll,
            "previous_duplicate_ranks_equal": previous_duplicate_rank,
            "implementation_passed": bool(
                duplicate_nll <= 1e-7
                and duplicate_rank
                and previous_duplicate_nll <= 1e-7
                and previous_duplicate_rank
            ),
        },
        "records_detail": records,
    }


@torch.no_grad()
def _evaluate_edge(
    document,
    parent_document,
    parent_result,
    plan,
    previous,
    current,
    version,
    update_date,
    eval_date,
    runtime,
):
    quality = document["quality"]
    selected, eligible = _selected_users(
        plan,
        update_date,
        eval_date,
        int(quality["record_limit_per_rank"]) * runtime.world_size,
        int(quality["sampling_seed"]) + version * 1009,
    )
    selected_hash = _hash_int_array(np.asarray(selected, dtype=np.int64))
    expected = parent_result["edges"][version - 1]["selected_user_ids_sha256"]
    if selected_hash != expected:
        raise RuntimeError("KuaiRand path-attribution selected users differ")
    local_users = selected[runtime.rank::runtime.world_size]
    embedding_only = _hybrid_model(
        parent_document,
        plan,
        previous,
        current,
        lambda name: name == "item_emb.weight",
        runtime.device,
    )
    dense_only = _hybrid_model(
        parent_document,
        plan,
        previous,
        current,
        lambda name: name != "item_emb.weight",
        runtime.device,
    )
    buffers = []
    suffix_chunk = int(quality["suffix_chunk"])
    for ordinal, user in enumerate(local_users):
        sequence = _evaluation_sequence(plan, user, eval_date)
        prefix = sequence["prefix"]
        current_cache = _stored_cache(current, prefix, runtime.device)
        current_hidden = _run_suffix(
            current,
            current_cache,
            sequence["suffix"],
            sequence["labels"],
            suffix_chunk,
            runtime.device,
        )
        current_duplicate = _run_suffix(
            current,
            _stored_cache(current, prefix, runtime.device),
            sequence["suffix"],
            sequence["labels"],
            suffix_chunk,
            runtime.device,
        )
        stale_hidden = _run_suffix(
            current,
            _stored_cache(previous, prefix, runtime.device),
            sequence["suffix"],
            sequence["labels"],
            suffix_chunk,
            runtime.device,
        )
        no_prefix_hidden = _run_suffix(
            current,
            _empty_cache(current_cache),
            sequence["suffix"],
            sequence["labels"],
            suffix_chunk,
            runtime.device,
        )
        previous_hidden = _run_suffix(
            previous,
            _stored_cache(previous, prefix, runtime.device),
            sequence["suffix"],
            sequence["labels"],
            suffix_chunk,
            runtime.device,
        )
        previous_duplicate = _run_suffix(
            previous,
            _stored_cache(previous, prefix, runtime.device),
            sequence["suffix"],
            sequence["labels"],
            suffix_chunk,
            runtime.device,
        )
        embedding_hidden = _run_suffix(
            embedding_only,
            _stored_cache(embedding_only, prefix, runtime.device),
            sequence["suffix"],
            sequence["labels"],
            suffix_chunk,
            runtime.device,
        )
        dense_hidden = _run_suffix(
            dense_only,
            _stored_cache(dense_only, prefix, runtime.device),
            sequence["suffix"],
            sequence["labels"],
            suffix_chunk,
            runtime.device,
        )
        targets = torch.from_numpy(sequence["targets"][sequence["labels"]]).long()
        positions = np.flatnonzero(sequence["labels"]).astype(np.int64)
        buffers.append(
            {
                "user_id": user,
                "targets": targets,
                "exposure_positions": positions,
                "positive_ordinals": np.arange(len(targets), dtype=np.int64),
                "hidden": {
                    "current_fresh": current_hidden,
                    "current_fresh_duplicate": current_duplicate,
                    "stale_previous": stale_hidden,
                    "no_prefix": no_prefix_hidden,
                    "previous_hidden_current_scorer": previous_hidden,
                    "embedding_update_only": embedding_hidden,
                    "previous_fresh": previous_hidden,
                    "previous_fresh_duplicate": previous_duplicate,
                    "current_hidden_previous_scorer": current_hidden,
                    "dense_update_only": dense_hidden,
                },
            }
        )
        print(
            f"phase=kuairand_path_edge{version}_hidden rank={runtime.rank} "
            f"user={ordinal + 1}/{len(local_users)} targets={len(targets)}",
            flush=True,
        )
    all_targets = torch.cat([value["targets"] for value in buffers])
    current_methods = list(document["methods"]["current_scorer"])
    previous_methods = list(document["methods"]["previous_scorer"])
    scores = {}
    for scorer, names in ((current, current_methods), (previous, previous_methods)):
        hidden = {
            name: torch.cat([value["hidden"][name] for value in buffers])
            for name in names
        }
        for pair in [names[index:index + 2] for index in range(0, len(names), 2)]:
            nll, ranks = _full_catalog_pair(
                scorer,
                torch.stack([hidden[name] for name in pair]),
                all_targets,
                int(quality["target_chunk"]),
                int(quality["full_catalog_item_chunk"]),
                runtime.device,
                f"kuairand_path_edge{version}_{pair[0]}_{pair[1]}_rank{runtime.rank}",
            )
            for row, name in enumerate(pair):
                scores[name] = (nll[row], ranks[row])
    methods = [*current_methods, *previous_methods]
    records = []
    start = 0
    for value in buffers:
        stop = start + len(value["targets"])
        metric_sums = {
            method: _metric_sums(
                scores[method][0][start:stop],
                scores[method][1][start:stop],
            )
            for method in methods
        }
        strata = {}
        for stratum, values, boundaries in (
            (
                "exposure_position",
                value["exposure_positions"],
                quality["exposure_position_boundaries"],
            ),
            (
                "positive_ordinal",
                value["positive_ordinals"],
                quality["positive_ordinal_boundaries"],
            ),
        ):
            strata[stratum] = {}
            for name, mask in _strata(values, boundaries).items():
                indices = torch.from_numpy(np.flatnonzero(mask)).long()
                strata[stratum][name] = {
                    "targets": len(indices),
                    "metric_sums": {
                        method: _metric_sums(
                            scores[method][0][start:stop].index_select(0, indices),
                            scores[method][1][start:stop].index_select(0, indices),
                        )
                        for method in methods
                    },
                }
        records.append(
            {
                "user_id": value["user_id"],
                "targets": len(value["targets"]),
                "metric_sums": metric_sums,
                "strata": strata,
                "sanity": {
                    "current_duplicate_nll_maximum_absolute_error": float(
                        (
                            scores["current_fresh"][0][start:stop]
                            - scores["current_fresh_duplicate"][0][start:stop]
                        )
                        .abs()
                        .max()
                        .item()
                    ),
                    "current_duplicate_ranks_equal": bool(
                        torch.equal(
                            scores["current_fresh"][1][start:stop],
                            scores["current_fresh_duplicate"][1][start:stop],
                        )
                    ),
                    "previous_duplicate_nll_maximum_absolute_error": float(
                        (
                            scores["previous_fresh"][0][start:stop]
                            - scores["previous_fresh_duplicate"][0][start:stop]
                        )
                        .abs()
                        .max()
                        .item()
                    ),
                    "previous_duplicate_ranks_equal": bool(
                        torch.equal(
                            scores["previous_fresh"][1][start:stop],
                            scores["previous_fresh_duplicate"][1][start:stop],
                        )
                    ),
                },
            }
        )
        start = stop
    gathered: list[object] | None = [None] * runtime.world_size if runtime.is_primary else None
    dist.gather_object(records, gathered, dst=0)
    del embedding_only, dense_only, buffers, scores
    gc.collect()
    torch.cuda.empty_cache()
    if not runtime.is_primary:
        return None
    combined = [record for shard in gathered for record in shard]
    aggregate = _aggregate(combined, methods, quality)
    aggregate.update(
        {
            "edge": version,
            "update_date": update_date,
            "evaluation_date": eval_date,
            "eligible_same_user_population": eligible,
            "selected_user_ids_sha256": selected_hash,
        }
    )
    return aggregate


def run_kuairand_path_attribution(config_path: Path) -> dict[str, object] | None:
    document = json.loads(config_path.read_text())
    parent_document = validate_attribution_document(document)
    runtime = init_distributed_runtime("cuda:0")
    if runtime.world_size != int(document["execution"]["world_size"]):
        close_distributed_runtime(runtime)
        raise ValueError("KuaiRand path-attribution world size differs")
    started = time.perf_counter()
    try:
        output = Path(document["outputs"]["result"])
        if output.exists():
            result = json.loads(output.read_text())
            if result.get("status") != "complete_development_attribution":
                raise FileExistsError("KuaiRand path-attribution result is incomplete")
            return result if runtime.is_primary else None
        _seed_everything(int(document["quality"]["sampling_seed"]) + runtime.rank)
        torch.set_float32_matmul_precision("high")
        plan, data_metadata = load_plan(parent_document)
        parent = document["parent"]
        parent_result = json.loads(Path(parent["evaluation_result"]).read_text())
        root = Path(parent["checkpoint_root"])
        for version in range(3):
            if not _valid_checkpoint(
                root,
                version,
                parent["config_sha256"],
                data_metadata,
            ):
                raise ValueError(f"KuaiRand path-attribution theta{version} differs")
        models = []
        for version in range(3):
            model = make_model(parent_document, plan, runtime.device)
            _load_checkpoint(model, root, version)
            model.eval()
            models.append(model)
        plan.init_base()
        dates = plan.base_dates + plan.stream_dates
        edges = []
        for version, (update_index, eval_index) in enumerate(
            zip(
                parent_document["schedule"]["update_date_indices"],
                parent_document["schedule"]["evaluation_date_indices"],
                strict=True,
            ),
            start=1,
        ):
            update_date = dates[int(update_index)]
            eval_date = dates[int(eval_index)]
            plan.ingest_day(update_date)
            edge = _evaluate_edge(
                document,
                parent_document,
                parent_result,
                plan,
                models[version - 1],
                models[version],
                version,
                update_date,
                eval_date,
                runtime,
            )
            if runtime.is_primary:
                edges.append(edge)
        if not runtime.is_primary:
            dist.barrier()
            return None
        result = {
            "protocol": PROTOCOL,
            "status": "complete_development_attribution",
            "scope": document["scope"],
            "scientific_result": False,
            "formal_result": False,
            "round_id": document["round_id"],
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "programs": {
                "runner": {
                    "path": "src/hstu_kvcache/streaming/kuairand_path_attribution.py",
                    "sha256": file_sha256(Path(__file__)),
                },
                "parent_runner": {
                    "path": parent["runner"],
                    "sha256": parent["runner_sha256"],
                },
            },
            "parent": parent,
            "data": data_metadata,
            "methods": document["methods"],
            "quality": document["quality"],
            "edges": edges,
            "sanity": {
                "implementation_passed": all(
                    edge["sanity"]["implementation_passed"] for edge in edges
                )
            },
            "execution": {
                "world_size": runtime.world_size,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "runtime_seconds": time.perf_counter() - started,
                "qualification_consumed": False,
                "final_consumed": False,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(runtime.device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(runtime.device),
            },
        }
        _atomic_json(output, result)
        dist.barrier()
        return result
    finally:
        close_distributed_runtime(runtime)
