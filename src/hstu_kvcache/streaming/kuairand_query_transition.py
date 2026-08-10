from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..models import HSTU, HSTUConfig
from .ml1m_opportunity import METRICS, _comparison

PROTOCOL = "evokv_kuairand_prediction_query_transition_v0"
METHODS = ("previous_fresh", "recompute", "reuse", "no_prefix")
PREDICTION_QUERY_OBJECTIVE = "normalized_prediction_query_sampled_softmax"
TRUE_NEXT_ITEM_OBJECTIVE = "normalized_true_next_item_sampled_softmax"


def _true_next_item(document: dict[str, Any]) -> bool:
    return document["training"]["objective"] == TRUE_NEXT_ITEM_OBJECTIVE


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    data = document.get("data")
    model = document.get("model")
    training = document.get("training")
    evaluation = document.get("evaluation")
    execution = document.get("execution")
    outputs = document.get("outputs")
    if (
        document.get("protocol") != PROTOCOL
        or document.get("status") != "ready_for_autonomous_execution"
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or not all(
            isinstance(value, dict)
            for value in (data, model, training, evaluation, execution, outputs)
        )
        or data.get("catalog_fit") != "all_base_exposures"
        or data.get("sequence_semantics") != "engaged_in_catalog_actions"
        or data.get("update_date") != "20220422"
        or data.get("evaluation_date") != "20220423"
        or int(data.get("base_training_days", 0)) < 1
        or int(data.get("base_targets_per_user_day", 0)) < 1
        or int(data.get("update_targets_per_user", 0)) < 1
        or not 1 <= int(data.get("training_target_horizon", 1)) <= 8
        or int(data.get("minimum_prefix_length", 0)) < 2
        or int(model.get("max_seq_len", 0)) < 8
        or int(model.get("hidden_size", 0)) < 8
        or int(model.get("num_layers", 0)) < 1
        or int(model.get("num_heads", 0)) < 1
        or int(model.get("hidden_size", 0)) % int(model.get("num_heads", 0)) != 0
        or model.get("field_mode", "video_only") not in ("video_only", "video_author")
        or model.get("query_mode", "learned_token") not in (
            "learned_token",
            "history_only_zero",
            "latest_item_query",
            "last_history_item",
        )
        or model.get("gating", "silu_gate") not in (
            "none",
            "silu_gate",
            "glu",
            "ffn",
        )
        or model.get("block_variant", "legacy") not in (
            "legacy",
            "hstu_reference",
        )
        or (
            model.get("block_variant", "legacy") == "hstu_reference"
            and (
                model.get("gating", "silu_gate") != "silu_gate"
                or model.get("activation") != "silu"
            )
        )
        or model.get("causal_diagonal", "inclusive") not in (
            "inclusive",
            "exclusive",
        )
        or training.get("objective")
        not in (PREDICTION_QUERY_OBJECTIVE, TRUE_NEXT_ITEM_OBJECTIVE)
        or (
            training.get("objective") == TRUE_NEXT_ITEM_OBJECTIVE
            and model.get("query_mode") != "last_history_item"
        )
        or (
            training.get("objective") == PREDICTION_QUERY_OBJECTIVE
            and model.get("query_mode") == "last_history_item"
        )
        or not isinstance(training.get("seeds"), list)
        or not training["seeds"]
        or int(training.get("base_epochs", 0)) < 1
        or int(training.get("update_epochs", 0)) < 1
        or float(training.get("base_embedding_lr", training.get("base_lr", 0)))
        <= 0
        or float(
            training.get("update_embedding_lr", training.get("update_lr", 0))
        )
        <= 0
        or int(training.get("negative_samples", 0)) < 1
        or not 0 < float(training.get("publish_alpha", 1.0)) <= 1.0
        or int(evaluation.get("candidate_count", 0)) not in (50, 100)
        or evaluation.get("candidate_source")
        not in (
            "base_exposure_popularity_unseen",
            "base_exposure_frequency_matched_unseen",
        )
        or training.get("negative_source")
        not in ("base_exposure_popularity_pool", "base_exposure_frequency_matched")
        or int(evaluation.get("bootstrap_samples", 0)) < 100
        or execution.get("cuda_visible_devices") != "0"
    ):
        raise ValueError("KuaiRand prediction-query config differs")
    for field in ("base_log", "stream_log"):
        source = data.get(field)
        if not isinstance(source, dict):
            raise ValueError("KuaiRand source binding is absent")
        source_path = Path(source.get("path", ""))
        if not source_path.is_file() or file_sha256(source_path) != source.get("sha256"):
            raise ValueError(f"KuaiRand source binding differs: {field}")
    parent_theta0 = training.get("parent_theta0")
    if parent_theta0 is not None:
        parent_path = (
            Path(parent_theta0.get("path", ""))
            if isinstance(parent_theta0, dict)
            else Path()
        )
        if (
            not isinstance(parent_theta0, dict)
            or not parent_path.is_file()
            or file_sha256(parent_path) != parent_theta0.get("sha256")
        ):
            raise ValueError("KuaiRand parent theta0 binding differs")
    if model.get("field_mode", "video_only") == "video_author":
        source = data.get("video_features")
        if not isinstance(source, dict):
            raise ValueError("KuaiRand video feature binding is absent")
        source_path = Path(source.get("path", ""))
        if not source_path.is_file() or file_sha256(source_path) != source.get("sha256"):
            raise ValueError("KuaiRand video feature binding differs")
    return document


def _engaged(frame: pd.DataFrame) -> np.ndarray:
    return frame[
        ["is_click", "is_like", "is_follow", "is_comment", "is_forward", "long_view"]
    ].to_numpy(dtype=np.bool_).any(axis=1)


def _positions(values: np.ndarray, maximum: int) -> np.ndarray:
    if len(values) <= maximum:
        return values
    indices = np.linspace(0, len(values) - 1, num=maximum, dtype=np.int64)
    return values[np.unique(indices)]


def _user_order(users: list[int], seed: int) -> list[int]:
    return sorted(
        users,
        key=lambda user: hashlib.sha256(f"{seed}:{user}".encode()).digest(),
    )


def _hash_ints(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def _future_targets(
    items: np.ndarray,
    dates: np.ndarray,
    eligible: np.ndarray,
    position: int,
    horizon: int,
) -> int | np.ndarray | None:
    date = dates[position]
    targets = []
    for index in range(position, len(items)):
        if dates[index] != date:
            break
        if eligible[index]:
            targets.append(int(items[index]))
            if len(targets) == horizon:
                break
    if len(targets) != horizon:
        return None
    if horizon == 1:
        return targets[0]
    return np.asarray(targets, dtype=np.int64)


def _frequency_matched_candidates(
    target: int,
    excluded: set[int],
    candidate_pool: np.ndarray,
    rank_by_item: np.ndarray,
    count: int,
    seed: int,
) -> list[int]:
    target_rank = int(rank_by_item[target]) + 1
    if target_rank < 1:
        raise ValueError("KuaiRand frequency-matched target is outside candidate pool")
    exponent = int(math.floor(math.log2(target_rank)))
    lower = 1 << exponent
    upper = min(len(candidate_pool) + 1, max(lower * 2, lower + 512))
    lower = max(1, min(lower, len(candidate_pool) + 1 - 512))
    if upper - lower <= count + len(excluded):
        lower = max(1, target_rank - 2048)
        upper = min(len(candidate_pool) + 1, lower + 4096)
        lower = max(1, upper - 4096)
    generator = np.random.default_rng(seed)
    output = []
    selected = set()
    attempts = 0
    while len(output) < count:
        value = int(candidate_pool[int(generator.integers(lower, upper)) - 1])
        attempts += 1
        if value not in excluded and value not in selected:
            output.append(value)
            selected.add(value)
        if attempts > 100000:
            raise RuntimeError("KuaiRand frequency-matched candidates are incomplete")
    return output


def _load_frame(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=[
            "user_id",
            "video_id",
            "time_ms",
            "date",
            "is_click",
            "is_like",
            "is_follow",
            "is_comment",
            "is_forward",
            "long_view",
        ],
    )


def build_workload(document: dict[str, Any]) -> dict[str, Any]:
    data = document["data"]
    base = _load_frame(data["base_log"]["path"])
    stream = _load_frame(data["stream_log"]["path"])
    base["date"] = base["date"].astype(str)
    stream["date"] = stream["date"].astype(str)
    popularity = base["video_id"].value_counts(sort=True)
    raw_video_ids = popularity.index.to_numpy(dtype=np.int64)
    item_index = pd.Series(
        np.arange(1, len(raw_video_ids) + 1, dtype=np.int64),
        index=raw_video_ids,
    )
    base["item_id"] = base["video_id"].map(item_index).astype(np.int64)
    stream["item_id"] = stream["video_id"].map(item_index)
    base_engaged = base.loc[
        _engaged(base), ["user_id", "item_id", "time_ms", "date"]
    ].copy()
    if data.get("target_catalog_fit", "all_base_exposures") == "base_engaged_items":
        target_popularity = base.loc[_engaged(base), "video_id"].value_counts(sort=True)
        popular_ids = item_index.loc[target_popularity.index].to_numpy(dtype=np.int64)
    elif data.get("target_catalog_fit", "all_base_exposures") == "all_base_exposures":
        popular_ids = np.arange(1, len(raw_video_ids) + 1, dtype=np.int64)
    else:
        raise ValueError("KuaiRand target catalog fit differs")
    rank_by_item = np.full(len(raw_video_ids) + 1, -1, dtype=np.int64)
    rank_by_item[popular_ids] = np.arange(len(popular_ids), dtype=np.int64)
    if document["model"].get("field_mode", "video_only") == "video_author":
        features = pd.read_csv(data["video_features"]["path"], usecols=["video_id", "author_id"])
        features = features.drop_duplicates("video_id", keep="first").set_index("video_id")
        raw_authors = features["author_id"].reindex(raw_video_ids)
        if raw_authors.isna().any():
            raise RuntimeError("KuaiRand base video is missing an author")
        raw_author_ids, author_inverse = np.unique(
            raw_authors.to_numpy(dtype=np.int64), return_inverse=True
        )
        author_by_item = np.zeros(len(raw_video_ids) + 1, dtype=np.int64)
        author_by_item[1:] = len(raw_video_ids) + 1 + author_inverse
        embedding_rows = len(raw_video_ids) + len(raw_author_ids)
        author_ids_sha256 = _hash_ints(raw_author_ids)
    else:
        author_by_item = np.zeros(len(raw_video_ids) + 1, dtype=np.int64)
        embedding_rows = len(raw_video_ids)
        author_ids_sha256 = None
    stream_known = stream["item_id"].notna().to_numpy()
    stream_engaged = stream.loc[
        _engaged(stream) & stream_known,
        ["user_id", "item_id", "time_ms", "date"],
    ].copy()
    stream_engaged["item_id"] = stream_engaged["item_id"].astype(np.int64)
    events = pd.concat((base_engaged, stream_engaged), ignore_index=True)
    events = events.sort_values(["user_id", "time_ms"], kind="stable")
    base_dates = sorted(base["date"].unique())
    training_dates = set(base_dates[-int(data["base_training_days"]) :])
    configured_update_dates = data.get("update_dates")
    if configured_update_dates is None:
        update_dates = [str(data["update_date"])]
    else:
        update_dates = [str(value) for value in configured_update_dates]
        if (
            not update_dates
            or update_dates != sorted(set(update_dates))
            or str(data["update_date"]) != update_dates[-1]
        ):
            raise ValueError("KuaiRand update dates differ")
    update_date = update_dates[-1]
    evaluation_date = str(data["evaluation_date"])
    max_prefix = int(document["model"]["max_seq_len"]) - int(
        not _true_next_item(document)
    )
    minimum_prefix = int(data["minimum_prefix_length"])
    training_target_horizon = int(data.get("training_target_horizon", 1))
    evaluation_targets_per_user = int(data.get("evaluation_targets_per_user", 1))
    if evaluation_targets_per_user < 1:
        raise ValueError("KuaiRand evaluation target count differs")
    user_payloads: dict[int, dict[str, Any]] = {}
    raw_eligible = []
    for raw_user, frame in events.groupby("user_id", sort=False):
        items = frame["item_id"].to_numpy(dtype=np.int64)
        dates = frame["date"].to_numpy(dtype=str)
        prediction_eligible = rank_by_item[items] >= 0
        base_examples = []
        for date in sorted(training_dates):
            candidates = np.flatnonzero(dates == date)
            candidates = candidates[candidates >= minimum_prefix]
            candidates = np.asarray(
                [
                    int(position)
                    for position in candidates
                    if _future_targets(
                        items,
                        dates,
                        prediction_eligible,
                        int(position),
                        training_target_horizon,
                    )
                    is not None
                ],
                dtype=np.int64,
            )
            for position in _positions(
                candidates,
                int(data["base_targets_per_user_day"]),
            ):
                start = max(0, int(position) - max_prefix)
                targets = _future_targets(
                    items,
                    dates,
                    prediction_eligible,
                    int(position),
                    training_target_horizon,
                )
                assert targets is not None
                base_examples.append((items[start:int(position)].copy(), targets))
        update_examples_by_date = {}
        for date in update_dates:
            date_positions = np.flatnonzero(
                (dates == date) & prediction_eligible
            )
            date_positions = date_positions[date_positions >= minimum_prefix]
            date_positions = np.asarray(
                [
                    int(position)
                    for position in date_positions
                    if _future_targets(
                        items,
                        dates,
                        prediction_eligible,
                        int(position),
                        training_target_horizon,
                    )
                    is not None
                ],
                dtype=np.int64,
            )
            date_examples = []
            for position in _positions(
                date_positions,
                int(data["update_targets_per_user"]),
            ):
                start = max(0, int(position) - max_prefix)
                targets = _future_targets(
                    items,
                    dates,
                    prediction_eligible,
                    int(position),
                    training_target_horizon,
                )
                assert targets is not None
                date_examples.append(
                    (items[start:int(position)].copy(), targets)
                )
            update_examples_by_date[date] = date_examples
        update_positions = np.flatnonzero(
            np.isin(dates, update_dates) & prediction_eligible
        )
        update_positions = update_positions[update_positions >= minimum_prefix]
        update_positions = np.asarray(
            [
                int(position)
                for position in update_positions
                if _future_targets(
                    items,
                    dates,
                    prediction_eligible,
                    int(position),
                    training_target_horizon,
                )
                is not None
            ],
            dtype=np.int64,
        )
        update_examples = []
        for position in _positions(
            update_positions,
            int(data["update_targets_per_user"]),
        ):
            start = max(0, int(position) - max_prefix)
            targets = _future_targets(
                items,
                dates,
                prediction_eligible,
                int(position),
                training_target_horizon,
            )
            assert targets is not None
            update_examples.append((items[start:int(position)].copy(), targets))
        evaluation_positions = np.flatnonzero(
            (dates == evaluation_date) & prediction_eligible
        )
        evaluation_positions = evaluation_positions[evaluation_positions >= minimum_prefix]
        if not base_examples or not update_examples or not len(evaluation_positions):
            continue
        evaluation_examples = []
        for evaluation_position in evaluation_positions[:evaluation_targets_per_user]:
            evaluation_position = int(evaluation_position)
            evaluation_start = max(0, evaluation_position - max_prefix)
            evaluation_examples.append(
                (
                    items[evaluation_start:evaluation_position].copy(),
                    int(items[evaluation_position]),
                )
            )
        raw_user = int(raw_user)
        user_payloads[raw_user] = {
            "base_examples": base_examples,
            "update_examples": update_examples,
            "update_examples_by_date": update_examples_by_date,
            "evaluation_examples": evaluation_examples,
        }
        raw_eligible.append(raw_user)
    ordered = _user_order(raw_eligible, int(data["selection_seed"]))
    limit = data.get("user_limit")
    selected = ordered if limit is None else ordered[: int(limit)]
    candidate_count = int(document["evaluation"]["candidate_count"])
    candidate_maps = {}
    base_examples = []
    update_examples = []
    update_examples_by_date = {date: [] for date in update_dates}
    evaluation = {}
    evaluation_keys = []
    for raw_user in selected:
        payload = user_payloads[raw_user]
        base_examples.extend(
            (raw_user, prefix, target)
            for prefix, target in payload["base_examples"]
        )
        update_examples.extend(
            (raw_user, prefix, target)
            for prefix, target in payload["update_examples"]
        )
        for date in update_dates:
            update_examples_by_date[date].extend(
                (raw_user, prefix, target)
                for prefix, target in payload["update_examples_by_date"][date]
            )
        for ordinal, (history, target) in enumerate(payload["evaluation_examples"]):
            key = raw_user if evaluation_targets_per_user == 1 else (raw_user, ordinal)
            excluded = set(int(value) for value in history)
            excluded.add(int(target))
            if (
                document["evaluation"]["candidate_source"]
                == "base_exposure_popularity_unseen"
            ):
                negatives = []
                for item_id in popular_ids:
                    value = int(item_id)
                    if value not in excluded:
                        negatives.append(value)
                    if len(negatives) == candidate_count - 1:
                        break
            else:
                seed_suffix = (
                    f"{raw_user}"
                    if evaluation_targets_per_user == 1
                    else f"{raw_user}:{ordinal}"
                )
                digest = hashlib.sha256(
                    f"{document['evaluation']['candidate_seed']}:{seed_suffix}".encode()
                ).digest()
                negatives = _frequency_matched_candidates(
                    int(target),
                    excluded,
                    popular_ids,
                    rank_by_item,
                    candidate_count - 1,
                    int.from_bytes(digest[:8], "little"),
                )
            if len(negatives) != candidate_count - 1:
                raise RuntimeError("KuaiRand candidate pool is incomplete")
            candidate_maps[key] = np.asarray(
                [int(target), *negatives],
                dtype=np.int64,
            )
            evaluation[key] = {
                "user_id": raw_user,
                "query_ordinal": ordinal,
                "history": history,
                "target": int(target),
            }
            evaluation_keys.append(key)
    date_rows = stream.groupby("date").size()
    date_engaged = stream.groupby("date").apply(
        lambda value: int(_engaged(value).sum()),
        include_groups=False,
    )
    target_known = np.zeros(len(stream), dtype=np.bool_)
    mapped_stream = stream.loc[stream_known, "item_id"].to_numpy(dtype=np.int64)
    target_known[np.flatnonzero(stream_known)] = rank_by_item[mapped_stream] >= 0
    known_engaged = stream.loc[_engaged(stream) & target_known].groupby("date").size()
    coverage = {}
    for date in [*update_dates, evaluation_date]:
        engaged_count = int(date_engaged.loc[date])
        known_count = int(known_engaged.loc[date])
        coverage[date] = {
            "rows": int(date_rows.loc[date]),
            "engaged_rows": engaged_count,
            "engaged_rows_in_base_catalog": known_count,
            "engaged_target_coverage_percent": 100.0 * known_count / engaged_count,
        }
    metadata = {
        "base_dates": base_dates,
        "base_training_dates": sorted(training_dates),
        "update_date": update_date,
        "update_dates": update_dates,
        "evaluation_date": evaluation_date,
        "prediction_items": len(raw_video_ids),
        "prediction_target_items": len(popular_ids),
        "author_items": embedding_rows - len(raw_video_ids),
        "embedding_rows": embedding_rows,
        "field_mode": document["model"].get("field_mode", "video_only"),
        "author_ids_sha256": author_ids_sha256,
        "prediction_video_ids_sha256": _hash_ints(raw_video_ids),
        "eligible_users": len(raw_eligible),
        "selected_users": len(selected),
        "selected_raw_user_ids_sha256": _hash_ints(np.asarray(selected, dtype=np.int64)),
        "base_examples": len(base_examples),
        "base_effective_targets": len(base_examples) * training_target_horizon,
        "update_examples": len(update_examples),
        "update_effective_targets": len(update_examples) * training_target_horizon,
        "update_examples_by_date": {
            date: len(update_examples_by_date[date]) for date in update_dates
        },
        "training_target_horizon": training_target_horizon,
        "evaluation_records": len(evaluation),
        "evaluation_targets_per_user": evaluation_targets_per_user,
        "target_coverage": coverage,
        "target_leakage": False,
        "catalog_fit": "all base-period exposures only",
        "target_catalog_fit": data.get("target_catalog_fit", "all_base_exposures"),
        "sequence_semantics": "engaged actions whose video is in the base-fitted catalog",
        "evaluation_semantics": (
            "first eligible engaged action on the next natural day"
            if evaluation_targets_per_user == 1
            else f"first {evaluation_targets_per_user} eligible engaged actions "
            "on the next natural day"
        ),
        "candidate_source": document["evaluation"]["candidate_source"],
        "training_negative_source": document["training"]["negative_source"],
        "training_objective": document["training"]["objective"],
        "prediction_state": (
            "last_history_item_hidden"
            if _true_next_item(document)
            else (
                "current_latest_item_plus_appended_query_hidden"
                if document["model"].get("query_mode") == "latest_item_query"
                else "appended_query_hidden"
            )
        ),
        "stale_serving_semantics": (
            "old_version_prefix_cache_plus_current_model_latest_item"
            if _true_next_item(document)
            else (
                "old_version_prefix_before_latest_plus_current_latest_and_query"
                if document["model"].get("query_mode") == "latest_item_query"
                else "old_version_full_history_cache_plus_current_model_query"
            )
        ),
    }
    return {
        "base_examples": base_examples,
        "update_examples": update_examples,
        "update_examples_by_date": update_examples_by_date,
        "evaluation": evaluation,
        "candidate_maps": candidate_maps,
        "popular_ids": popular_ids,
        "rank_by_item": rank_by_item,
        "author_by_item": author_by_item,
        "selected_users": selected,
        "evaluation_keys": evaluation_keys,
        "metadata": metadata,
    }


def make_model(document: dict[str, Any], num_items: int, device: torch.device) -> HSTU:
    model = document["model"]
    hidden_size = int(model["hidden_size"])
    config = HSTUConfig(
        num_items=num_items,
        num_prediction_items=num_items,
        num_behaviors=1,
        hidden_size=hidden_size,
        num_layers=int(model["num_layers"]),
        num_heads=int(model["num_heads"]),
        head_dim=hidden_size // int(model["num_heads"]),
        max_seq_len=int(model["max_seq_len"]),
        input_dropout=float(model["input_dropout"]),
        activation=str(model["activation"]),
        qk_scale=float(model["qk_scale"]),
        gating=str(model.get("gating", "silu_gate")),
        block_variant=str(model.get("block_variant", "legacy")),
        relative_position_bias=bool(model.get("relative_position_bias", False)),
        causal_diagonal=str(model.get("causal_diagonal", "inclusive")),
    )
    result = HSTU(config).to(device)
    result.query_mode = str(model.get("query_mode", "learned_token"))
    return result


def _input_vectors(
    model: HSTU,
    item_ids: torch.Tensor,
    author_by_item: torch.Tensor,
) -> torch.Tensor:
    vectors = model.lookup_item_embeddings(item_ids)
    author_ids = author_by_item.index_select(0, item_ids.reshape(-1)).reshape_as(item_ids)
    if bool(torch.any(author_ids != 0)):
        has_author = author_ids != 0
        author_vectors = model.lookup_item_embeddings(author_ids)
        combined = (vectors + author_vectors) * (2.0**-0.5)
        vectors = torch.where(has_author.unsqueeze(-1), combined, vectors)
    return vectors


def _forward(
    model: HSTU,
    item_ids: torch.Tensor,
    behaviors: torch.Tensor,
    deltas: torch.Tensor,
    lengths: torch.Tensor,
    author_by_item: torch.Tensor,
) -> torch.Tensor:
    embedded = model.combine_input_features(
        _input_vectors(model, item_ids, author_by_item), behaviors, deltas
    )
    if getattr(model, "query_mode", "learned_token") == "history_only_zero":
        rows = torch.arange(len(lengths), device=embedded.device)
        embedded[rows, lengths - 1] = 0
    hidden, _ = model.forward_embedded(embedded, lengths=lengths)
    return hidden


def _score(
    model: HSTU,
    hidden: torch.Tensor,
    candidates: torch.Tensor,
    temperature: float,
    author_by_item: torch.Tensor,
) -> torch.Tensor:
    candidate_vectors = _input_vectors(model, candidates, author_by_item)
    return torch.einsum(
        "nh,nch->nc",
        F.normalize(hidden, dim=-1),
        F.normalize(candidate_vectors, dim=-1),
    ) / temperature


def _candidate_metrics_tie_aware(
    scores: torch.Tensor,
    target_index: torch.Tensor,
) -> dict[str, torch.Tensor]:
    ce = F.cross_entropy(scores, target_index, reduction="none")
    rows = torch.arange(scores.shape[0], device=scores.device)
    positive = scores[rows, target_index]
    greater = (scores > positive.unsqueeze(1)).sum(dim=1).float()
    equal = (scores == positive.unsqueeze(1)).sum(dim=1).float() - 1.0
    ranks = 1.0 + greater + 0.5 * equal
    return {
        "candidate_cross_entropy": ce,
        "mrr": ranks.reciprocal(),
        "ndcg_at_5": torch.where(
            ranks <= 5,
            torch.log2(ranks + 1).reciprocal(),
            torch.zeros_like(ranks),
        ),
        "ndcg_at_10": torch.where(
            ranks <= 10,
            torch.log2(ranks + 1).reciprocal(),
            torch.zeros_like(ranks),
        ),
        "hit_rate_at_1": (ranks <= 1).float(),
        "hit_rate_at_5": (ranks <= 5).float(),
        "hit_rate_at_10": (ranks <= 10).float(),
    }


def _collate_targets(examples, device: torch.device) -> torch.Tensor:
    values = [np.asarray(value[2], dtype=np.int64).reshape(-1) for value in examples]
    widths = {len(value) for value in values}
    if len(widths) != 1 or not values or not len(values[0]):
        raise ValueError("KuaiRand training target horizon differs")
    result = torch.as_tensor(
        np.stack(values), dtype=torch.long, device=device
    )
    return result[:, 0] if result.shape[1] == 1 else result


def _collate(examples, device: torch.device):
    lengths = torch.tensor(
        [len(value[1]) + 1 for value in examples],
        dtype=torch.long,
        device=device,
    )
    width = int(lengths.max().item())
    items = torch.zeros(len(examples), width, dtype=torch.long, device=device)
    for row, (_, prefix, _) in enumerate(examples):
        items[row, : len(prefix)] = torch.as_tensor(prefix, dtype=torch.long, device=device)
    behaviors = (
        torch.arange(width, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    ).long()
    deltas = torch.zeros_like(items, dtype=torch.float32)
    targets = _collate_targets(examples, device)
    return items, behaviors, deltas, lengths, targets


def _collate_true_next_item(examples, device: torch.device):
    lengths = torch.tensor(
        [len(value[1]) for value in examples],
        dtype=torch.long,
        device=device,
    )
    width = int(lengths.max().item())
    items = torch.zeros(len(examples), width, dtype=torch.long, device=device)
    for row, (_, prefix, _) in enumerate(examples):
        items[row, : len(prefix)] = torch.as_tensor(
            prefix, dtype=torch.long, device=device
        )
    behaviors = (
        torch.arange(width, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    ).long()
    deltas = torch.zeros_like(items, dtype=torch.float32)
    targets = _collate_targets(examples, device)
    return items, behaviors, deltas, lengths, targets


def _training_candidates(
    targets: torch.Tensor,
    negative_pool: torch.Tensor,
    negative_count: int,
    generator: torch.Generator,
    source: str,
    num_items: int,
    rank_by_item: torch.Tensor,
) -> torch.Tensor:
    if source == "base_exposure_popularity_pool":
        positions = torch.randint(
            0,
            len(negative_pool),
            (len(targets), negative_count),
            device=targets.device,
            generator=generator,
        )
        negatives = negative_pool.index_select(0, positions.reshape(-1)).reshape_as(positions)
        replacements = negative_pool.index_select(
            0, (positions + 1).remainder(len(negative_pool)).reshape(-1)
        ).reshape_as(positions)
    else:
        target_ranks = rank_by_item.index_select(0, targets).long() + 1
        if bool(torch.any(target_ranks < 1)):
            raise RuntimeError("KuaiRand training target is outside negative pool")
        exponents = torch.floor(torch.log2(target_ranks.double())).long()
        lower = torch.bitwise_left_shift(torch.ones_like(exponents), exponents)
        upper = torch.minimum(
            lower * 2,
            torch.full_like(lower, len(negative_pool) + 1),
        )
        upper = torch.maximum(
            upper,
            torch.minimum(
                lower + 512,
                torch.full_like(lower, len(negative_pool) + 1),
            ),
        )
        lower = torch.minimum(
            lower,
            torch.full_like(lower, max(1, len(negative_pool) + 1 - 512)),
        )
        spans = (upper - lower).clamp_min(1)
        draws = torch.rand(
            (len(targets), negative_count),
            device=targets.device,
            generator=generator,
        )
        negative_positions = lower.unsqueeze(1) + torch.floor(
            draws * spans.unsqueeze(1)
        ).long()
        replacement_positions = lower.unsqueeze(1) + (
            negative_positions - lower.unsqueeze(1) + 1
        ).remainder(spans.unsqueeze(1))
        negatives = negative_pool.index_select(
            0, (negative_positions - 1).reshape(-1)
        ).reshape_as(negative_positions)
        replacements = negative_pool.index_select(
            0, (replacement_positions - 1).reshape(-1)
        ).reshape_as(replacement_positions)
    negatives = torch.where(negatives == targets.unsqueeze(1), replacements, negatives)
    return torch.cat((targets.unsqueeze(1), negatives), dim=1)


def _train(
    model: HSTU,
    examples,
    negative_ids: np.ndarray,
    rank_by_item_ids: np.ndarray,
    author_by_item_ids: np.ndarray,
    document: dict[str, Any],
    phase: str,
    seed: int,
) -> dict[str, Any]:
    training = document["training"]
    epochs = int(training["base_epochs"] if phase == "base" else training["update_epochs"])
    learning_rate = float(training["base_lr"] if phase == "base" else training["update_lr"])
    embedding_learning_rate = float(
        training.get(
            "base_embedding_lr" if phase == "base" else "update_embedding_lr",
            learning_rate,
        )
    )
    embedding_parameter = model.item_emb.weight
    dense_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter is not embedding_parameter
    ]
    kv_key = "base_kv_lr" if phase == "base" else "update_kv_lr"
    kv_learning_rate = float(training.get(kv_key, learning_rate))
    if kv_key in training:
        kv_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if ".attn.k_proj." in name or ".attn.v_proj." in name
        ]
        kv_parameter_ids = {id(parameter) for parameter in kv_parameters}
        other_dense_parameters = [
            parameter
            for parameter in dense_parameters
            if id(parameter) not in kv_parameter_ids
        ]
        optimizer_parameters = [
            {"params": other_dense_parameters, "lr": learning_rate},
            {"params": kv_parameters, "lr": kv_learning_rate},
        ]
    else:
        optimizer_parameters = dense_parameters
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=learning_rate,
        weight_decay=float(training["weight_decay"]),
        foreach=False,
    )
    embedding_optimizer = torch.optim.SGD(
        [embedding_parameter],
        lr=embedding_learning_rate,
        momentum=0.0,
        weight_decay=0.0,
        foreach=False,
    )
    model.item_emb.sparse_gradient = True
    device = next(model.parameters()).device
    negative_pool = torch.as_tensor(
        np.asarray(negative_ids).copy(), dtype=torch.long, device=device
    )
    rank_by_item = torch.as_tensor(rank_by_item_ids, dtype=torch.long, device=device)
    author_by_item = torch.as_tensor(
        np.asarray(author_by_item_ids).copy(), dtype=torch.long, device=device
    )
    generator = torch.Generator(device=device).manual_seed(seed + 9173)
    rng = np.random.default_rng(seed)
    epoch_results = []
    started = time.monotonic()
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(examples))
        loss_sum = 0.0
        count = 0
        target_count = 0
        for start in range(0, len(order), int(training["batch_size"])):
            batch = [examples[int(index)] for index in order[start : start + int(training["batch_size"])]]
            items, behaviors, deltas, lengths, targets = (
                _collate_true_next_item(batch, device)
                if _true_next_item(document)
                else _collate(batch, device)
            )
            optimizer.zero_grad(set_to_none=True)
            embedding_optimizer.zero_grad(set_to_none=True)
            hidden = _forward(
                model, items, behaviors, deltas, lengths, author_by_item
            )
            vectors = model.last_hidden(hidden, lengths)
            if targets.ndim == 2:
                scoring_vectors = vectors.unsqueeze(1).expand(
                    -1, targets.shape[1], -1
                ).reshape(-1, vectors.shape[1])
                scoring_targets = targets.reshape(-1)
            else:
                scoring_vectors = vectors
                scoring_targets = targets
            candidates = _training_candidates(
                scoring_targets,
                negative_pool,
                int(training["negative_samples"]),
                generator,
                str(training["negative_source"]),
                model.cfg.num_prediction_items,
                rank_by_item,
            )
            scores = _score(
                model,
                scoring_vectors,
                candidates,
                float(training["temperature"]),
                author_by_item,
            )
            loss = F.cross_entropy(
                scores,
                torch.zeros(
                    len(scoring_targets), dtype=torch.long, device=device
                ),
            )
            if not torch.isfinite(loss):
                raise RuntimeError("KuaiRand prediction-query loss is nonfinite")
            loss.backward()
            embedding_gradient = embedding_parameter.grad
            if (
                embedding_gradient is None
                or not embedding_gradient.is_sparse
                or not bool(
                    torch.all(
                        torch.isfinite(embedding_gradient.coalesce().values())
                    )
                )
            ):
                raise RuntimeError("KuaiRand sparse embedding gradient differs")
            torch.nn.utils.clip_grad_norm_(
                dense_parameters, float(training["gradient_clip_norm"])
            )
            optimizer.step()
            embedding_optimizer.step()
            with torch.no_grad():
                embedding_parameter[0].zero_()
            loss_sum += float(loss.detach().item()) * len(batch)
            count += len(batch)
            target_count += int(targets.numel())
        epoch_results.append(
            {
                "epoch": epoch + 1,
                "mean_sampled_cross_entropy": loss_sum / count,
                "examples": count,
                "effective_targets": target_count,
            }
        )
        print(
            f"phase=kuairand_query_{phase} epoch={epoch + 1}/{epochs} "
            f"loss={loss_sum / count:.6f} examples={count} "
            f"targets={target_count}",
            flush=True,
        )
    model.item_emb.sparse_gradient = False
    model.eval()
    return {
        "phase": phase,
        "epochs": epoch_results,
        "dense_optimizer": "adamw",
        "embedding_optimizer": "row_sparse_sgd",
        "dense_learning_rate": learning_rate,
        "kv_learning_rate": kv_learning_rate,
        "embedding_learning_rate": embedding_learning_rate,
        "elapsed_seconds": time.monotonic() - started,
    }


def _empty_cache(reference):
    from ..models import HSTUKVCache

    return HSTUKVCache(k=reference.k[:, :, :0], v=reference.v[:, :, :0], seq_len=0)


def _relative_rows(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    numerator = torch.linalg.vector_norm(
        (value - reference).double(), dim=tuple(range(2, value.ndim))
    )
    denominator = torch.linalg.vector_norm(
        reference.double(), dim=tuple(range(2, value.ndim))
    )
    return numerator / denominator.clamp_min(1e-12)


@torch.no_grad()
def _evaluate_true_next_item(
    previous: HSTU,
    current: HSTU,
    workload: dict[str, Any],
    document: dict[str, Any],
) -> dict[str, Any]:
    device = next(current.parameters()).device
    groups = defaultdict(list)
    evaluation_keys = workload.get("evaluation_keys", workload["selected_users"])
    for key in evaluation_keys:
        record = workload["evaluation"][key]
        groups[len(record["history"])].append(key)
    details = []
    maximum_hidden_error = 0.0
    maximum_score_error = 0.0
    temperature = float(document["training"]["temperature"])
    author_by_item = torch.as_tensor(
        np.asarray(workload["author_by_item"]).copy(),
        dtype=torch.long,
        device=device,
    )
    batch_size = int(document["evaluation"]["batch_size"])
    for history_length, keys in sorted(groups.items()):
        if history_length < 2:
            raise RuntimeError("KuaiRand true next-item history is too short")
        for start in range(0, len(keys), batch_size):
            batch_keys = keys[start : start + batch_size]
            batch_users = [
                int(workload["evaluation"][key].get("user_id", key))
                for key in batch_keys
            ]
            history = torch.stack(
                [
                    torch.as_tensor(
                        workload["evaluation"][key]["history"],
                        dtype=torch.long,
                        device=device,
                    )
                    for key in batch_keys
                ]
            )
            prefix = history[:, :-1]
            latest = history[:, -1:]
            prefix_behaviors = torch.ones_like(prefix)
            prefix_deltas = torch.zeros_like(prefix, dtype=torch.float32)
            latest_behaviors = torch.ones_like(latest)
            latest_deltas = torch.zeros_like(latest, dtype=torch.float32)
            candidates = torch.stack(
                [
                    torch.as_tensor(
                        workload["candidate_maps"][key],
                        dtype=torch.long,
                        device=device,
                    )
                    for key in batch_keys
                ]
            )
            previous_cache = previous.compute_kv_from_item_embeddings(
                _input_vectors(previous, prefix, author_by_item),
                prefix_behaviors,
                prefix_deltas,
            )
            current_cache = current.compute_kv_from_item_embeddings(
                _input_vectors(current, prefix, author_by_item),
                prefix_behaviors,
                prefix_deltas,
            )
            current_latest_embedded = current.combine_input_features(
                _input_vectors(current, latest, author_by_item),
                latest_behaviors,
                latest_deltas,
            )
            previous_latest_embedded = previous.combine_input_features(
                _input_vectors(previous, latest, author_by_item),
                latest_behaviors,
                latest_deltas,
            )
            reuse_hidden, _ = current.forward_with_cache_embedded(
                previous_cache, current_latest_embedded
            )
            fresh_incremental, _ = current.forward_with_cache_embedded(
                current_cache, current_latest_embedded
            )
            previous_incremental, _ = previous.forward_with_cache_embedded(
                previous_cache, previous_latest_embedded
            )
            no_prefix_hidden, _ = current.forward_with_cache_embedded(
                _empty_cache(current_cache), current_latest_embedded
            )
            full_behaviors = torch.ones_like(history)
            full_deltas = torch.zeros_like(history, dtype=torch.float32)
            full_lengths = torch.full(
                (len(batch_users),),
                history_length,
                dtype=torch.long,
                device=device,
            )
            fresh_full = _forward(
                current,
                history,
                full_behaviors,
                full_deltas,
                full_lengths,
                author_by_item,
            )
            previous_full = _forward(
                previous,
                history,
                full_behaviors,
                full_deltas,
                full_lengths,
                author_by_item,
            )
            method_vectors = {
                "previous_fresh": previous_full[:, -1],
                "recompute": fresh_full[:, -1],
                "reuse": reuse_hidden[:, -1],
                "no_prefix": no_prefix_hidden[:, -1],
            }
            method_scores = {
                method: _score(
                    previous if method == "previous_fresh" else current,
                    vector,
                    candidates,
                    temperature,
                    author_by_item,
                )
                for method, vector in method_vectors.items()
            }
            duplicate_scores = _score(
                current,
                fresh_incremental[:, -1],
                candidates,
                temperature,
                author_by_item,
            )
            maximum_hidden_error = max(
                maximum_hidden_error,
                float(
                    (fresh_incremental[:, -1] - fresh_full[:, -1])
                    .abs()
                    .max()
                    .item()
                ),
                float(
                    (previous_incremental[:, -1] - previous_full[:, -1])
                    .abs()
                    .max()
                    .item()
                ),
            )
            maximum_score_error = max(
                maximum_score_error,
                float(
                    (duplicate_scores - method_scores["recompute"])
                    .abs()
                    .max()
                    .item()
                ),
            )
            metric_values = {
                method: _candidate_metrics_tie_aware(
                    scores,
                    torch.zeros(
                        len(batch_users), dtype=torch.long, device=device
                    ),
                )
                for method, scores in method_scores.items()
            }
            cache_k_error_by_layer = _relative_rows(
                previous_cache.k, current_cache.k
            )
            cache_v_error_by_layer = _relative_rows(
                previous_cache.v, current_cache.v
            )
            cache_k_error = cache_k_error_by_layer.mean(dim=0)
            cache_v_error = cache_v_error_by_layer.mean(dim=0)
            hidden_error = torch.linalg.vector_norm(
                (reuse_hidden[:, -1] - fresh_full[:, -1]).double(), dim=1
            ) / torch.linalg.vector_norm(
                fresh_full[:, -1].double(), dim=1
            ).clamp_min(1e-12)
            for row, (key, user) in enumerate(
                zip(batch_keys, batch_users, strict=True)
            ):
                source = workload["evaluation"][key]
                details.append(
                    {
                        "user_id": int(user),
                        "query_ordinal": int(source.get("query_ordinal", 0)),
                        "history_length": history_length,
                        "cache_prefix_length": history_length - 1,
                        "metrics": {
                            method: {
                                metric: float(
                                    metric_values[method][metric][row].item()
                                )
                                for metric in METRICS
                            }
                            for method in METHODS
                        },
                        "cache_k_relative_error": float(
                            cache_k_error[row].item()
                        ),
                        "cache_v_relative_error": float(
                            cache_v_error[row].item()
                        ),
                        "cache_k_relative_error_by_layer": [
                            float(value)
                            for value in cache_k_error_by_layer[:, row].tolist()
                        ],
                        "cache_v_relative_error_by_layer": [
                            float(value)
                            for value in cache_v_error_by_layer[:, row].tolist()
                        ],
                        "hidden_relative_error": float(hidden_error[row].item()),
                    }
                )
    return {
        "records": details,
        "sanity": {
            "maximum_same_model_incremental_hidden_absolute_error": maximum_hidden_error,
            "maximum_same_model_incremental_score_absolute_error": maximum_score_error,
            "passed": maximum_hidden_error <= 1e-4
            and maximum_score_error <= 1e-4,
        },
    }


@torch.no_grad()
def _evaluate(
    previous: HSTU,
    current: HSTU,
    workload: dict[str, Any],
    document: dict[str, Any],
) -> dict[str, Any]:
    if _true_next_item(document):
        return _evaluate_true_next_item(previous, current, workload, document)
    device = next(current.parameters()).device
    groups = defaultdict(list)
    evaluation_keys = workload.get("evaluation_keys", workload["selected_users"])
    for key in evaluation_keys:
        record = workload["evaluation"][key]
        groups[len(record["history"])].append(key)
    details = []
    maximum_hidden_error = 0.0
    maximum_score_error = 0.0
    temperature = float(document["training"]["temperature"])
    author_by_item = torch.as_tensor(
        np.asarray(workload["author_by_item"]).copy(), dtype=torch.long, device=device
    )
    batch_size = int(document["evaluation"]["batch_size"])
    for history_length, keys in sorted(groups.items()):
        for start in range(0, len(keys), batch_size):
            batch_keys = keys[start : start + batch_size]
            batch_users = [
                int(workload["evaluation"][key].get("user_id", key))
                for key in batch_keys
            ]
            history = torch.stack(
                [
                    torch.as_tensor(
                        workload["evaluation"][key]["history"],
                        dtype=torch.long,
                        device=device,
                    )
                    for key in batch_keys
                ]
            )
            latest_item_query = (
                getattr(current, "query_mode", "learned_token")
                == "latest_item_query"
            )
            prefix = history[:, :-1] if latest_item_query else history
            prefix_behaviors = torch.ones_like(prefix)
            prefix_deltas = torch.zeros_like(prefix, dtype=torch.float32)
            query = torch.zeros(len(batch_users), 1, dtype=torch.long, device=device)
            suffix_items = (
                torch.cat((history[:, -1:], query), dim=1)
                if latest_item_query
                else query
            )
            suffix_behaviors = torch.ones_like(suffix_items)
            suffix_deltas = torch.zeros_like(suffix_items, dtype=torch.float32)
            candidates = torch.stack(
                [
                    torch.as_tensor(
                        workload["candidate_maps"][key],
                        dtype=torch.long,
                        device=device,
                    )
                    for key in batch_keys
                ]
            )
            previous_cache = previous.compute_kv_from_item_embeddings(
                _input_vectors(previous, prefix, author_by_item),
                prefix_behaviors,
                prefix_deltas,
            )
            current_cache = current.compute_kv_from_item_embeddings(
                _input_vectors(current, prefix, author_by_item),
                prefix_behaviors,
                prefix_deltas,
            )
            if getattr(current, "query_mode", "learned_token") == "history_only_zero":
                current_suffix_embedded = torch.zeros(
                    len(batch_users), 1, current.cfg.hidden_size, device=device
                )
                previous_suffix_embedded = torch.zeros_like(current_suffix_embedded)
            else:
                current_suffix_embedded = current.combine_input_features(
                    _input_vectors(current, suffix_items, author_by_item),
                    suffix_behaviors,
                    suffix_deltas,
                )
                previous_suffix_embedded = previous.combine_input_features(
                    _input_vectors(previous, suffix_items, author_by_item),
                    suffix_behaviors,
                    suffix_deltas,
                )
            reuse_hidden, _ = current.forward_with_cache_embedded(
                previous_cache, current_suffix_embedded
            )
            fresh_incremental, _ = current.forward_with_cache_embedded(
                current_cache, current_suffix_embedded
            )
            previous_incremental, _ = previous.forward_with_cache_embedded(
                previous_cache, previous_suffix_embedded
            )
            no_prefix_hidden, _ = current.forward_with_cache_embedded(
                _empty_cache(current_cache), current_suffix_embedded
            )
            full_items = torch.cat((history, query), dim=1)
            full_behaviors = torch.ones_like(full_items)
            full_deltas = torch.zeros_like(full_items, dtype=torch.float32)
            full_lengths = torch.full(
                (len(batch_users),), full_items.shape[1], dtype=torch.long, device=device
            )
            fresh_full = _forward(
                current,
                full_items,
                full_behaviors,
                full_deltas,
                full_lengths,
                author_by_item,
            )
            previous_full = _forward(
                previous,
                full_items,
                full_behaviors,
                full_deltas,
                full_lengths,
                author_by_item,
            )
            method_vectors = {
                "previous_fresh": previous_full[:, -1],
                "recompute": fresh_full[:, -1],
                "reuse": reuse_hidden[:, -1],
                "no_prefix": no_prefix_hidden[:, -1],
            }
            method_scores = {
                method: _score(
                    previous if method == "previous_fresh" else current,
                    vector,
                    candidates,
                    temperature,
                    author_by_item,
                )
                for method, vector in method_vectors.items()
            }
            duplicate_scores = _score(
                current,
                fresh_incremental[:, -1],
                candidates,
                temperature,
                author_by_item,
            )
            maximum_hidden_error = max(
                maximum_hidden_error,
                float((fresh_incremental[:, -1] - fresh_full[:, -1]).abs().max().item()),
                float((previous_incremental[:, -1] - previous_full[:, -1]).abs().max().item()),
            )
            maximum_score_error = max(
                maximum_score_error,
                float((duplicate_scores - method_scores["recompute"]).abs().max().item()),
            )
            metric_values = {
                method: _candidate_metrics_tie_aware(
                    scores,
                    torch.zeros(len(batch_users), dtype=torch.long, device=device),
                )
                for method, scores in method_scores.items()
            }
            cache_k_error_by_layer = _relative_rows(previous_cache.k, current_cache.k)
            cache_v_error_by_layer = _relative_rows(previous_cache.v, current_cache.v)
            cache_k_error = cache_k_error_by_layer.mean(dim=0)
            cache_v_error = cache_v_error_by_layer.mean(dim=0)
            hidden_error = torch.linalg.vector_norm(
                (reuse_hidden[:, -1] - fresh_full[:, -1]).double(), dim=1
            ) / torch.linalg.vector_norm(fresh_full[:, -1].double(), dim=1).clamp_min(1e-12)
            for row, (key, user) in enumerate(zip(batch_keys, batch_users, strict=True)):
                source = workload["evaluation"][key]
                details.append(
                    {
                        "user_id": int(user),
                        "query_ordinal": int(source.get("query_ordinal", 0)),
                        "history_length": history_length,
                        "cache_prefix_length": int(prefix.shape[1]),
                        "metrics": {
                            method: {
                                metric: float(metric_values[method][metric][row].item())
                                for metric in METRICS
                            }
                            for method in METHODS
                        },
                        "cache_k_relative_error": float(cache_k_error[row].item()),
                        "cache_v_relative_error": float(cache_v_error[row].item()),
                        "cache_k_relative_error_by_layer": [
                            float(value)
                            for value in cache_k_error_by_layer[:, row].tolist()
                        ],
                        "cache_v_relative_error_by_layer": [
                            float(value)
                            for value in cache_v_error_by_layer[:, row].tolist()
                        ],
                        "hidden_relative_error": float(hidden_error[row].item()),
                    }
                )
    return {
        "records": details,
        "sanity": {
            "maximum_same_model_incremental_hidden_absolute_error": maximum_hidden_error,
            "maximum_same_model_incremental_score_absolute_error": maximum_score_error,
            "passed": maximum_hidden_error <= 1e-4 and maximum_score_error <= 1e-4,
        },
    }


def _summary(evaluation: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    raw_records = evaluation["records"]
    grouped = defaultdict(list)
    for record in raw_records:
        grouped[int(record["user_id"])].append(record)
    records = []
    for user_id, values in sorted(grouped.items()):
        records.append(
            {
                "user_id": user_id,
                "metrics": {
                    method: {
                        metric: float(
                            np.mean(
                                [value["metrics"][method][metric] for value in values]
                            )
                        )
                        for metric in METRICS
                    }
                    for method in METHODS
                },
                **{
                    field: float(np.mean([value[field] for value in values]))
                    for field in (
                        "cache_k_relative_error",
                        "cache_v_relative_error",
                        "hidden_relative_error",
                    )
                },
                **(
                    {
                        field: np.mean(
                            np.asarray(
                                [value[field] for value in values], dtype=np.float64
                            ),
                            axis=0,
                        ).tolist()
                        for field in (
                            "cache_k_relative_error_by_layer",
                            "cache_v_relative_error_by_layer",
                        )
                    }
                    if all(
                        field in value
                        for value in values
                        for field in (
                            "cache_k_relative_error_by_layer",
                            "cache_v_relative_error_by_layer",
                        )
                    )
                    else {}
                ),
            }
        )
    endpoints = {
        method: {
            metric: float(np.mean([record["metrics"][method][metric] for record in records]))
            for metric in METRICS
        }
        for method in METHODS
    }
    samples = int(document["evaluation"]["bootstrap_samples"])
    seed = int(document["evaluation"]["bootstrap_seed"])
    comparisons = {
        "fresh_update_value": _comparison(records, "recompute", "previous_fresh", samples, seed),
        "recompute_over_reuse": _comparison(records, "recompute", "reuse", samples, seed + 101),
        "history_value": _comparison(records, "recompute", "no_prefix", samples, seed + 202),
    }
    stale = comparisons["recompute_over_reuse"]
    ranking = ("hit_rate_at_5", "hit_rate_at_10", "mrr", "ndcg_at_5", "ndcg_at_10")
    passing = [
        metric
        for metric in ranking
        if stale[metric]["positive_direction_with_ci"]
        and stale[metric]["relative_percent"] >= float(document["evaluation"]["minimum_relative_percent"])
    ]
    return {
        "users": len(records),
        "evaluation_records": len(raw_records),
        "endpoints": endpoints,
        "comparisons": comparisons,
        "representation_drift": {
            field: {
                "mean": float(np.mean([record[field] for record in records])),
                "median": float(np.median([record[field] for record in records])),
                "p95": float(np.quantile([record[field] for record in records], 0.95)),
            }
            for field in ("cache_k_relative_error", "cache_v_relative_error", "hidden_relative_error")
        },
        "layerwise_cache_drift": [
            {
                "layer": layer,
                **{
                    field: {
                        "mean": float(
                            np.mean([record[field][layer] for record in records])
                        ),
                        "median": float(
                            np.median([record[field][layer] for record in records])
                        ),
                        "p95": float(
                            np.quantile(
                                [record[field][layer] for record in records], 0.95
                            )
                        ),
                    }
                    for field in (
                        "cache_k_relative_error_by_layer",
                        "cache_v_relative_error_by_layer",
                    )
                },
            }
            for layer in range(
                len(records[0]["cache_k_relative_error_by_layer"])
                if records and "cache_k_relative_error_by_layer" in records[0]
                else 0
            )
        ],
        "gate": {
            "same_model_sanity": evaluation["sanity"]["passed"],
            "fresh_update_ranking_positive": any(
                comparisons["fresh_update_value"][metric]["positive_direction_with_ci"]
                for metric in ranking
            ),
            "history_ranking_positive": any(
                comparisons["history_value"][metric]["positive_direction_with_ci"]
                for metric in ranking
            ),
            "stale_candidate_ce_positive_ci": stale["candidate_cross_entropy"][
                "positive_direction_with_ci"
            ],
            "passing_ranking_metrics": passing,
            "minimum_passing_ranking_metrics": int(
                document["evaluation"]["minimum_passing_ranking_metrics"]
            ),
            "passed": bool(
                evaluation["sanity"]["passed"]
                and any(
                    comparisons["fresh_update_value"][metric]["positive_direction_with_ci"]
                    for metric in ranking
                )
                and any(
                    comparisons["history_value"][metric]["positive_direction_with_ci"]
                    for metric in ranking
                )
                and stale["candidate_cross_entropy"]["positive_direction_with_ci"]
                and len(passing)
                >= int(document["evaluation"]["minimum_passing_ranking_metrics"])
            ),
        },
        "sanity": evaluation["sanity"],
    }


def _state_delta(previous: dict[str, torch.Tensor], current: dict[str, torch.Tensor]):
    groups = defaultdict(lambda: [0.0, 0.0])
    for name, old in previous.items():
        new = current[name]
        group = "item_embedding" if name.startswith("item_emb") else "cache_core"
        groups[group][0] += float((new.double() - old.double()).pow(2).sum().item())
        groups[group][1] += float(old.double().pow(2).sum().item())
    return {
        group: {
            "absolute_l2": math.sqrt(values[0]),
            "relative_l2": math.sqrt(values[0]) / max(math.sqrt(values[1]), 1e-12),
        }
        for group, values in sorted(groups.items())
    }


def _checkpoint(path: Path, model: HSTU, metadata: dict[str, Any]) -> dict[str, Any]:
    _atomic_torch(
        path,
        {
            "protocol": PROTOCOL,
            "config": vars(model.cfg),
            "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "metadata": metadata,
        },
    )
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def run(config_path: str | Path) -> dict[str, Any]:
    document = load_config(config_path)
    output_root = Path(document["outputs"]["root"])
    summary_path = output_root / "summary.json"
    if summary_path.is_file():
        result = json.loads(summary_path.read_text())
        validate_summary(result, document)
        return result
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    workload = build_workload(document)
    _atomic_json(output_root / "workload.json", workload["metadata"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("KuaiRand prediction-query experiment requires CUDA")
    negative_pool_size = min(
        int(document["training"]["negative_pool_size"]),
        len(workload["popular_ids"]),
    )
    negative_ids = workload["popular_ids"][:negative_pool_size]
    seed_results = []
    for seed in document["training"]["seeds"]:
        seed = int(seed)
        print(f"phase=kuairand_query_seed_start seed={seed}", flush=True)
        _seed_everything(seed)
        previous = make_model(
            document, int(workload["metadata"]["embedding_rows"]), device
        )
        parent_theta0 = document["training"].get("parent_theta0")
        if parent_theta0 is None:
            base_training = _train(
                previous,
                workload["base_examples"],
                negative_ids,
                workload["rank_by_item"],
                workload["author_by_item"],
                document,
                "base",
                seed + 1009,
            )
            theta0 = _checkpoint(
                output_root / "checkpoints" / f"seed_{seed}" / "theta0.pt",
                previous,
                {"training": base_training, "workload": workload["metadata"]},
            )
        else:
            parent_payload = torch.load(
                parent_theta0["path"], map_location=device, weights_only=True
            )
            if (
                parent_payload.get("protocol") != PROTOCOL
                or parent_payload.get("config") != vars(previous.cfg)
            ):
                raise ValueError("KuaiRand parent theta0 payload differs")
            previous.load_state_dict(parent_payload["state_dict"])
            previous.eval()
            base_training = {
                "status": "imported_parent_theta0",
                "source": parent_theta0,
                "parent_metadata": parent_payload.get("metadata"),
            }
            theta0_path = Path(parent_theta0["path"])
            theta0 = {
                "path": str(theta0_path),
                "bytes": theta0_path.stat().st_size,
                "sha256": file_sha256(theta0_path),
            }
            del parent_payload
        previous_state = {
            name: value.detach().cpu().clone() for name, value in previous.state_dict().items()
        }
        current = deepcopy(previous)
        update_training = _train(
            current,
            workload["update_examples"],
            negative_ids,
            workload["rank_by_item"],
            workload["author_by_item"],
            document,
            "update",
            seed + 2003,
        )
        raw_current_state = {
            name: value.detach().cpu().clone() for name, value in current.state_dict().items()
        }
        publish_alpha = float(document["training"].get("publish_alpha", 1.0))
        if publish_alpha < 1.0:
            with torch.no_grad():
                for name, target in current.state_dict().items():
                    previous_value = previous_state[name].to(target.device)
                    raw_value = raw_current_state[name].to(target.device)
                    target.copy_(
                        previous_value + (raw_value - previous_value) * publish_alpha
                    )
        current_state = {
            name: value.detach().cpu().clone() for name, value in current.state_dict().items()
        }
        raw_parameter_delta = _state_delta(previous_state, raw_current_state)
        parameter_delta = _state_delta(previous_state, current_state)
        theta1 = _checkpoint(
            output_root / "checkpoints" / f"seed_{seed}" / "theta1.pt",
            current,
            {
                "training": update_training,
                "workload": workload["metadata"],
                "parameter_delta": parameter_delta,
                "raw_parameter_delta": raw_parameter_delta,
                "publish_alpha": publish_alpha,
            },
        )
        evaluation = _evaluate(previous, current, workload, document)
        compact = _summary(evaluation, document)
        result_path = output_root / "seeds" / f"seed_{seed}.json"
        _atomic_json(
            result_path,
            {
                "seed": seed,
                "base_training": base_training,
                "update_training": update_training,
                "parameter_delta": parameter_delta,
                "raw_parameter_delta": raw_parameter_delta,
                "publish_alpha": publish_alpha,
                "checkpoints": {"theta0": theta0, "theta1": theta1},
                "summary": compact,
                "records": evaluation["records"],
            },
        )
        seed_results.append(
            {
                "seed": seed,
                "result_path": str(result_path),
                "result_sha256": file_sha256(result_path),
                "summary": compact,
                "checkpoints": {"theta0": theta0, "theta1": theta1},
            }
        )
        stale = compact["comparisons"]["recompute_over_reuse"]
        print(
            f"phase=kuairand_query_seed_complete seed={seed} "
            f"mrr={stale['mrr']['relative_percent']:.3f}% "
            f"ndcg10={stale['ndcg_at_10']['relative_percent']:.3f}% "
            f"hr10={stale['hit_rate_at_10']['relative_percent']:.3f}% "
            f"passed={compact['gate']['passed']}",
            flush=True,
        )
        del previous, current, previous_state, raw_current_state, current_state
        torch.cuda.empty_cache()
    passed_seeds = [value["seed"] for value in seed_results if value["summary"]["gate"]["passed"]]
    summary = {
        "protocol": PROTOCOL,
        "round_id": document["round_id"],
        "status": "complete",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "workload": workload["metadata"],
        "model": {
            **document["model"],
            "parameters": sum(
                parameter.numel()
                for parameter in make_model(
                    document, int(workload["metadata"]["embedding_rows"]), torch.device("cpu")
                ).parameters()
            ),
        },
        "seed_results": seed_results,
        "decision": {
            "passed_seeds": passed_seeds,
            "all_seeds_passed": len(passed_seeds) == len(seed_results),
            "next": "independent_replication_then_capacity_scale"
            if len(passed_seeds) == len(seed_results)
            else "bounded_training_or_objective_diagnosis",
        },
        "elapsed_seconds": time.monotonic() - started,
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
        },
    }
    validate_summary(summary, document)
    _atomic_json(summary_path, summary)
    return summary


def validate_summary(summary: dict[str, Any], document: dict[str, Any]) -> None:
    seeds = summary.get("seed_results")
    if (
        summary.get("protocol") != PROTOCOL
        or summary.get("round_id") != document["round_id"]
        or summary.get("status") != "complete"
        or summary.get("scientific_result") is not False
        or summary.get("formal_result") is not False
        or not isinstance(seeds, list)
        or [value.get("seed") for value in seeds] != document["training"]["seeds"]
        or summary.get("workload", {}).get("target_leakage") is not False
        or summary.get("workload", {}).get("evaluation_records")
        != summary.get("workload", {}).get("selected_users")
    ):
        raise ValueError("KuaiRand prediction-query summary differs")
    for seed in seeds:
        result_path = Path(seed["result_path"])
        if not result_path.is_file() or file_sha256(result_path) != seed["result_sha256"]:
            raise ValueError("KuaiRand prediction-query seed binding differs")
        if not seed.get("summary", {}).get("sanity", {}).get("passed"):
            raise ValueError("KuaiRand prediction-query incremental sanity failed")


def reevaluate(config_path: str | Path) -> dict[str, Any]:
    document = load_config(config_path)
    output_root = Path(document["outputs"]["root"])
    workload = build_workload(document)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("KuaiRand prediction-query reevaluation requires CUDA")
    seed_results = []
    started = time.monotonic()
    for seed in document["training"]["seeds"]:
        seed = int(seed)
        result_path = output_root / "seeds" / f"seed_{seed}.json"
        previous_result = json.loads(result_path.read_text())
        backup_path = (
            output_root
            / "failed_attempts"
            / f"pre_fix_multifield_input_scaling_seed_{seed}.json"
        )
        if not backup_path.is_file():
            _atomic_json(backup_path, previous_result)
        previous = make_model(
            document, int(workload["metadata"]["embedding_rows"]), device
        )
        current = make_model(
            document, int(workload["metadata"]["embedding_rows"]), device
        )
        theta0_path = Path(previous_result["checkpoints"]["theta0"]["path"])
        theta1_path = Path(previous_result["checkpoints"]["theta1"]["path"])
        theta0 = torch.load(theta0_path, map_location=device, weights_only=True)
        theta1 = torch.load(theta1_path, map_location=device, weights_only=True)
        previous.load_state_dict(theta0["state_dict"])
        current.load_state_dict(theta1["state_dict"])
        previous.eval()
        current.eval()
        evaluation = _evaluate(previous, current, workload, document)
        compact = _summary(evaluation, document)
        corrected = {
            **previous_result,
            "summary": compact,
            "records": evaluation["records"],
            "reevaluation": {
                "reason": "per-token video-author scaling corrected for the learned query row",
                "invalid_predecessor": str(backup_path),
                "invalid_predecessor_sha256": file_sha256(backup_path),
            },
        }
        _atomic_json(result_path, corrected)
        seed_results.append(
            {
                "seed": seed,
                "result_path": str(result_path),
                "result_sha256": file_sha256(result_path),
                "summary": compact,
                "checkpoints": previous_result["checkpoints"],
            }
        )
        del previous, current, theta0, theta1
        torch.cuda.empty_cache()
    passed_seeds = [value["seed"] for value in seed_results if value["summary"]["gate"]["passed"]]
    parameter_model = make_model(
        document, int(workload["metadata"]["embedding_rows"]), torch.device("cpu")
    )
    summary = {
        "protocol": PROTOCOL,
        "round_id": document["round_id"],
        "status": "complete",
        "scientific_result": False,
        "formal_result": False,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "workload": workload["metadata"],
        "model": {
            **document["model"],
            "parameters": sum(parameter.numel() for parameter in parameter_model.parameters()),
        },
        "seed_results": seed_results,
        "decision": {
            "passed_seeds": passed_seeds,
            "all_seeds_passed": len(passed_seeds) == len(seed_results),
            "next": "independent_replication_then_capacity_scale"
            if len(passed_seeds) == len(seed_results)
            else "bounded_training_or_objective_diagnosis",
        },
        "elapsed_seconds": time.monotonic() - started,
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
        },
        "reevaluation_only": True,
    }
    del parameter_model
    validate_summary(summary, document)
    _atomic_json(output_root / "summary.json", summary)
    return summary
