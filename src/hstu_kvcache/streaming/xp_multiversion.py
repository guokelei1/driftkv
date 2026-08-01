from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..data.qk_xp_edge_inputs import ROLE_NAMES, array_sha256
from .trainer import build_next_item_targets
from .xp_projected_edge import XPProjectedModelSpec
from .xp_version_training import (
    XPFixedEdgeCorpus,
    load_xp_fixed_edge_corpus,
)

XP_MULTIVERSION_PROTOCOL = (
    "evokv_xp_multiversion_stream_training_development_v0"
)
XP_PREQUENTIAL_MULTIVERSION_PROTOCOL = (
    "evokv_xp_prequential_stream_training_development_v1"
)
SPLIT_NAMES = ("train", "tuning", "quality")


@dataclass(frozen=True)
class XPLearningRateCandidate:
    name: str
    dense: float
    projection: float
    embedding: float


@dataclass(frozen=True)
class XPUpdateWindow:
    source_version: int
    target_version: int
    history_end: int
    update_end: int

    @property
    def width(self) -> int:
        return self.update_end - self.history_end


@dataclass(frozen=True)
class XPEvaluationWindow:
    model_version: int
    history_end: int
    evaluation_end: int

    @property
    def width(self) -> int:
        return self.evaluation_end - self.history_end


@dataclass(frozen=True)
class XPMultiversionSchedule:
    path: Path
    file_sha256: str
    semantic_sha256: str
    protocol: str
    stack_identity: str
    edge_inputs: Path
    edge_summary: Path
    base_version: int
    split_roles: Mapping[str, str]
    updates: tuple[XPUpdateWindow, ...]
    evaluation_windows: tuple[XPEvaluationWindow, ...]
    learning_rate_candidates: tuple[XPLearningRateCandidate, ...]
    fixed_learning_rate_name: str | None
    admission_policy: str
    minimum_tuning_cross_entropy_reduction: float
    epochs_per_update: int
    weight_decay: float
    train_negatives: int
    tuning_negatives: int
    quality_negatives: int
    training_seed: int
    tuning_seed: int
    quality_seed: int
    document: Mapping[str, object]


@dataclass(frozen=True)
class XPValidatedMultiversionCorpus:
    schedule: XPMultiversionSchedule
    corpus: XPFixedEdgeCorpus
    audit: Mapping[str, object]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _semantic_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _resolve_path(base: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"XP multiversion {name} is invalid")
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"XP multiversion {name} is invalid")
    return value


def _nonnegative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"XP multiversion {name} is invalid")
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"XP multiversion {name} is invalid")
    return result


def load_xp_multiversion_schedule(
    path: str | Path,
) -> XPMultiversionSchedule:
    resolved = Path(path).resolve()
    document = json.loads(resolved.read_text())
    protocol = document.get("protocol") if isinstance(document, dict) else None
    if (
        not isinstance(document, dict)
        or protocol
        not in {
            XP_MULTIVERSION_PROTOCOL,
            XP_PREQUENTIAL_MULTIVERSION_PROTOCOL,
        }
        or document.get("scientific_result") is not False
        or document.get("formal_result") is not False
        or document.get("quality_used_for_selection") is not False
        or document.get("quality_controls_training") is not False
    ):
        raise ValueError("XP multiversion schedule contract differs")
    stack_identity = document.get("stack_identity")
    if not isinstance(stack_identity, str) or not stack_identity:
        raise ValueError("XP multiversion stack identity is invalid")
    split_roles = document.get("split_roles")
    if (
        not isinstance(split_roles, dict)
        or set(split_roles) != set(SPLIT_NAMES)
        or any(
            not isinstance(split_roles[name], str)
            or split_roles[name] not in ROLE_NAMES
            for name in SPLIT_NAMES
        )
        or len(set(split_roles.values())) != len(SPLIT_NAMES)
    ):
        raise ValueError("XP multiversion split roles differ")
    base_version = document.get("base_version")
    if (
        isinstance(base_version, bool)
        or not isinstance(base_version, int)
        or base_version < 0
    ):
        raise ValueError("XP multiversion base version is invalid")
    raw_updates = document.get("updates")
    minimum_updates = (
        4 if protocol == XP_MULTIVERSION_PROTOCOL else 3
    )
    if (
        not isinstance(raw_updates, list)
        or len(raw_updates) < minimum_updates
    ):
        raise ValueError(
            f"XP multiversion requires at least {minimum_updates} updates"
        )
    updates = []
    previous_target = base_version
    previous_end = None
    for index, raw in enumerate(raw_updates):
        if not isinstance(raw, dict):
            raise ValueError("XP multiversion update is invalid")
        source_version = raw.get("source_version")
        target_version = raw.get("target_version")
        history_end = raw.get("history_end")
        update_end = raw.get("update_end")
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (
                    source_version,
                    target_version,
                    history_end,
                    update_end,
                )
            )
            or source_version != previous_target
            or target_version != source_version + 1
            or history_end < 1
            or update_end <= history_end
            or (previous_end is not None and history_end != previous_end)
        ):
            raise ValueError(
                f"XP multiversion update {index} is not contiguous"
            )
        updates.append(
            XPUpdateWindow(
                source_version=source_version,
                target_version=target_version,
                history_end=history_end,
                update_end=update_end,
            )
        )
        previous_target = target_version
        previous_end = update_end
    if protocol == XP_MULTIVERSION_PROTOCOL:
        evaluation_windows = tuple(
            XPEvaluationWindow(
                model_version=update.target_version,
                history_end=update.history_end,
                evaluation_end=update.update_end,
            )
            for update in updates
        )
        policy = document.get("learning_rate_screen")
        if (
            not isinstance(policy, dict)
            or policy.get("edge_index") != 0
            or policy.get("selection_role") != "tuning"
            or policy.get("primary_metric")
            != "sampled_cross_entropy_reduction"
            or policy.get("quality_observed_during_screen") is not False
        ):
            raise ValueError("XP multiversion LR screen contract differs")
        raw_candidates = policy.get("candidates")
        if not isinstance(raw_candidates, list) or len(raw_candidates) < 2:
            raise ValueError("XP multiversion LR screen is too small")
        fixed_learning_rate_name = None
        admission_policy = "positive_tuning_cross_entropy_reduction"
        minimum_tuning_reduction = _nonnegative_float(
            policy.get("minimum_cross_entropy_reduction"),
            "minimum tuning reduction",
        )
    else:
        if (
            document.get("tuning_used_for_selection") is not False
            or document.get("tuning_controls_training") is not False
            or document.get("evaluation_semantics")
            != "next_unseen_window"
        ):
            raise ValueError("XP prequential observation contract differs")
        raw_evaluations = document.get("prequential_evaluations")
        if (
            not isinstance(raw_evaluations, list)
            or len(raw_evaluations) != len(updates) + 1
        ):
            raise ValueError("XP prequential evaluation count differs")
        parsed_evaluations = []
        for index, raw in enumerate(raw_evaluations):
            if not isinstance(raw, dict):
                raise ValueError("XP prequential evaluation is invalid")
            model_version = raw.get("model_version")
            history_end = raw.get("history_end")
            evaluation_end = raw.get("evaluation_end")
            expected_history_end = (
                updates[0].history_end
                if index == 0
                else updates[index - 1].update_end
            )
            expected_evaluation_end = (
                updates[index].update_end
                if index < len(updates)
                else None
            )
            if (
                any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in (
                        model_version,
                        history_end,
                        evaluation_end,
                    )
                )
                or model_version != base_version + index
                or history_end != expected_history_end
                or evaluation_end <= history_end
                or (
                    expected_evaluation_end is not None
                    and evaluation_end != expected_evaluation_end
                )
            ):
                raise ValueError(
                    f"XP prequential evaluation {index} is not the next window"
                )
            parsed_evaluations.append(
                XPEvaluationWindow(
                    model_version=model_version,
                    history_end=history_end,
                    evaluation_end=evaluation_end,
                )
            )
        evaluation_windows = tuple(parsed_evaluations)
        policy = document.get("learning_rate_policy")
        if (
            not isinstance(policy, dict)
            or policy.get("mode") != "predeclared_fixed"
            or policy.get("selection_role") != "none"
            or policy.get("quality_observed_for_selection") is not False
        ):
            raise ValueError("XP prequential LR policy differs")
        raw_candidates = policy.get("candidates")
        fixed_learning_rate_name = policy.get("selected_candidate")
        if (
            not isinstance(raw_candidates, list)
            or len(raw_candidates) < 1
            or not isinstance(fixed_learning_rate_name, str)
            or not fixed_learning_rate_name
        ):
            raise ValueError("XP prequential fixed LR is invalid")
        admission_policy = document.get("checkpoint_admission")
        if admission_policy != "numerical_stability_and_nonzero_update":
            raise ValueError("XP prequential admission policy differs")
        minimum_tuning_reduction = 0.0
    candidates = []
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            raise ValueError("XP multiversion LR candidate is invalid")
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("XP multiversion LR candidate name is invalid")
        rates = []
        for field in ("dense", "projection", "embedding"):
            value = raw.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(
                    f"XP multiversion LR candidate {field} is invalid"
                )
            rates.append(float(value))
        candidates.append(
            XPLearningRateCandidate(name, rates[0], rates[1], rates[2])
        )
    if len({value.name for value in candidates}) != len(candidates):
        raise ValueError("XP multiversion LR candidate names overlap")
    if len(
        {
            (value.dense, value.projection, value.embedding)
            for value in candidates
        }
    ) != len(candidates):
        raise ValueError("XP multiversion LR candidates overlap")
    if (
        fixed_learning_rate_name is not None
        and fixed_learning_rate_name
        not in {candidate.name for candidate in candidates}
    ):
        raise ValueError("XP prequential selected LR is absent")
    training = document.get("training")
    if not isinstance(training, dict):
        raise ValueError("XP multiversion training contract is absent")
    seeds = training.get("seeds")
    if (
        not isinstance(seeds, dict)
        or set(seeds) != {"training", "tuning", "quality"}
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in seeds.values()
        )
        or len(set(seeds.values())) != 3
    ):
        raise ValueError("XP multiversion seeds differ")
    return XPMultiversionSchedule(
        path=resolved,
        file_sha256=_file_sha256(resolved),
        semantic_sha256=_semantic_sha256(document),
        protocol=protocol,
        stack_identity=stack_identity,
        edge_inputs=_resolve_path(
            resolved.parent,
            document.get("edge_inputs"),
            "edge inputs",
        ),
        edge_summary=_resolve_path(
            resolved.parent,
            document.get("edge_summary"),
            "edge summary",
        ),
        base_version=base_version,
        split_roles={name: split_roles[name] for name in SPLIT_NAMES},
        updates=tuple(updates),
        evaluation_windows=evaluation_windows,
        learning_rate_candidates=tuple(candidates),
        fixed_learning_rate_name=fixed_learning_rate_name,
        admission_policy=admission_policy,
        minimum_tuning_cross_entropy_reduction=minimum_tuning_reduction,
        epochs_per_update=_positive_int(
            training.get("epochs_per_update"),
            "epochs per update",
        ),
        weight_decay=_nonnegative_float(
            training.get("weight_decay"),
            "weight decay",
        ),
        train_negatives=_positive_int(
            training.get("train_negatives"),
            "train negatives",
        ),
        tuning_negatives=_positive_int(
            training.get("tuning_negatives"),
            "tuning negatives",
        ),
        quality_negatives=_positive_int(
            training.get("quality_negatives"),
            "quality negatives",
        ),
        training_seed=seeds["training"],
        tuning_seed=seeds["tuning"],
        quality_seed=seeds["quality"],
        document=document,
    )


def validate_xp_multiversion_corpus(
    schedule: XPMultiversionSchedule,
    spec: XPProjectedModelSpec,
) -> XPValidatedMultiversionCorpus:
    corpus = load_xp_fixed_edge_corpus(
        schedule.edge_inputs,
        schedule.edge_summary,
        num_embeddings=spec.num_embeddings,
        num_prediction_items=spec.num_prediction_items,
        num_behaviors=spec.num_behaviors,
    )
    split_users = {
        split: corpus.arrays["record_user_ids"][
            corpus.role_records(role)
        ]
        for split, role in schedule.split_roles.items()
    }
    combined = np.concatenate(list(split_users.values()))
    if len(np.unique(combined)) != len(combined):
        raise ValueError("XP multiversion split users overlap")
    maximum_end = max(
        max(value.update_end for value in schedule.updates),
        max(
            value.evaluation_end
            for value in schedule.evaluation_windows
        ),
    )
    available = {}
    for split, role in schedule.split_roles.items():
        records = corpus.role_records(role)
        lengths = np.diff(corpus.arrays["record_offsets"])[records]
        if len(records) == 0 or np.any(lengths < maximum_end):
            raise ValueError(
                f"XP multiversion {split} records do not cover all windows"
            )
        available[split] = {
            "source_role": role,
            "users": len(records),
            "user_ids_sha256": array_sha256(split_users[split]),
            "minimum_materialized_events": int(lengths.min()),
            "maximum_materialized_events": int(lengths.max()),
        }
    return XPValidatedMultiversionCorpus(
        schedule=schedule,
        corpus=corpus,
        audit={
            "stack_identity": schedule.stack_identity,
            "schedule_semantic_sha256": schedule.semantic_sha256,
            "dataset": corpus.metadata["dataset"],
            "edge_input_content_sha256": corpus.content_sha256,
            "split_users_pairwise_disjoint": True,
            "split_users": available,
            "windows_contiguous_and_nonoverlapping": True,
            "windows": [
                {
                    "source_version": value.source_version,
                    "target_version": value.target_version,
                    "history_end": value.history_end,
                    "update_end": value.update_end,
                    "update_width": value.width,
                }
                for value in schedule.updates
            ],
            "prequential_evaluations": [
                {
                    "model_version": value.model_version,
                    "history_end": value.history_end,
                    "evaluation_end": value.evaluation_end,
                    "evaluation_width": value.width,
                }
                for value in schedule.evaluation_windows
            ],
            "evaluation_semantics": (
                "same_update_window"
                if schedule.protocol == XP_MULTIVERSION_PROTOCOL
                else "next_unseen_window"
            ),
            "quality_used_for_selection": False,
            "quality_controls_training": False,
            "legacy_two_edge_result_compatible": False,
        },
    )


def _record_has_window_target(
    corpus: XPFixedEdgeCorpus,
    record: int,
    history_end: int,
    update_end: int,
) -> bool:
    offset = int(corpus.arrays["record_offsets"][record])
    labels = corpus.arrays["label"][
        offset + history_end : offset + update_end
    ]
    sources = corpus.arrays["item_idx"][
        offset + history_end - 1 : offset + update_end - 1
    ]
    return bool(np.any((labels > 0) & (sources > 0)))


def _materialize_window_record(
    corpus: XPFixedEdgeCorpus,
    record: int,
    history_end: int,
    update_end: int,
    max_seq_len: int,
) -> dict[str, torch.Tensor]:
    offset = int(corpus.arrays["record_offsets"][record])
    window_start = max(0, update_end - max_seq_len)
    start = offset + window_start
    stop = offset + update_end
    width = stop - start
    ordinals = corpus.arrays["raw_ordinal"][start:stop].astype(
        np.float32,
        copy=True,
    )
    time_deltas = np.empty(width, dtype=np.float32)
    if window_start == 0:
        time_deltas[0] = 0.0
    else:
        time_deltas[0] = (
            ordinals[0]
            - float(corpus.arrays["raw_ordinal"][start - 1])
        )
    time_deltas[1:] = np.diff(ordinals)
    train_mask = np.zeros(width, dtype=np.bool_)
    train_mask[
        history_end - window_start : update_end - window_start
    ] = True
    return {
        "item_ids": torch.from_numpy(
            corpus.arrays["item_idx"][start:stop].astype(
                np.int64,
                copy=True,
            )
        ),
        "behaviors": torch.from_numpy(
            corpus.arrays["behavior"][start:stop].astype(
                np.int64,
                copy=True,
            )
        ),
        "time_deltas": torch.from_numpy(time_deltas),
        "labels": torch.from_numpy(
            corpus.arrays["label"][start:stop].astype(
                np.int64,
                copy=True,
            )
        ),
        "train_mask": torch.from_numpy(train_mask),
        "length": torch.tensor(width, dtype=torch.int64),
        "record": torch.tensor(record, dtype=torch.int64),
        "window_start": torch.tensor(window_start, dtype=torch.int64),
    }


def build_window_batches(
    corpus: XPFixedEdgeCorpus,
    role: str,
    update: XPUpdateWindow,
    *,
    max_seq_len: int,
    batch_size_per_rank: int,
    rank: int,
    world_size: int,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, object]]:
    if (
        role not in ROLE_NAMES
        or max_seq_len < 2
        or batch_size_per_rank < 1
        or world_size < 1
        or not 0 <= rank < world_size
    ):
        raise ValueError("XP multiversion batch request differs")
    records = corpus.role_records(role)
    eligible = np.asarray(
        [
            int(record)
            for record in records
            if _record_has_window_target(
                corpus,
                int(record),
                update.history_end,
                update.update_end,
            )
        ],
        dtype=np.int64,
    )
    if len(eligible) == 0:
        raise RuntimeError(
            f"XP multiversion role {role} has no window targets"
        )
    global_batch = batch_size_per_rank * world_size
    steps = (len(eligible) + global_batch - 1) // global_batch
    materialized = {
        int(record): _materialize_window_record(
            corpus,
            int(record),
            update.history_end,
            update.update_end,
            max_seq_len,
        )
        for record in eligible
    }
    width = min(update.update_end, max_seq_len)
    batches = []
    local_real_records = 0
    local_targets = 0
    local_tokens = 0
    for step in range(steps):
        left = step * global_batch + rank * batch_size_per_rank
        right = min(left + batch_size_per_rank, len(eligible))
        selected = eligible[left:right]
        values = [materialized[int(record)] for record in selected]
        local_real_records += len(values)
        while len(values) < batch_size_per_rank:
            values.append(
                {
                    "item_ids": torch.zeros(width, dtype=torch.int64),
                    "behaviors": torch.zeros(width, dtype=torch.int64),
                    "time_deltas": torch.zeros(
                        width,
                        dtype=torch.float32,
                    ),
                    "labels": torch.zeros(width, dtype=torch.int64),
                    "train_mask": torch.zeros(width, dtype=torch.bool),
                    "length": torch.tensor(0, dtype=torch.int64),
                    "record": torch.tensor(-1, dtype=torch.int64),
                    "window_start": torch.tensor(0, dtype=torch.int64),
                }
            )
        batch = {
            name: torch.stack([value[name] for value in values])
            for name in (
                "item_ids",
                "behaviors",
                "time_deltas",
                "labels",
                "train_mask",
            )
        }
        batch["lengths"] = torch.stack(
            [value["length"] for value in values]
        )
        batch["record_indices"] = torch.stack(
            [value["record"] for value in values]
        )
        batch["window_starts"] = torch.stack(
            [value["window_start"] for value in values]
        )
        _, valid = build_next_item_targets(
            batch["item_ids"],
            batch["lengths"],
            batch["labels"],
            batch["train_mask"],
        )
        local_targets += int(valid.sum().item())
        local_tokens += int(batch["lengths"].sum().item())
        batches.append(batch)
    digest = hashlib.sha256()
    digest.update(np.asarray(eligible, dtype="<i8").tobytes())
    digest.update(
        np.asarray(
            [update.history_end, update.update_end],
            dtype="<i8",
        ).tobytes()
    )
    for record in eligible:
        offset = int(corpus.arrays["record_offsets"][record])
        digest.update(
            np.asarray(
                corpus.arrays["item_idx"][
                    offset + update.history_end :
                    offset + update.update_end
                ],
                dtype="<u4",
            ).tobytes()
        )
        digest.update(
            np.asarray(
                corpus.arrays["label"][
                    offset + update.history_end :
                    offset + update.update_end
                ],
                dtype=np.uint8,
            ).tobytes()
        )
    return batches, {
        "role": role,
        "source_version": update.source_version,
        "target_version": update.target_version,
        "history_end": update.history_end,
        "update_end": update.update_end,
        "update_width": update.width,
        "global_records": len(records),
        "global_eligible_records": len(eligible),
        "global_zero_target_records_removed": len(records) - len(eligible),
        "steps_per_rank": steps,
        "batch_size_per_rank": batch_size_per_rank,
        "local_real_records": local_real_records,
        "local_padding_records": (
            steps * batch_size_per_rank - local_real_records
        ),
        "local_tokens": local_tokens,
        "local_targets": local_targets,
        "maximum_model_context": max_seq_len,
        "physical_sequence_width": width,
        "causal_window_start": max(0, update.update_end - max_seq_len),
        "window_targets_sha256": digest.hexdigest(),
    }


def qualification_signal(
    before: Mapping[str, float | int],
    after: Mapping[str, float | int],
    minimum_cross_entropy_reduction: float = 0.0,
) -> dict[str, object]:
    reduction = float(before["sampled_cross_entropy"]) - float(
        after["sampled_cross_entropy"]
    )
    return {
        "primary_metric": "sampled_cross_entropy_reduction",
        "before": dict(before),
        "after": dict(after),
        "sampled_cross_entropy_reduction": reduction,
        "minimum_cross_entropy_reduction": (
            minimum_cross_entropy_reduction
        ),
        "ndcg_at_10_delta": float(after["ndcg_at_10"])
        - float(before["ndcg_at_10"]),
        "hit_rate_at_10_delta": float(after["hit_rate_at_10"])
        - float(before["hit_rate_at_10"]),
        "mean_reciprocal_rank_delta": float(
            after["mean_reciprocal_rank"]
        )
        - float(before["mean_reciprocal_rank"]),
        "positive_signal_gate_passed": (
            reduction > minimum_cross_entropy_reduction
        ),
    }


def select_learning_rate_candidate(
    candidates: Sequence[XPLearningRateCandidate],
    reports: Sequence[Mapping[str, object]],
    minimum_cross_entropy_reduction: float,
) -> XPLearningRateCandidate | None:
    if len(candidates) != len(reports):
        raise ValueError("XP LR screen report count differs")
    values = []
    for index, (candidate, report) in enumerate(
        zip(candidates, reports, strict=True)
    ):
        if report.get("candidate") != candidate.name:
            raise ValueError("XP LR screen report order differs")
        signal = report.get("tuning_signal")
        if not isinstance(signal, Mapping):
            raise ValueError("XP LR screen tuning signal is absent")
        reduction = float(signal["sampled_cross_entropy_reduction"])
        if not np.isfinite(reduction):
            raise ValueError("XP LR screen tuning signal is not finite")
        values.append((reduction, -index, candidate))
    winner = max(values, key=lambda value: (value[0], value[1]))
    return (
        winner[2]
        if winner[0] > minimum_cross_entropy_reduction
        else None
    )
