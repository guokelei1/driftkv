"""One checked source for the user-fixed history and proposed release chain."""
from __future__ import annotations

import json
from pathlib import Path

from .protocol import version_schedule

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/proposal.json"


def validate_proposal(config: dict) -> None:
    history, architecture, timeline = config["history"], config["architecture"], config["timeline"]
    if history["max_events"] != 1024 or architecture["max_seq_len_including_query"] != 1025:
        raise ValueError("User-fixed history is 1024, plus one transient candidate query")
    if history["minimum_resident_history_at_d26"] != 1024:
        raise ValueError("Resident history cutoff must match the fixed full history window")
    expected = {
        "dataset_timezone": "Asia/Shanghai", "request_timestamp_unit": "milliseconds_since_unix_epoch",
        "interval_convention": "half_open_start_inclusive_end_exclusive",
        "history_only_days": [1, 11], "foundation_observed_feedback_warmup_days": [12, 23],
        "foundation_full_candidate_alignment_days": [24, 26], "foundation_end_day": 26,
        "updates": 9, "update_days": [27, 35], "final_confirmation_day": 37,
    }
    if any(timeline.get(key) != value for key, value in expected.items()):
        raise ValueError("Proposal timeline disagrees with the complete-candidate version schedule")
    intervals = [config["roles"][role] for role in ["train", "validation", "development", "confirmation"]]
    buckets = [bucket for lo, hi in intervals for bucket in range(lo, hi + 1)]
    if len(buckets) != 100 or sorted(buckets) != list(range(100)):
        raise ValueError("UID roles must partition hash buckets 0..99 without overlap")
    if config["metric"]["candidate_subsampling_at_evaluation"]:
        raise ValueError("Evaluation must score all logged stage candidates")
    if architecture["candidate_cross_attention"] or architecture["query_writes_to_persistent_cache"]:
        raise ValueError("Candidate reads must be independent and transient")
    if config["budget"]["foundation_wall_hours_hard_limit"] != 15:
        raise ValueError("Foundation wall-time limit must remain 15 hours")
    for scale in ["pilot", "max"]:
        shape = architecture[scale]
        if shape["heads"] * shape["head_dim"] != shape["hidden_size"] or shape["layers"] < 1:
            raise ValueError(f"Inconsistent backbone dimensions for {scale}")
    if architecture["max"] != {"layers": 16, "hidden_size": 320, "heads": 10, "head_dim": 32,
                               "item_embedding_dimension": 320}:
        raise ValueError("Formal backbone must retain the user-selected RecFlow Max scale")
    if len(version_schedule()) != timeline["updates"] + 1:
        raise ValueError("Version count differs")


def load_proposal(path=DEFAULT_CONFIG) -> dict:
    config = json.loads(Path(path).read_text())
    validate_proposal(config)
    return config


def backbone_config(config: dict, scale: str, *, num_items: int):
    """Explicit num_items supports either a small oracle or an external encoder.

    A feature-aware wrapper must supply external history/candidate vectors;
    this function alone does not implement the proposed catalog feature table.
    """
    from hstu_kvcache.models.hstu import HSTUConfig
    validate_proposal(config)
    architecture = config["architecture"]
    shape = architecture[scale]
    return HSTUConfig(
        num_items=num_items, num_behaviors=4,
        hidden_size=shape["hidden_size"], num_layers=shape["layers"],
        num_heads=shape["heads"], head_dim=shape["head_dim"],
        max_seq_len=architecture["max_seq_len_including_query"],
        block_variant=architecture["block_variant"], activation=architecture["activation"],
        causal_diagonal=architecture["causal_diagonal"],
        relative_position_bias=architecture["relative_position_bias"],
        attn_dropout=architecture["attention_dropout"], input_dropout=architecture["input_dropout"],
    )
