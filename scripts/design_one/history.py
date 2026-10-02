"""Read the exact bounded histories prepared by the checked UID semi-join."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hstu_kvcache.training.foundation import FoundationHistoryIndex
from read_correction_v4.evaluate_full import signature
from unified_reuse_2026_09.common import sha256


def history_binding(directory, panel, groups):
    path = directory.resolve() / "binding.json"
    record = json.loads(path.read_text())
    inputs = record["inputs"]
    if (record["status"] != "complete" or signature(inputs) != record["cache_key"]
            or inputs["dataset"] != panel["sources"]["dataset"]
            or inputs["configuration"] != panel["configuration"]
            or inputs["users_file"] != panel["users_file"]
            or inputs["cutover"] != panel["cutover"]
            or inputs["end_timestamp"] != panel["days"][1] * 86400
            or inputs["history_length"] != 1024
            or inputs["uids"] != sorted(groups["design1_evaluation"])):
        raise RuntimeError("history packs differ from the frozen benchmark inputs")
    return {"path": str(path), "sha256": sha256(path), "cache_key": record["cache_key"],
        "packs": record["packs"], "known_vocab_size": inputs["known_vocab_size"],
        "oov_buckets": inputs["oov_buckets"], "preparation_source": record["preparation_source"]}


def load_pack(binding, requested_uids, *, known_vocab_size, oov_buckets):
    """Restore any selected users, including a pilot spanning all four packs."""
    path = Path(binding["path"])
    if sha256(path) != binding["sha256"]:
        raise RuntimeError("history pack metadata changed after input binding")
    record = json.loads(path.read_text())
    if (record["cache_key"] != binding["cache_key"]
            or known_vocab_size != binding["known_vocab_size"]
            or oov_buckets != binding["oov_buckets"]):
        raise RuntimeError("history pack item vocabulary or cache key differs")
    requested, rows = set(requested_uids), {}
    for uids, pack in zip(record["inputs"]["uid_shards"], binding["packs"], strict=True):
        selected = requested.intersection(uids)
        if not selected:
            continue
        pack_path = path.parent / pack["filename"]
        if sha256(pack_path) != pack["sha256"]:
            raise RuntimeError("history pack bytes changed")
        with np.load(pack_path, allow_pickle=False) as arrays:
            saved_uids, offsets = arrays["uids"], arrays["offsets"]
            times, items, behaviors = arrays["timestamps"], arrays["item_ids"], arrays["behaviors"]
        if saved_uids.tolist() != uids:
            raise RuntimeError("history pack UID index differs")
        for index, uid in enumerate(saved_uids):
            uid = int(uid)
            if uid in selected:
                start, end = int(offsets[index]), int(offsets[index + 1])
                rows[uid] = (times[start:end], items[start:end], behaviors[start:end])
    if set(rows) != requested:
        raise RuntimeError("history packs do not contain every requested user")
    return FoundationHistoryIndex(rows)
