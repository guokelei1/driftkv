#!/usr/bin/env python3
"""Prepare frozen causal IDs/deltas for a Large or Max binary-feedback model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from hstu_kvcache.data.oov import stable_oov_bucket


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scale", choices=["large", "max"], default="large")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.output = args.output or Path(f"data/processed/recflow_feedback_{args.scale}_v1")
    cfg = json.loads(args.config.read_text())
    source = Path(cfg["data_root"])
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Prepared feedback inputs already exist; reuse their manifest")
    args.output.mkdir(parents=True, exist_ok=True)
    events = np.load(source / "events.npy", mmap_mode="r")
    req = np.load(source / "requests.npy", mmap_mode="r")
    catalog = np.load(source / "catalog.npz")
    valid = (req["role"] == 0) & (req["day"] <= 18) & (req["history_stop"] > req["history_start"])
    event_mask = np.repeat(valid, req["event_stop"] - req["event_start"])
    size = len(catalog["raw_item_ids"])
    support = np.bincount(events["item"][event_mask], minlength=size + 2)[1:-1]
    k = cfg["scales"][args.scale]["known_videos"]
    selected = np.lexsort((catalog["raw_item_ids"], -support))[:k]
    known = np.sort(catalog["raw_item_ids"][selected])
    assert support[selected].min() > 0
    np.save(args.output / "known_ids.npy", known)
    unique, inverse = np.unique(events["raw_item_id"], return_inverse=True)
    mapped = np.searchsorted(known, unique)
    hit = mapped < len(known)
    hit[hit] &= known[mapped[hit]] == unique[hit]
    mapped[hit] += 1
    buckets = cfg["catalog"]["oov_buckets"]
    missing = np.flatnonzero(~hit)
    for offset in range(0, len(missing), 1000000):
        part = missing[offset:offset + 1000000]
        mapped[part] = [stable_oov_bucket(int(raw), known_vocab_size=k, buckets=buckets, bucket_start=k + 1) for raw in unique[part]]
        print(json.dumps(dict(hashed_ids=min(offset + len(part), len(missing)), total=len(missing))), flush=True)
    np.save(args.output / "items.npy", mapped[inverse].astype(np.int32))
    del mapped, inverse, unique
    deltas = np.zeros(len(events), dtype=np.float32)
    same_user = events["uid"][1:] == events["uid"][:-1]
    deltas[1:] = np.where(same_user, np.diff(events["ts"]) / 1000, 0)
    assert (deltas >= 0).all()
    np.save(args.output / "deltas.npy", deltas)
    inputs = {name: dict(bytes=(source / name).stat().st_size, sha256=sha(source / name)) for name in ["events.npy", "requests.npy", "population.npz", "catalog.npz", "manifest.json"]}
    outputs = {name: dict(bytes=(args.output / name).stat().st_size, sha256=sha(args.output / name)) for name in ["known_ids.npy", "items.npy", "deltas.npy"]}
    manifest = dict(status="frozen_feedback_inputs", scale=args.scale, source_root=str(source), known_videos=k, oov_buckets=buckets,
        source_inputs=inputs, outputs=outputs, source_config_sha256=sha(args.config), events=len(events), requests=len(req),
        known_ids_sha256=hashlib.sha256(known.astype("<i4").tobytes()).hexdigest(),
        causality="original request history_stop excludes entire same-timestamp block; deltas preserve original predecessor even after FIFO")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
