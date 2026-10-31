"""Vectorized mmap gathers; CPU workers never initialize CUDA or copy catalogs."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_common import stable_hash
from .sampling import stratified_indices


class RequestBatches(Dataset):
    def __init__(self, root, days, *, role=0, pilot=False, first_only=False, limit=None,
                 allow_confirmation=False, history_limit=1024):
        if role == 3 and not allow_confirmation:
            raise ValueError("Confirmation metrics stay sealed until the recipe is frozen")
        if history_limit not in [128, 1024]:
            raise ValueError("1024 is fixed; 128 is a declared inference diagnostic only")
        self.root, self.history_limit = Path(root), history_limit
        manifest = json.loads((self.root / "stage_index_manifest.json").read_text())
        self.stage_dir = manifest["stage_dir"]
        population = np.load(self.root / "history/population.npz")
        parts = []
        for day in days:
            directory = "exposures" if day < 24 else self.stage_dir
            records = np.load(self.root / directory / f"d{day:02d}/index.npy", mmap_mode="r")
            selected = (records["role"] == role) & records["resident"] & records["valid_request"]
            if role == 0:
                selected &= (records["observed"] > 0) & (records["history_length"] > 0)
                if pilot:
                    selected &= np.isin(records["uid"], population["pilot_train_uids"])
            if first_only:
                selected &= records["first_request"]
            parts.append(records[selected])
        self.records = np.concatenate(parts)
        if limit is not None and len(self.records) > limit:
            if role == 0:
                selected = stratified_indices(self.records, limit)
            else:
                hashed = stable_hash(self.records["rid"], 9427)
                selected = np.argsort(hashed, kind="stable")[:limit]
            self.records = self.records[np.sort(selected)]
        self._events, self._daily = None, {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_events"], state["_daily"] = None, {}
        return state

    def __len__(self):
        return len(self.records)

    def __getitem__(self, indices):
        indices = np.asarray(indices, np.int64)
        padding = indices < 0
        safe = np.maximum(indices, 0)
        req = self.records[safe]
        if self._events is None:
            self._events = np.load(self.root / "history/events.npy", mmap_mode="r")
        events = self._events
        lengths = np.minimum(req["history_length"].astype(np.int64), self.history_limit)
        offsets = np.arange(self.history_limit)[None, :]
        valid = offsets < lengths[:, None]
        locations = (req["history_stop"]-lengths)[:, None] + offsets
        locations = np.clip(locations, 0, len(events)-1)
        hist = events[locations]
        history_features = hist["features"].copy()
        history_features[~valid] = -1
        behaviors = np.where(valid, hist["behavior"].astype(np.int64)+1, 0)
        times = hist["ts"]
        deltas = np.zeros(times.shape, np.float32)
        deltas[:, 1:] = np.maximum(0, np.diff(times, axis=1))/1000
        deltas[~valid] = 0
        gap = np.zeros(len(req), np.float32)
        nonempty = lengths > 0
        gap[nonempty] = (req["ts"][nonempty]-events["ts"][req["history_stop"][nonempty]-1])/1000
        counts = (req["stop"]-req["start"]).astype(np.int64)
        width = max(1, math.ceil(int(counts.max())/32)*32)
        features = np.full((len(req), width, 5), -1, np.int32)
        labels = np.full((len(req), width), -1, np.int64)
        columns = np.arange(width)[None, :]
        candidate_mask = columns < counts[:, None]
        for day in np.unique(req["day"]):
            if day not in self._daily:
                directory = "exposures" if day < 24 else self.stage_dir
                path = self.root / directory / f"d{day:02d}"
                if day < 24:
                    rows = np.load(path / "events.npy", mmap_mode="r")
                    self._daily[day] = (rows["features"], rows["label"])
                else:
                    self._daily[day] = (np.load(path / "features.npy", mmap_mode="r"),
                                        np.load(path / "labels.npy", mmap_mode="r"))
            feat, lab = self._daily[day]
            local = np.flatnonzero(req["day"] == day)
            locs = req["start"][local, None] + columns
            keep = candidate_mask[local]
            locs = np.minimum(locs, len(lab)-1)
            features[local] = np.where(keep[..., None], feat[locs], -1)
            labels[local] = np.where(keep, lab[locs], -1)
        candidate_mask[padding] = False
        # Padding-only DDP examples have a connected zero loss and no weight in
        # the global normalization; their actual input never affects metrics.
        result = {"history_features": history_features, "behaviors": behaviors,
                  "deltas": deltas, "lengths": lengths, "query_deltas": gap,
                  "candidate_features": features, "candidate_mask": candidate_mask, "labels": labels,
                  "uid": req["uid"].copy(), "rid": req["rid"].copy(), "day": req["day"].copy(),
                  "counts": counts, "padding": padding}
        return {key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in result.items()}


class GlobalBatchSampler:
    """Identical global order across 1/2/4 ranks; no duplicate training targets."""
    def __init__(self, count, *, global_batch=64, micro_batch=4, world_size=1, rank=0,
                 seed=17, epoch=0, shuffle=True, start_step=0, stop_step=None):
        if global_batch % (micro_batch*world_size):
            raise ValueError("Global batch must divide into rank-local microbatches")
        self.count, self.global_batch = count, global_batch
        self.micro_batch, self.world_size, self.rank = micro_batch, world_size, rank
        self.accumulation = global_batch//(micro_batch*world_size)
        self.order = np.arange(count)
        if shuffle:
            np.random.default_rng(seed+epoch).shuffle(self.order)
        self.start = start_step
        self.stop = min(math.ceil(count/global_batch), stop_step) if stop_step is not None else math.ceil(count/global_batch)

    def __iter__(self):
        for step in range(self.start, self.stop):
            for micro in range(self.accumulation):
                start = step*self.global_batch + (micro*self.world_size+self.rank)*self.micro_batch
                batch = self.order[start:min(start+self.micro_batch, (step+1)*self.global_batch)]
                yield np.pad(batch, (0, self.micro_batch-len(batch)), constant_values=-1).tolist()

    def __len__(self):
        return max(0, self.stop-self.start)*self.accumulation
