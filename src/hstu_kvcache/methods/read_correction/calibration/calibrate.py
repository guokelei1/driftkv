"""Shared causal snapshot capture and candidate sampling; the v1 run is retired."""
from collections import defaultdict
import json
import numpy as np
import torch
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.methods.read_correction.core import score_corrected
from hstu_kvcache.adaptation.reader import history_read
from hstu_kvcache.methods.baselines.protocol.evaluate import native_backend


def candidates(uid, known, count):
    return np.random.default_rng(np.random.SeedSequence([17, int(uid)])).choice(
        known - 1, count, replace=False) + 1


def groups(uids, rows, batch_size):
    lengths = defaultdict(list)
    for uid in uids:
        lengths[rows[uid]["parent"].seq_len].append(uid)
    for _, group in sorted(lengths.items()):
        for start in range(0, len(group), batch_size):
            yield group[start:start + batch_size]


def stacked(rows, uids, key, device):
    return HSTUKVCache(
        torch.cat([rows[uid][key].k for uid in uids], 1).to(device),
        torch.cat([rows[uid][key].v for uid in uids], 1).to(device),
        rows[uids[0]][key].seq_len,
    )


@torch.no_grad()
def capture(parent, current, histories, uids, *, cutover, known, queries, device,
            batch_size, history_length, attention_backend):
    """CPU retains a bounded pair of caches per fitting user; GPU holds one batch."""
    by_length, rows = defaultdict(list), {}
    for uid in uids:
        by_length[len(histories[uid][0])].append(uid)
    for length, group in sorted(by_length.items()):
        backend = attention_backend if length == history_length else "torch"
        for start in range(0, len(group), batch_size):
            selected = group[start:start + batch_size]
            raw = [histories[uid] for uid in selected]
            timestamps = torch.as_tensor(np.stack([x[0] for x in raw]), device=device)
            items = torch.as_tensor(np.stack([x[1] for x in raw]), dtype=torch.long, device=device)
            behaviors = torch.as_tensor(np.stack([x[2] for x in raw]), dtype=torch.long, device=device)
            deltas = torch.zeros_like(timestamps, dtype=torch.float32)
            deltas[:, 1:] = timestamps[:, 1:] - timestamps[:, :-1]
            with native_backend([parent, current], backend):
                old, full = parent.compute_kv(items, behaviors, deltas), current.compute_kv(items, behaviors, deltas)
            for index, uid in enumerate(selected):
                rows[uid] = {
                    "parent": HSTUKVCache(old.k[:, index:index+1].cpu(), old.v[:, index:index+1].cpu(), length),
                    "teacher": HSTUKVCache(full.k[:, index:index+1].cpu(), full.v[:, index:index+1].cpu(), length),
                    "candidates": torch.tensor(candidates(uid, known, queries), dtype=torch.long),
                    "query_delta": float(cutover - histories[uid][0][-1]),
                }
            print(json.dumps({"status": "cache_capture", "users": len(rows), "total": len(uids)}), flush=True)
            del old, full
    return rows


@torch.no_grad()
def layer_targets(current, rows, uids, modules, layer, *, batch_size, device):
    """Teacher sees the actual already-corrected branch query at this layer."""
    collected = {}
    for selected in groups(uids, rows, batch_size):
        cache = stacked(rows, selected, "parent", device)
        teacher_k = torch.cat([rows[uid]["teacher"].k[layer] for uid in selected], 0).to(device)
        teacher_v = torch.cat([rows[uid]["teacher"].v[layer] for uid in selected], 0).to(device)
        counts = torch.full((len(selected),), cache.seq_len, device=device, dtype=torch.long)
        ids = torch.stack([rows[uid]["candidates"] for uid in selected]).to(device)
        dt = torch.tensor([rows[uid]["query_delta"] for uid in selected], device=device)
        _, trace = score_corrected(current, cache, ids, dt, modules, counts, trace=True)
        query = trace.queries[layer]
        target = (history_read(current.blocks[layer].attn, query, teacher_k, teacher_v)
                  - trace.history_heads[layer]) / counts[:, None, None, None]
        for index, uid in enumerate(selected):
            collected[uid] = (query[index].cpu(), target[index].cpu())
    return (torch.stack([collected[uid][0] for uid in uids]),
            torch.stack([collected[uid][1] for uid in uids]))
