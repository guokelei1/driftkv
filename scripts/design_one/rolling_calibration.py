"""Pre-release calibration scenes with real writes, evictions and birth context."""

from collections import defaultdict

import numpy as np
import torch

from design_one.calibrate import DAY, terminal_tuple_hash
from hstu_kvcache.design_one.nonlinear import append_context, append_context_flops
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.models.state_transition import append_with_rolling_band
from read_correction_2026_09.cost import CostModel
from read_correction_v5.history_conditioned.capture import prepare_timeline
from selective_recompute_2026_09.evaluate import native_backend


@torch.no_grad()
def build_rolling_scenes(parent, current, rows, histories, expanded_histories, uids, *,
                         cutover, batch_size, device, max_length, attention_backend,
                         costs, append_targets=(0, 128, 384, 768), full_queries=False):
    """Split each user's existing queries across four causal terminal states.

    The pure scene is unchanged. Rolling scenes start from older Parent state;
    their aligned Full teacher can be shared only for identical terminal tuples.
    Context is fixed at each token's birth, including real within-band evictions.
    """
    if len(append_targets) != (2 if full_queries else 4) or append_targets[0] != 0:
        raise ValueError("use four split-query scenes, or two full-query scenes including pure Parent")
    scenes, records, timelines, buckets, teachers = {}, [], {}, defaultdict(list), {}
    for name in ("rolling_parent_cache_flops", "rolling_native_append_flops",
                 "rolling_context_flops", "rolling_teacher_cache_flops"):
        costs.setdefault(name, 0)
    models = {backend: CostModel(current.cfg.hidden_size, len(current.blocks),
                                 current.cfg.num_heads, backend)
              for backend in ("torch", "triton")}
    for uid in uids:
        base, terminal = rows[uid], histories[uid]
        if len(base["candidates"]) != 16:
            raise ValueError("keep the fixed sixteen queries per fitting user")
        digest = terminal_tuple_hash(terminal)
        teachers[uid, digest] = base["teacher"]
        key = f"{uid}:a0"
        scenes[key] = {**base, "uid": uid, "scene_kind": "a0",
                       "candidates": base["candidates"] if full_queries else base["candidates"][::4],
                       "context": torch.ones(1, len(terminal[0]), 2)}
        if full_queries:
            scenes[key]["item_ids"] = torch.as_tensor(terminal[1], dtype=torch.long)[None]
        records.append(dict(scene_id=key, scene_key=key, uid=uid, kind="a0", queries=16 if full_queries else 4,
            inherited_count=len(terminal[0]), native_writes=0, rolling_evictions=0,
            terminal_tuple_sha256=digest, terminal_last_timestamp=int(terminal[0][-1]),
            teacher_reused=True))
        for scene_index, target in enumerate(append_targets[1:], 1):
            timeline = prepare_timeline(expanded_histories[uid], cutover=cutover,
                                         history_length=max_length, append_target=target)
            timelines[uid, target] = timeline
            buckets[scene_index, target, len(timeline["prefix"][0]),
                    timeline["actual_append"], len(timeline["terminal"][0])].append(uid)
    for (scene_index, target, prefix_length, suffix_length, terminal_length), bucket in sorted(buckets.items()):
        backend = attention_backend if prefix_length == max_length else "torch"
        cost_model = models[backend]
        for start in range(0, len(bucket), batch_size):
            selected = bucket[start:start+batch_size]
            times, items, behaviors = [torch.as_tensor(np.stack([
                timelines[uid, target]["prefix"][column] for uid in selected]), device=device)
                for column in range(3)]
            deltas = torch.zeros_like(times, dtype=torch.float32)
            deltas[:, 1:] = times[:, 1:]-times[:, :-1]
            context = torch.ones(len(selected), prefix_length, 2, device=device)
            with native_backend([parent, current], backend):
                cache = parent.compute_kv(items.long(), behaviors.long(), deltas)
                costs["rolling_parent_cache_flops"] += cost_model.full_cache(prefix_length, batch=len(selected))
                if suffix_length:
                    new_times, new_items, new_behaviors = [torch.as_tensor(np.stack([
                        timelines[uid, target]["suffix"][column] for uid in selected]), device=device)
                        for column in range(3)]
                    previous_times = torch.cat((times[:, -1:], new_times[:, :-1]), dim=1)
                    new_deltas = (new_times-previous_times).clamp(0, 7*DAY).float()
                    offset = 0
                    while offset < suffix_length:
                        width = 1 << (min(32, suffix_length-offset).bit_length()-1)
                        end, before = offset+width, cache.seq_len
                        context = append_context(context, before, width, max_length)
                        costs["rolling_context_flops"] += append_context_flops(
                            batch=len(selected), old_length=before, width=width, max_length=max_length)
                        cache = append_with_rolling_band(current, cache, new_items[:, offset:end].long(),
                            new_behaviors[:, offset:end].long(), new_deltas[:, offset:end], max_length)
                        costs["rolling_native_append_flops"] += cost_model.band_append(
                            before, width, batch=len(selected), window_size=max_length)
                        offset = end
            if cache.seq_len != terminal_length or context.shape[1] != terminal_length:
                raise RuntimeError("rolling cache, context and teacher positions differ")
            for index, uid in enumerate(selected):
                timeline, base = timelines[uid, target], rows[uid]
                digest = terminal_tuple_hash(timeline["terminal"])
                teacher_reused = (uid, digest) in teachers
                if not teacher_reused:
                    ts, ids, bs = [torch.as_tensor(values[None], device=device)
                                   for values in timeline["terminal"]]
                    dt = torch.zeros_like(ts, dtype=torch.float32)
                    dt[:, 1:] = ts[:, 1:]-ts[:, :-1]
                    teacher_backend = attention_backend if terminal_length == max_length else "torch"
                    with native_backend([current], teacher_backend):
                        teachers[uid, digest] = current.compute_kv(ids.long(), bs.long(), dt).to("cpu")
                    costs["rolling_teacher_cache_flops"] += models[teacher_backend].full_cache(terminal_length)
                key, inherited = f"{uid}:a{target}", timeline["inherited_count"]
                scenes[key] = {**base, "uid": uid, "scene_kind": f"a{target}",
                    "parent": HSTUKVCache(cache.k[:, index:index+1].cpu(),
                                          cache.v[:, index:index+1].cpu(), terminal_length),
                    "teacher": teachers[uid, digest], "context": context[index:index+1].cpu(),
                    "candidates": base["candidates"] if full_queries else base["candidates"][scene_index::4],
                    "query_delta": float(cutover-timeline["terminal_last_timestamp"]),
                    "producer": [4]*inherited+[5]*(terminal_length-inherited),
                    "native_writes": suffix_length}
                if full_queries:
                    scenes[key]["item_ids"] = torch.as_tensor(timeline["terminal"][1], dtype=torch.long)[None]
                records.append(dict(scene_id=key, scene_key=key, uid=uid, kind=f"a{target}", queries=16 if full_queries else 4,
                    append_target=target, parent_length=prefix_length, terminal_length=terminal_length,
                    inherited_count=inherited, native_writes=suffix_length,
                    rolling_evictions=prefix_length+suffix_length-terminal_length,
                    parent_first_timestamp=int(timeline["prefix"][0][0]),
                    parent_last_timestamp=timeline["parent_last_timestamp"],
                    first_append_timestamp=timeline["pseudo_release_timestamp"],
                    terminal_last_timestamp=timeline["terminal_last_timestamp"],
                    native_backend=backend, teacher_reused=teacher_reused,
                    terminal_tuple_sha256=digest))
    return scenes, records
