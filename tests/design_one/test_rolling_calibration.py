"""Compare calibration windows and immutable birth context to serial native writes."""

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from design_one.rolling_calibration import build_rolling_scenes
from design_one import calibrate_nonlinear as calibration
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.models.state_transition import append_with_rolling_cap
from read_correction_v5.history_conditioned.capture import prepare_timeline


@torch.no_grad()
def test_four_scenes_match_serial_native_and_aligned_teacher(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.manual_seed(172008)
    cfg = HSTUConfig(num_items=64, num_behaviors=3, hidden_size=32, num_layers=2,
                     num_heads=1, max_seq_len=8, input_dropout=0., attn_dropout=0.)
    parent, current = HSTU(cfg).eval(), HSTU(cfg).eval()
    times = {0: np.array([1, 2, 3, 4, 4, 4, 5, 6, 7, 8, 8*86400+9, 8*86400+10]),
             1: np.arange(1, 6)}
    expanded, histories, rows = {}, {}, {}
    cutover = 8*86400+20

    def full(model, events):
        ts, ids, bs = [torch.tensor(value)[None] for value in events]
        dt = torch.zeros_like(ts, dtype=torch.float32)
        dt[:, 1:] = ts[:, 1:]-ts[:, :-1]
        return model.compute_kv(ids, bs, dt)

    for uid, ts in times.items():
        expanded[uid] = ts, np.arange(uid*16+1, uid*16+len(ts)+1), np.ones(len(ts), dtype=np.int64)
        histories[uid] = tuple(value[-8:] for value in expanded[uid])
        if uid == 0:
            histories[uid] = tuple(np.r_[value[3:4], value[5:]] for value in expanded[uid])
        rows[uid] = dict(parent=full(parent, histories[uid]), teacher=full(current, histories[uid]),
                         candidates=torch.arange(1, 17), query_delta=float(cutover-ts[-1]))
    costs = defaultdict(int)
    scenes, records = build_rolling_scenes(parent, current, rows, histories, expanded, list(rows),
        cutover=cutover, batch_size=2, device=torch.device("cpu"), max_length=8,
        attention_backend="torch", costs=costs, append_targets=(0, 1, 3, 6))
    assert len(scenes) == len(records) == 8
    for uid in rows:
        assert scenes[f"{uid}:a0"]["teacher"] is rows[uid]["teacher"]
        assert sorted(torch.cat([scenes[f"{uid}:a{target}"]["candidates"] for target in (0, 1, 3, 6)]).tolist()) == list(range(1, 17))
        for target in (1, 3, 6):
            timeline = prepare_timeline(expanded[uid], cutover=cutover, history_length=8, append_target=target)
            actual = scenes[f"{uid}:a{target}"]
            cache = full(parent, timeline["prefix"])
            context = [[1., 1.] for _ in range(cache.seq_len)]
            last = int(timeline["prefix"][0][-1])
            for ts, item, behavior in zip(*timeline["suffix"], strict=True):
                dt = torch.tensor([[min(int(ts)-last, 7*86400)]], dtype=torch.float32)
                cache = append_with_rolling_cap(current, cache, torch.tensor([[item]]),
                                                 torch.tensor([[behavior]]), dt, 8)
                if len(context) == 8:
                    context.pop(0)
                fraction = sum(row[0] for row in context)/(len(context)+1)
                context.append([0., fraction])
                last = int(ts)
            torch.testing.assert_close(actual["parent"].k, cache.k, atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(actual["parent"].v, cache.v, atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(actual["context"], torch.tensor([context]))
            teacher = full(current, timeline["terminal"])
            torch.testing.assert_close(actual["teacher"].k, teacher.k)
            torch.testing.assert_close(actual["teacher"].v, teacher.v)
            record = next(row for row in records if row["scene_key"] == f"{uid}:a{target}")
            assert record["parent_last_timestamp"] <= record["first_append_timestamp"] < cutover
            assert record["terminal_last_timestamp"] < cutover
    # All rolling teachers for a user share the identical terminal; its oldest
    # timestamp boundary differs from the pure capture for user0, hence one rebuild.
    assert sum(not row["teacher_reused"] for row in records) == 1
    assert costs["rolling_teacher_cache_flops"] > 0
    assert costs["rolling_native_append_flops"] > 0
    assert costs["rolling_context_flops"] > 0

    # The new paired mode keeps all16 candidates in each state. The forced
    # terminal-tuple difference above must also change the item features.
    paired, pair_records = build_rolling_scenes(parent, current, rows, histories, expanded, list(rows),
        cutover=cutover, batch_size=2, device=torch.device("cpu"), max_length=8,
        attention_backend="torch", costs=defaultdict(int), append_targets=(0, 6), full_queries=True)
    calibration.attach_scene_item_features(current, paired, torch.device("cpu"))
    for uid in rows:
        for target in (0, 6):
            row = paired[f"{uid}:a{target}"]
            terminal = (histories[uid] if target == 0 else
                prepare_timeline(expanded[uid], cutover=cutover, history_length=8, append_target=6)["terminal"])
            expected_ids = torch.as_tensor(terminal[1])[None]
            torch.testing.assert_close(row["item_ids"], expected_ids)
            torch.testing.assert_close(row["item_features"], current.lookup_item_embeddings(expected_ids))
            torch.testing.assert_close(row["candidates"], rows[uid]["candidates"])
            torch.testing.assert_close(row["parent"].k, scenes[f"{uid}:a{target}"]["parent"].k)
    assert not torch.equal(paired["0:a0"]["item_features"], paired["0:a6"]["item_features"])
    assert all(record["queries"] == 16 for record in pair_records)

    # Run the real joint loop: each of two epochs must train only one scene
    # for the sole fitting UID. The other UID remains diagnostic-only.
    adapter = calibration.make_adapter(current, "kv_item")
    with torch.enable_grad():
        _, stats = calibration.refine_logits(current, adapter, paired, ["0:a0"], ["1:a0"],
            mixed_train=["0:a6"], mixed_validation=["1:a6"], method="kv_item", epochs=2,
            batch_size=2, device=torch.device("cpu"), costs=defaultdict(int))
    epochs = stats["joint_logit"]["epochs"]
    assert [record["users"] for record in epochs] == [1, 1]
    assert [record["queries_per_user"] for record in epochs] == [16, 16]
    assert [record["batches"] for record in epochs] == [1, 1]
    assert stats["joint_logit"]["optimizer_steps"] == 2
    assert "mixed_validation" in stats
