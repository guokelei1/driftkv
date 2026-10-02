"""Causal mixed scenes, fixed per-user queries and fitting-only coordinates."""

from collections import defaultdict
from io import BytesIO
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from design_one.calibrate import fit, fit_without_response, rolling_scenes
from hstu_kvcache.design_one import ProducerSummary, SharedReadAdapter, SummaryProjection
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.models.state_transition import append_with_rolling_cap


@torch.no_grad()
def test_rolling_scene_uses_older_parent_dependencies_and_real_evictions(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    torch.manual_seed(172005)
    config = HSTUConfig(num_items=64, num_behaviors=3, hidden_size=32, num_layers=2,
                        num_heads=1, max_seq_len=8, input_dropout=0., attn_dropout=0.)
    parent, current = HSTU(config).eval(), HSTU(config).eval()
    # The long user's requested tail starts inside timestamp 8, so the split
    # moves after that complete group: eight Parent rows, two native writes.
    times = {0: np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8*86400+11, 8*86400+12]),
             1: np.asarray([1, 2, 3, 4, 5]),
             2: np.asarray([1, 2, 3, 4, 4, 4, 5, 6, 7, 8, 9, 10])}
    cutover = 8*86400+20
    expanded, terminal, rows = {}, {}, {}
    for uid in times:
        items = np.arange(1+uid*16, len(times[uid])+1+uid*16)
        expanded[uid] = times[uid], items, np.ones(len(items), dtype=np.int64)
        terminal[uid] = tuple(values[-8:] for values in expanded[uid])
        if uid == 2:
            # A raw-ID cutoff can retain a different member of the oldest
            # timestamp group than the expanded mapped-ID ordered window.
            terminal[uid] = tuple(np.r_[values[3:4], values[5:]] for values in expanded[uid])
        ts, ids, bs = [torch.tensor(values)[None] for values in terminal[uid]]
        dt = torch.zeros_like(ts, dtype=torch.float32)
        dt[:, 1:] = ts[:, 1:]-ts[:, :-1]
        rows[uid] = dict(parent=parent.compute_kv(ids, bs, dt),
                         teacher=current.compute_kv(ids, bs, dt),
                         candidates=torch.arange(1, 17), query_delta=float(cutover-ts[0, -1]))
    costs = defaultdict(int)
    scenes, records = rolling_scenes(parent, current, rows, terminal, expanded, list(rows),
        cutover=cutover, batch_size=2, device=torch.device("cpu"), max_length=8,
        attention_backend="torch", costs=costs)
    for uid, prefix_slice, suffix_start, evictions in ((0, slice(2, 10), 10, 2), (1, slice(0, 3), 3, 0),
                                                     (2, slice(0, 8), 8, 4)):
        mixed = scenes[f"{uid}:mixed"]
        record = next(row for row in records if row["scene_id"] == f"{uid}:mixed")
        writes = len(expanded[uid][0])-suffix_start
        assert record["rolling_evictions"] == evictions and record["native_writes"] == writes
        assert record["parent_last_timestamp"] < record["first_append_timestamp"] < cutover
        assert scenes[f"{uid}:pure"]["teacher"] is rows[uid]["teacher"]
        if uid != 2:
            assert mixed["teacher"] is rows[uid]["teacher"] and record["teacher_reused"]
        else:
            assert mixed["teacher"] is not rows[uid]["teacher"] and not record["teacher_reused"]
            teacher_times, teacher_ids, teacher_bs = [torch.tensor(values[-8:])[None] for values in expanded[uid]]
            teacher_dt = torch.zeros_like(teacher_times, dtype=torch.float32)
            teacher_dt[:, 1:] = teacher_times[:, 1:]-teacher_times[:, :-1]
            fresh_teacher = current.compute_kv(teacher_ids, teacher_bs, teacher_dt)
            torch.testing.assert_close(mixed["teacher"].k, fresh_teacher.k)
            torch.testing.assert_close(mixed["teacher"].v, fresh_teacher.v)
        ts, ids, bs = [torch.tensor(values[prefix_slice])[None] for values in expanded[uid]]
        dt = torch.zeros_like(ts, dtype=torch.float32)
        dt[:, 1:] = ts[:, 1:]-ts[:, :-1]
        old = parent.compute_kv(ids, bs, dt)
        new_times, new_ids, new_bs = [torch.tensor(values[suffix_start:])[None] for values in expanded[uid]]
        new_dt = (new_times-torch.cat((ts[:, -1:], new_times[:, :-1]), 1)).clamp(0, 7*86400).float()
        reference = append_with_rolling_cap(current, old, new_ids, new_bs, new_dt, 8)
        torch.testing.assert_close(mixed["parent"].k, reference.k, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(mixed["parent"].v, reference.v, atol=3e-6, rtol=3e-5)
        inherited = record["inherited_count"]
        torch.testing.assert_close(mixed["parent"].k[:, :, :inherited], old.k[:, :, evictions:], atol=0, rtol=0)
        assert mixed["producer"] == [4]*inherited+[5]*writes
    assert costs["mixed_extra_parent_capture_flops"] > 0
    assert costs["mixed_native_append_flops"] > 0
    assert costs["mixed_extra_teacher_capture_flops"] > 0

    # Keep both scenes of a user within the same role; PCA sees fitting users only.
    train = [f"{uid}:{kind}" for uid in (0, 1) for kind in ("pure", "mixed")]
    validation = [f"2:{kind}" for kind in ("pure", "mixed")]
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        adapter, diagnostics = fit(current, scenes, train, validation, batch_size=2,
            device=torch.device("cpu"), max_length=8, rank=2, costs=costs)
    finally:
        torch.set_num_threads(previous_threads)
    features = []
    for key in train:
        row = scenes[key]
        summary = ProducerSummary.from_cache(row["parent"], row.get("producer", 4), max_length=8)
        summary.native_writes.fill_(row.get("native_writes", 0))
        features.append(summary.features()[0])
    torch.testing.assert_close(adapter.projection.center, torch.stack(features).double().mean(0))
    assert all(layer["fitting_scenes"] == 4 and layer["fitting_queries"] == 8
               for layer in diagnostics["layers"])
    assert set(diagnostics["validation_by_scene"]) == {"pure", "mixed"}
    assert all(row["scenes"] == 1 for group in diagnostics["validation_by_scene"].values() for row in group)


@torch.no_grad()
def test_without_response_matches_weighted_ridge_and_native_invariant_saved_correction():
    generator = torch.Generator().manual_seed(172007)
    random = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.double)
    features = random(7, 9)
    projection = SummaryProjection.fit(features, rank=2)
    latent = projection.encode_features(features)
    query, wanted = random(7, 2, 5, 3), random(7, 2, 5, 3)
    counts = torch.tensor([1., 2., 3., 4., 5., 7., 8.], dtype=torch.double)
    parameters, stats = fit_without_response(latent, query, wanted, counts)
    assert stats["read_dimension"] == 0 and stats["cpu_solver_threads"] == 1
    assert parameters["read_weights"].shape == (2, 0, 3)
    assert parameters["read_center"].shape == parameters["read_scale"].shape == (0,)
    normalized = (query-query.mean((0, 2), keepdim=True))/query.std((0, 2), keepdim=True).clamp_min(1e-4)
    augmented = torch.cat((torch.ones_like(normalized[..., :1]), normalized), -1)
    design = torch.einsum("br,bhqj->bhqrj", latent, augmented).flatten(-2)
    weight = counts.square()/counts.square().mean()
    gram = torch.einsum("bhqi,bhqj,b->hij", design, design, weight)/5
    rhs = torch.einsum("bhqi,bhqd,b->hid", design, wanted, weight)/5
    reference = torch.linalg.solve(gram+.01*len(features)*torch.eye(design.shape[-1], dtype=torch.double), rhs)
    actual_weights = parameters["weights"].transpose(0, 1).flatten(1, 2)
    torch.testing.assert_close(actual_weights, reference, atol=1e-10, rtol=1e-10)
    adapter = SharedReadAdapter(projection, [parameters], max_length=8)
    payload = BytesIO()
    torch.save(adapter.state_dict(), payload)
    payload.seek(0)
    restored = SharedReadAdapter.from_state_dict(torch.load(payload, weights_only=True))
    view = restored.prepare_features(features, counts)
    native_a, native_b = random(7, 2, 5, 3), random(7, 2, 5, 3)*11
    delta_a = view(0, query, native_a)-native_a
    delta_b = view(0, query, native_b)-native_b
    expected = torch.einsum("bhqi,hid->bhqd", design, reference)*counts[:, None, None, None]
    torch.testing.assert_close(delta_a, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(delta_b, expected, atol=1e-10, rtol=1e-10)
    assert restored.estimate_flops()["shared_response_preparation"] == 0
