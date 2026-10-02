"""Item views consume actual lifecycle IDs and discard temporary features."""
from collections import Counter
from pathlib import Path
import sys
import weakref

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from design_one.scoring import ITEM_KV_KIND, PRODUCER_SCALED_ITEM_KV_KIND, ViewObserver, score_unit
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.design_one.item_kv import ItemKVReadViewAdapter
from hstu_kvcache.design_one.producer_scaled_item import ProducerScaledItemKVAdapter
from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache
from hstu_kvcache.training import FoundationHistoryIndex
from selective_recompute_2026_09.evaluate import snapshots


@torch.inference_mode()
@pytest.mark.parametrize('producer_scaled', [False, True])
def test_item_views_follow_real_ids_and_leave_native_cache_unchanged(monkeypatch, producer_scaled):
    monkeypatch.setenv('EVOKV_ATTENTION_BACKEND', 'torch')
    torch.manual_seed(1786)
    config = HSTUConfig(num_items=32, num_behaviors=3, hidden_size=8, num_layers=2,
                        num_heads=2, max_seq_len=4, input_dropout=0., attn_dropout=0.)
    parent, current = HSTU(config).eval(), HSTU(config).eval()
    adapter = ItemKVReadViewAdapter(2, 2, 4, hidden_width=64 if producer_scaled else 5,
                                    max_length=4).eval()
    for parameter in adapter.parameters():
        parameter.add_(torch.randn_like(parameter) * .01)
    policies = [dict(tag=tag, kind=ITEM_KV_KIND, adapter=adapter) for tag in ('item_a', 'item_b')]
    if producer_scaled:
        scaled = ProducerScaledItemKVAdapter.from_item_adapter(adapter).eval()
        identity = ProducerScaledItemKVAdapter.from_item_adapter(adapter).eval()
        for layer, alpha in zip(scaled.layers, (.25, 1.5), strict=True):
            layer.native_scale.fill_(alpha)
        policies = [policies[0],
                    dict(tag='scaled', kind=PRODUCER_SCALED_ITEM_KV_KIND, adapter=scaled),
                    dict(tag='alpha_one', kind=PRODUCER_SCALED_ITEM_KV_KIND, adapter=identity)]
    events = {
        1: [(1, 1, 1), (2, 2, 2), (3, 3, 1), (4, 4, 2),
            (20, 5, 1), (20, 6, 2), (22, 7, 1), (25, 8, 2)],
        2: [(1, 9, 1), (2, 10, 2), (3, 11, 1), (4, 12, 2),
            (20, 13, 1), (22, 14, 1), (22, 15, 2), (25, 16, 2)],
        3: [(1, 1, 1), (2, 2, 2), (20, 3, 1), (22, 4, 2), (22, 5, 1)],
    }
    query_times = {1: [20, 20, 21, 22, 26], 2: [20, 23, 23, 26], 3: [20, 21, 23]}
    requests = {uid: [dict(request_id=f'{uid}:{i}', uid=uid, query_timestamp=t,
                          item_idx=17 + i, label=float(i % 2), full_logit=0., reuse_logit=0.,
                          append_count_since_cutover=sum(20 <= event[0] < t for event in events[uid]))
                      for i, t in enumerate(times)] for uid, times in query_times.items()}
    history = FoundationHistoryIndex({uid: tuple(np.asarray(col, dtype=np.int64) for col in zip(*rows))
                                     for uid, rows in events.items()})
    total_lookup_bytes = initial_cost = update_cost = extra_reads = current_rows = 0
    expected_scores = {policy['tag']: {} for policy in policies}
    for cohort in ([1, 2], [3]):
        def replay(observer=None):
            stats = {name: Counter() for name in ('initial_history_hist', 'full_history_hist')}
            return snapshots(cohort, requests, history, parent, current, 20,
                             max_length=4, query_batch=2, stats=stats,
                             append_band_size=4, observer=observer)

        plain = list(replay())
        observer = ViewObserver(policies, current_model=current)
        feature_refs, looked_up_ids = [], []
        original_lookup = observer._item_features

        def tracked_lookup(raw_events):
            features = original_lookup(raw_events)
            looked_up_ids.extend(raw_events[0].flatten().tolist())
            feature_refs.append(weakref.ref(features))
            return features

        monkeypatch.setattr(observer, '_item_features', tracked_lookup)
        for native, snap in zip(plain, replay(observer), strict=True):
            torch.testing.assert_close(snap.state.cache.k, native.state.cache.k, rtol=0, atol=0)
            torch.testing.assert_close(snap.state.cache.v, native.state.cache.v, rtol=0, atol=0)
            assert observer.write_context is None
            assert all(reference() is None for reference in feature_refs)
            # Reconstruct the strict causal item IDs independently of observer
            # metadata and snapshot raw_events; same-time events stay excluded.
            ids = torch.tensor([[event[1] for event in sorted(events[row['uid']])
                                 if event[0] < row['query_timestamp']][-4:] for row in snap.requests])
            reference = adapter.map_cache(snap.state.cache, current.lookup_item_embeddings(ids))
            for policy in policies:
                target = reference
                if policy['kind'] == PRODUCER_SCALED_ITEM_KV_KIND:
                    # Independent producer reference from strictly causal raw
                    # timestamps, never observer metadata or the mapped view.
                    native_rows = torch.tensor([[event[0] >= 20 for event in sorted(events[row['uid']])
                                                 if event[0] < row['query_timestamp']][-4:]
                                                for row in snap.requests])[None, :, :, None]
                    alpha = torch.stack([layer.native_scale for layer in policy['adapter'].layers])
                    alpha = alpha[:, None, None, None]
                    source = snap.state.cache
                    target = HSTUKVCache(torch.where(native_rows, source.k + alpha * (reference.k - source.k), reference.k),
                                        torch.where(native_rows, source.v + alpha * (reference.v - source.v), reference.v),
                                        source.seq_len)
                expected = score(current, target, snap.candidates, snap.query_deltas)[0][:, 0]
                expected_scores[policy['tag']].update((row['request_id'], float(value))
                    for row, value in zip(snap.requests, expected, strict=True))
                view = snap.context[policy['tag']]
                torch.testing.assert_close(view.k, target.k, rtol=2e-5, atol=2e-7)
                torch.testing.assert_close(view.v, target.v, rtol=2e-5, atol=2e-7)
            before = len(looked_up_ids)
            observer.reading([cohort.index(row['uid']) for row in snap.requests])
            assert len(looked_up_ids) == before
            extra_reads += adapter.estimate_flops(batch=len(snap.requests),
                                                  tokens=snap.state.cache.seq_len)['extra_history_read']
        expected_ids = [event[1] for uid in cohort for event in events[uid]
                        if event[0] < max(query_times[uid])]
        assert Counter(looked_up_ids) == Counter(expected_ids)
        prefix = sum(sum(event[0] < 20 for event in events[uid]) for uid in cohort)
        appends = len(expected_ids) - prefix
        current_rows += appends
        lookup_bytes = len(expected_ids) * config.hidden_size * current.item_emb.weight.element_size()
        for policy in policies:
            assert observer.item_lookup_bytes[policy['tag']] == lookup_bytes
        total_lookup_bytes += lookup_bytes
        initial_cost += adapter.estimate_flops(tokens=prefix)['token_transform']
        update_cost += adapter.estimate_flops(tokens=appends)['token_transform']
    settings = dict(history_length=4, cohort_sizes={'medium': 2}, query_batches={'medium': 2},
                    attention_backend='torch', append_band_size=4)
    records, _, _ = score_unit(current, parent, history, requests, [1, 2, 3], 20,
                              policies, settings, 'medium', torch.device('cpu'), verify=False)
    for policy in policies:
        rows = records[policy['tag']]
        # Only Current rows incur one additional multiply per K/V component;
        # alpha=1 is still executed and charged, with no initial-state surcharge.
        scale_cost = (current_rows * config.num_layers * 2 * config.hidden_size
                      if policy['kind'] == PRODUCER_SCALED_ITEM_KV_KIND else 0)
        assert sum(row['item_lookup_bytes'] for row in rows) == total_lookup_bytes
        assert sum(row['view_initial_flops'] for row in rows) == initial_cost
        assert sum(row['view_update_flops'] for row in rows) == update_cost + scale_cost
        assert sum(row['correction_flops'] for row in rows) == initial_cost + update_cost + scale_cost + extra_reads
        torch.testing.assert_close(torch.tensor([row['hstu_logit'] for row in rows]),
                                   torch.tensor([expected_scores[policy['tag']][row['request_id']] for row in rows]),
                                   rtol=2e-5, atol=2e-7)
