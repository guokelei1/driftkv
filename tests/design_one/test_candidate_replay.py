"""Incremental candidate views follow native appends without changing writes."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))

from design_one.scoring import AFFINE_KV_KIND, CONTEXT_KV_KIND, KV_KIND, RESPONSE_KIND, ViewObserver, score_unit
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.design_one.nonlinear import (
    NonlinearResponseAdapter, KVReadViewAdapter, ContextKVReadViewAdapter, append_context_flops)
from hstu_kvcache.design_one.affine import AffineKVReadViewAdapter
from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache
from hstu_kvcache.training import FoundationHistoryIndex
from selective_recompute_2026_09.evaluate import snapshots


@torch.inference_mode()
@pytest.mark.parametrize('conditional_kind', [CONTEXT_KV_KIND, AFFINE_KV_KIND])
def test_incremental_views_match_fresh_native_state_and_complete_costs(monkeypatch, conditional_kind):
    monkeypatch.setenv('EVOKV_ATTENTION_BACKEND', 'torch')
    torch.manual_seed(1784)
    cfg = HSTUConfig(num_items=32, num_behaviors=3, hidden_size=8, num_layers=2,
                    num_heads=2, max_seq_len=4, input_dropout=0., attn_dropout=0.)
    parent, current = HSTU(cfg).eval(), HSTU(cfg).eval()
    response = NonlinearResponseAdapter(2, 2, 4, hidden_width=5, max_length=4).eval()
    kv = KVReadViewAdapter(2, 2, 4, hidden_width=5, max_length=4).eval()
    conditional = (AffineKVReadViewAdapter(2, 2, 4, max_length=4) if conditional_kind == AFFINE_KV_KIND
                   else ContextKVReadViewAdapter(2, 2, 4, hidden_width=5, max_length=4)).eval()
    for adapter in (response, kv, conditional):
        for parameter in adapter.parameters():
            parameter.add_(torch.randn_like(parameter) * .01)
    if conditional_kind == AFFINE_KV_KIND:
        for layer in conditional.layers:
            layer.map_weight.normal_(0, .001)
            layer.map_bias.normal_(0, .001)
    policies = [dict(tag='response', kind=RESPONSE_KIND, adapter=response),
                dict(tag='kv', kind=KV_KIND, adapter=kv),
                dict(tag='kv_parent', kind=KV_KIND, adapter=kv, producer_scope='parent_only'),
                dict(tag='kv_context', kind=conditional_kind, adapter=conditional)]
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
                          append_count_since_cutover=sum(20 <= e[0] < t for e in events[uid]))
                      for i, t in enumerate(times)] for uid, times in query_times.items()}
    history = FoundationHistoryIndex({uid: tuple(np.asarray(column, dtype=np.int64)
                                                for column in zip(*rows))
                                     for uid, rows in events.items()})
    direct_map = kv.map_cache
    mapped_tokens = []

    def counted_map(cache, **kwargs):
        mapped_tokens.append(cache.k.shape[1] * cache.seq_len)
        return direct_map(cache, **kwargs)

    monkeypatch.setattr(kv, 'map_cache', counted_map)
    direct_context_map = conditional.map_cache
    context_mapped_tokens = []

    def counted_context_map(cache, **kwargs):
        context_mapped_tokens.append(cache.k.shape[1] * cache.seq_len)
        return direct_context_map(cache, **kwargs)

    monkeypatch.setattr(conditional, 'map_cache', counted_context_map)
    initial_cost = update_cost = extra_read_cost = 0
    conditional_initial_cost = conditional_update_cost = context_cost = 0
    parent_scores, conditional_scores = {}, {}
    for cohort in ([1, 2], [3]):
        def replay(observer=None):
            stats = {key: Counter() for key in ('initial_history_hist', 'full_history_hist')}
            return snapshots(cohort, requests, history, parent, current, 20,
                             max_length=4, query_batch=2, stats=stats,
                             append_band_size=4, observer=observer)

        plain = list(replay())
        observer = ViewObserver(policies)
        mapped_tokens.clear()
        context_mapped_tokens.clear()
        append_shapes = []
        original_append = observer.appended

        def tracked_append(indices, old, new, width):
            append_shapes.append((len(indices), old.seq_len, width))
            original_append(indices, old, new, width)

        monkeypatch.setattr(observer, 'appended', tracked_append)
        for native_snap, snap in zip(plain, replay(observer), strict=True):
            torch.testing.assert_close(snap.state.cache.k, native_snap.state.cache.k, rtol=0, atol=0)
            torch.testing.assert_close(snap.state.cache.v, native_snap.state.cache.v, rtol=0, atol=0)
            assert observer.summary is None
            mapped = snap.context['kv']
            reference = direct_map(snap.state.cache)
            torch.testing.assert_close(mapped.k, reference.k, rtol=2e-5, atol=2e-7)
            torch.testing.assert_close(mapped.v, reference.v, rtol=2e-5, atol=2e-7)
            # Independent sequential lineage reference. Each current row stores
            # the capped window fraction at its own birth, including itself;
            # future appends never revise that row's original fraction.
            contexts = []
            for row in snap.requests:
                source = events[row['uid']]
                birth = [(1., 1.) for event in source if event[0] < 20][-4:]
                for event in sorted(e for e in source if 20 <= e[0] < row['query_timestamp']):
                    birth = (birth + [(0., 0.)])[-4:]
                    birth[-1] = (0., sum(entry[0] for entry in birth) / len(birth))
                contexts.append(birth)
            expected_context = torch.tensor(contexts)
            owners = [cohort.index(row['uid']) for row in snap.requests]
            torch.testing.assert_close(observer.write_context[owners], expected_context, rtol=0, atol=0)
            expected_view = direct_context_map(snap.state.cache, context=expected_context)
            torch.testing.assert_close(snap.context['kv_context'].k, expected_view.k, rtol=2e-5, atol=2e-7)
            torch.testing.assert_close(snap.context['kv_context'].v, expected_view.v, rtol=2e-5, atol=2e-7)
            expected_conditional = score(current, expected_view, snap.candidates, snap.query_deltas)[0][:, 0]
            conditional_scores.update((row['request_id'], float(value))
                                      for row, value in zip(snap.requests, expected_conditional, strict=True))
            parent_view = snap.context['kv_parent']
            inherited = torch.tensor([max(0, reference.seq_len - row['append_count_since_cutover'])
                                      for row in snap.requests])
            old_mask = (torch.arange(reference.seq_len)[None] < inherited[:, None])[None, :, :, None]
            expected_k = torch.where(old_mask, reference.k, snap.state.cache.k)
            expected_v = torch.where(old_mask, reference.v, snap.state.cache.v)
            torch.testing.assert_close(parent_view.k, expected_k, rtol=2e-5, atol=2e-7)
            torch.testing.assert_close(parent_view.v, expected_v, rtol=2e-5, atol=2e-7)
            native_mask = (~old_mask).expand_as(parent_view.k)
            torch.testing.assert_close(parent_view.k[native_mask], snap.state.cache.k[native_mask], rtol=0, atol=0)
            torch.testing.assert_close(parent_view.v[native_mask], snap.state.cache.v[native_mask], rtol=0, atol=0)
            expected_parent = score(current, HSTUKVCache(expected_k, expected_v, reference.seq_len),
                                    snap.candidates, snap.query_deltas)[0][:, 0]
            parent_scores.update((row['request_id'], float(value))
                                 for row, value in zip(snap.requests, expected_parent, strict=True))
            # An extra native read before override must not alter mapped-read scores.
            actual = score(current, snap.state.cache, snap.candidates, snap.query_deltas,
                           history_override=kv.make_history_override(current, mapped))[0]
            expected = score(current, reference, snap.candidates, snap.query_deltas)[0]
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
            counts = torch.full((len(snap.requests),), snap.state.cache.seq_len)
            actual = score(current, snap.state.cache, snap.candidates, snap.query_deltas,
                           history_override=snap.context['response'])[0]
            expected = score(current, snap.state.cache, snap.candidates, snap.query_deltas,
                             history_override=response.make_history_override(counts))[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            before = len(mapped_tokens), len(context_mapped_tokens)
            observer.reading(owners)
            assert (len(mapped_tokens), len(context_mapped_tokens)) == before
            extra_read_cost += kv.estimate_flops(batch=len(snap.requests),
                                                tokens=snap.state.cache.seq_len)['extra_history_read']
        prefix_tokens = sum(sum(e[0] < 20 for e in events[uid]) for uid in cohort)
        append_tokens = sum(sum(20 <= e[0] < max(query_times[uid]) for e in events[uid])
                            for uid in cohort)
        assert sum(mapped_tokens) == 2 * prefix_tokens + append_tokens
        assert sum(context_mapped_tokens) == prefix_tokens + append_tokens
        expected_initial = kv.estimate_flops(tokens=prefix_tokens)['token_transform']
        expected_updates = kv.estimate_flops(tokens=append_tokens)['token_transform']
        assert observer.costs['kv']['view_initial_flops'] == expected_initial
        assert observer.costs['kv']['view_update_flops'] == expected_updates
        assert observer.costs['kv_parent']['view_initial_flops'] == expected_initial
        assert observer.costs['kv_parent']['view_update_flops'] == 0
        expected_conditional_initial = conditional.estimate_flops(tokens=prefix_tokens)['token_transform']
        expected_conditional_updates = conditional.estimate_flops(tokens=append_tokens)['token_transform']
        expected_context_cost = sum(append_context_flops(batch=b, old_length=n, width=w, max_length=4)
                                    for b, n, w in append_shapes)
        assert observer.costs['kv_context']['view_initial_flops'] == expected_conditional_initial
        assert observer.costs['kv_context']['view_update_flops'] == expected_conditional_updates
        assert observer.costs['kv_context']['context_update_flops'] == expected_context_cost
        assert observer.costs['kv']['context_update_flops'] == 0
        assert observer.costs['kv_parent']['context_update_flops'] == 0
        assert not observer.costs['response']
        initial_cost += expected_initial
        update_cost += expected_updates
        conditional_initial_cost += expected_conditional_initial
        conditional_update_cost += expected_conditional_updates
        context_cost += expected_context_cost
    settings = dict(history_length=4, cohort_sizes={'medium': 2}, query_batches={'medium': 2},
                    attention_backend='torch', append_band_size=4)
    records, _, _ = score_unit(current, parent, history, requests, [1, 2, 3], 20,
                              policies, settings, 'medium', torch.device('cpu'), verify=False)
    assert sum(row['read_flops'] for row in records['kv']) == extra_read_cost
    assert sum(row['view_initial_flops'] for row in records['kv']) == initial_cost
    assert sum(row['view_update_flops'] for row in records['kv']) == update_cost
    assert sum(row['correction_flops'] for row in records['kv']) == initial_cost + update_cost + extra_read_cost
    assert sum(row['read_flops'] for row in records['kv_parent']) == extra_read_cost
    assert sum(row['view_initial_flops'] for row in records['kv_parent']) == initial_cost
    assert sum(row['view_update_flops'] for row in records['kv_parent']) == 0
    assert sum(row['correction_flops'] for row in records['kv_parent']) == initial_cost + extra_read_cost
    torch.testing.assert_close(torch.tensor([row['hstu_logit'] for row in records['kv_parent']]),
                               torch.tensor([parent_scores[row['request_id']] for row in records['kv_parent']]),
                               rtol=2e-5, atol=2e-7)
    assert sum(row['read_flops'] for row in records['kv_context']) == extra_read_cost
    assert sum(row['view_initial_flops'] for row in records['kv_context']) == conditional_initial_cost
    assert sum(row['view_update_flops'] for row in records['kv_context']) == conditional_update_cost
    assert sum(row['context_update_flops'] for row in records['kv_context']) == context_cost
    assert sum(row['correction_flops'] for row in records['kv_context']) == (
        conditional_initial_cost + conditional_update_cost + context_cost + extra_read_cost)
    torch.testing.assert_close(torch.tensor([row['hstu_logit'] for row in records['kv_context']]),
                               torch.tensor([conditional_scores[row['request_id']] for row in records['kv_context']]),
                               rtol=2e-5, atol=2e-7)
    assert sum(row['correction_flops'] for row in records['response']) == (
        len(records['response']) * response.estimate_flops()['candidate_reads'])
