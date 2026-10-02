"""The composite reuses causal Item views and applies one response per read."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import design_one.scoring as scoring
from hstu_kvcache.adaptation.reader import history_read, score
from hstu_kvcache.design_one.composite import FrozenItemResponseAdapter
from hstu_kvcache.design_one.item_kv import ItemKVReadViewAdapter
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.training import FoundationHistoryIndex
from selective_recompute_2026_09.evaluate import snapshots


@torch.inference_mode()
def test_composite_incremental_view_and_single_read_callback(monkeypatch):
    monkeypatch.setenv('EVOKV_ATTENTION_BACKEND', 'torch')
    torch.manual_seed(1804)
    config = HSTUConfig(num_items=32, num_behaviors=3, hidden_size=8, num_layers=2,
                       num_heads=2, max_seq_len=4, input_dropout=0., attn_dropout=0.)
    parent, current = HSTU(config).eval(), HSTU(config).eval()
    item = ItemKVReadViewAdapter(2, 2, 4, hidden_width=64, max_length=4).eval()
    for parameter in item.parameters():
        parameter.add_(torch.randn_like(parameter) * .01)
    composite = FrozenItemResponseAdapter.from_item_adapter(item).eval()
    for parameter in composite.response.parameters():
        parameter.add_(torch.randn_like(parameter) * .01)
    policies = [dict(tag='item', kind=scoring.ITEM_KV_KIND, adapter=item),
                dict(tag='composite', kind=scoring.ITEM_RESPONSE_KIND, adapter=composite)]
    events = {1: [(1, 1, 1), (2, 2, 2), (3, 3, 1), (4, 4, 2), (20, 5, 1), (22, 6, 2)],
              2: [(1, 7, 1), (2, 8, 2), (20, 9, 1), (22, 10, 2)]}
    requests = {uid: [dict(request_id=f'{uid}:{i}', uid=uid, query_timestamp=t,
                          item_idx=17+i, label=float(i % 2), full_logit=0., reuse_logit=0.,
                          append_count_since_cutover=sum(20 <= event[0] < t for event in rows))
                      for i, t in enumerate([20, 21, 23, 23])] for uid, rows in events.items()}
    history = FoundationHistoryIndex({uid: tuple(np.asarray(col, dtype=np.int64) for col in zip(*rows))
                                     for uid, rows in events.items()})
    expected = {policy['tag']: {} for policy in policies}
    extra_reads = snapshots_seen = 0
    for uid in events:
        def replay(observer=None):
            stats = {name: Counter() for name in ('initial_history_hist', 'full_history_hist')}
            return snapshots([uid], requests, history, parent, current, 20, max_length=4,
                             query_batch=2, stats=stats, append_band_size=4, observer=observer)

        observer = scoring.ViewObserver(policies, current_model=current)
        for plain, snap in zip(replay(), replay(observer), strict=True):
            snapshots_seen += 1
            native = snap.state.cache
            torch.testing.assert_close(native.k, plain.state.cache.k, rtol=0, atol=0)
            torch.testing.assert_close(native.v, plain.state.cache.v, rtol=0, atol=0)
            ids = torch.tensor([[event[1] for event in events[row['uid']]
                                 if event[0] < row['query_timestamp']][-4:] for row in snap.requests])
            mapped = item.map_cache(native, current.lookup_item_embeddings(ids))
            counts = torch.full((len(snap.requests),), native.seq_len)
            for policy in policies:
                actual_view = snap.context[policy['tag']]
                torch.testing.assert_close(actual_view.k, mapped.k, rtol=2e-5, atol=2e-7)
                torch.testing.assert_close(actual_view.v, mapped.v, rtol=2e-5, atol=2e-7)

                def reference(layer, query, unused_native):
                    response = history_read(current.blocks[layer].attn, query, mapped.k[layer], mapped.v[layer])
                    if policy['tag'] == 'composite':
                        response = composite.response.layers[layer](query, response, counts)
                    return response

                values = score(current, native, snap.candidates, snap.query_deltas,
                               history_override=reference)[0][:, 0]
                expected[policy['tag']].update((row['request_id'], float(value))
                    for row, value in zip(snap.requests, values, strict=True))
            extra_reads += item.estimate_flops(batch=len(snap.requests), tokens=native.seq_len)['extra_history_read']
    calls = []

    def tracked_score(model, cache, *args, **kwargs):
        original_k, original_v = cache.k.clone(), cache.v.clone()
        calls.append(cache.k.shape[1])
        result = score(model, cache, *args, **kwargs)
        torch.testing.assert_close(cache.k, original_k, rtol=0, atol=0)
        torch.testing.assert_close(cache.v, original_v, rtol=0, atol=0)
        return result

    monkeypatch.setattr(scoring, 'corrected_score', tracked_score)
    settings = dict(history_length=4, cohort_sizes={'medium': 2}, query_batches={'medium': 2},
                    attention_backend='torch', append_band_size=4)
    records, _, _ = scoring.score_unit(current, parent, history, requests, [1, 2], 20,
        policies, settings, 'medium', torch.device('cpu'), verify=False)
    assert len(calls) == snapshots_seen * len(policies)
    initial_cost = item.estimate_flops(tokens=6)['token_transform']
    update_cost = item.estimate_flops(tokens=4)['token_transform']
    for policy in policies:
        rows = records[policy['tag']]
        response_cost = (composite.response.estimate_flops(batch=len(rows))['candidate_reads']
                         if policy['tag'] == 'composite' else 0)
        assert sum(row['item_lookup_bytes'] for row in rows) == 10 * config.hidden_size * 4
        assert sum(row['view_initial_flops'] for row in rows) == initial_cost
        assert sum(row['view_update_flops'] for row in rows) == update_cost
        assert sum(row['read_flops'] for row in rows) == extra_reads + response_cost
        assert sum(row['correction_flops'] for row in rows) == initial_cost + update_cost + extra_reads + response_cost
        torch.testing.assert_close(torch.tensor([row['hstu_logit'] for row in rows]),
            torch.tensor([expected[policy['tag']][row['request_id']] for row in rows]), rtol=2e-5, atol=2e-7)
