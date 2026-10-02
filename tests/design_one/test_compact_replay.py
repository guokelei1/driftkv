"""Compact execution preserves real replay and reports persistent state only."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from design_one.benchmark import design_policy, pending_setup_flops
from design_one.scoring import ITEM_RESPONSE_KIND, ViewObserver, score_unit
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.design_one.compact import CompactItemResponseAdapter
from hstu_kvcache.design_one.composite import FrozenItemResponseAdapter
from hstu_kvcache.design_one.item_kv import ItemKVReadViewAdapter
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.training import FoundationHistoryIndex
from read_correction_v4.evaluate_full import signature
from read_correction_v5.common import sha256
from selective_recompute_2026_09.evaluate import snapshots


@torch.inference_mode()
def test_compact_incremental_replay_memory_and_cost(monkeypatch):
    monkeypatch.setenv('EVOKV_ATTENTION_BACKEND', 'torch')
    torch.manual_seed(1910)
    config = HSTUConfig(num_items=32, num_behaviors=3, hidden_size=192, num_layers=2,
                       num_heads=6, max_seq_len=4, input_dropout=0., attn_dropout=0.)
    parent, current = HSTU(config).eval(), HSTU(config).eval()
    base = FrozenItemResponseAdapter.from_item_adapter(
        ItemKVReadViewAdapter(2, 6, 32, hidden_width=64, max_length=4)).eval()
    for layer in (*base.item.layers, *base.response.layers):
        layer.output.weight.normal_(std=.01)
        layer.output.bias.normal_(std=.01)
        layer.output_scale.uniform_(.5, 1.5)
    compact = CompactItemResponseAdapter(base).eval()
    policies = [dict(tag='mapped', kind=ITEM_RESPONSE_KIND, adapter=base),
                dict(tag='compact', kind=ITEM_RESPONSE_KIND, adapter=compact, execution_mode='compact_hidden')]
    events = {1: [(1, 1, 1), (2, 2, 2), (3, 3, 1), (4, 4, 2), (20, 5, 1), (22, 6, 2)],
              2: [(1, 7, 1), (2, 8, 2), (3, 9, 1), (4, 10, 2), (20, 11, 1), (20, 12, 2), (22, 13, 1)],
              3: [(1, 14, 1), (2, 15, 2), (20, 16, 1), (22, 17, 2)]}
    requests = {uid: [dict(request_id=f'{uid}:{i}', uid=uid, query_timestamp=t,
                          item_idx=20+i, label=float(i % 2), full_logit=0., reuse_logit=0.,
                          append_count_since_cutover=sum(20 <= event[0] < t for event in rows))
                      for i, t in enumerate([20, 21, 23, 23])] for uid, rows in events.items()}
    history = FoundationHistoryIndex({uid: tuple(np.asarray(col, dtype=np.int64) for col in zip(*rows))
                                     for uid, rows in events.items()})
    read_cost = Counter()
    expected = {}
    for cohort in ([1, 2], [3]):
        def replay(observer=None):
            stats = {name: Counter() for name in ('initial_history_hist', 'full_history_hist')}
            return snapshots(cohort, requests, history, parent, current, 20, max_length=4,
                             query_batch=3, stats=stats, append_band_size=4, observer=observer)

        observer = ViewObserver(policies, current_model=current)
        for plain, snap in zip(replay(), replay(observer), strict=True):
            native = snap.state.cache
            torch.testing.assert_close(native.k, plain.state.cache.k, rtol=0, atol=0)
            torch.testing.assert_close(native.v, plain.state.cache.v, rtol=0, atol=0)
            ids = torch.tensor([[event[1] for event in events[row['uid']]
                                 if event[0] < row['query_timestamp']][-4:] for row in snap.requests])
            fresh = compact.encode_cache(native, current.lookup_item_embeddings(ids))
            torch.testing.assert_close(snap.context['compact'].hidden, fresh.hidden, rtol=2e-5, atol=2e-6)
            assert 'compact' not in observer.kv_views
            assert not hasattr(observer.compact_views['compact'], 'k')
            assert set(vars(observer.compact_views['compact'])) == {'hidden', 'seq_len'}
            memory = {tag: dict(value) for tag, value in observer.memory.items()}
            observer.reading([0] * 20)  # Query expansion is temporary, never persistent state.
            assert observer.memory == memory
            assert memory['compact']['persistent_extra_to_native_ratio_peak'] == 1 / 6
            logits = score(current, native, snap.candidates, snap.query_deltas,
                           history_override=base.make_history_override(current, snap.context['mapped']))[0][:, 0]
            actual = score(current, native, snap.candidates, snap.query_deltas,
                           history_override=compact.make_history_override(current, native, snap.context['compact']))[0][:, 0]
            torch.testing.assert_close(actual, logits, rtol=2e-5, atol=2e-6)
            expected.update((row['request_id'], float(value)) for row, value in zip(snap.requests, logits, strict=True))
            for policy in policies:
                estimate = policy['adapter'].estimate_flops(batch=len(snap.requests), tokens=native.seq_len)
                read_cost[policy['tag']] += estimate['extra_history_read'] + estimate['candidate_reads']
    settings = dict(history_length=4, cohort_sizes={'medium': 2}, query_batches={'medium': 3},
                    attention_backend='torch', append_band_size=4)
    records, _, _ = score_unit(current, parent, history, requests, [1, 2, 3], 20,
                              policies, settings, 'medium', torch.device('cpu'), verify=False)
    native_peak = 2 * 2 * 2 * 4 * 192 * 4  # K/V * layers * owners * N * W * bytes.
    for policy in policies:
        rows, adapter = records[policy['tag']], policy['adapter']
        initial = adapter.estimate_flops(tokens=10)['token_transform']
        updates = adapter.estimate_flops(tokens=7)['token_transform']
        assert sum(row['view_initial_flops'] for row in rows) == initial
        assert sum(row['view_update_flops'] for row in rows) == updates
        assert sum(row['read_flops'] for row in rows) == read_cost[policy['tag']]
        assert sum(row['extra_history_read_flops'] + row['response_flops'] for row in rows) == read_cost[policy['tag']]
        assert sum(row['correction_flops'] for row in rows) == initial + updates + read_cost[policy['tag']]
        assert sum(row['item_lookup_bytes'] for row in rows) == 17 * 192 * 4
        assert max(row['native_kv_peak_bytes'] for row in rows) == native_peak
        fraction = 1 / 6 if policy['tag'] == 'compact' else 1.
        assert max(row['persistent_extra_peak_bytes'] for row in rows) == native_peak * fraction
        assert max(row['persistent_extra_to_native_ratio_peak'] for row in rows) == fraction
        torch.testing.assert_close(torch.tensor([row['hstu_logit'] for row in rows]),
            torch.tensor([expected[row['request_id']] for row in rows]), rtol=2e-5, atol=2e-6)
    assert max(row['shared_projection_bytes'] for row in records['compact']) == compact.projection_storage_bytes()
    setup = pending_setup_flops(policies, [None, None])
    assert setup == {'compact': compact.setup_flops()}
    assert pending_setup_flops(policies, [{'execution_setup_flops': setup}, None]) == {}


def test_execution_mode_is_bound_to_policy_signature(tmp_path):
    (tmp_path / 'calibration.pt').write_bytes(b'weights')
    metadata = dict(kind=ITEM_RESPONSE_KIND, configuration={'sha256': 'config'}, users_file_sha256='users',
                    weights_sha256=sha256(tmp_path / 'calibration.pt'), fit_uids=[1], validation_uids=[2],
                    cutover=20, history_length=1024, checkpoint_hashes={'v4': 'parent', 'v5': 'current'},
                    cost={'calibration_flops': 123}, initial_item_calibration={'weights_sha256': 'item'})
    (tmp_path / 'calibration.json').write_text(json.dumps(metadata))
    panel = dict(configuration={'sha256': 'config'}, users_file={'sha256': 'users'}, cutover=20,
                 sources={'parent': {'sha256': 'parent'}, 'current': {'sha256': 'current'}})
    groups = dict(design1_fit=[1], design1_validation=[2], design1_evaluation=[3])
    mapped = design_policy(tmp_path, 'same', panel, groups)
    compact = design_policy(tmp_path, 'same', panel, groups, execution_mode='compact_hidden')
    assert mapped['execution_mode'] == 'mapped_kv'
    assert compact['execution_mode'] == 'compact_hidden'
    assert mapped['calibration_flops'] == compact['calibration_flops'] == 123
    assert mapped['calibration'] == compact['calibration']
    assert signature({'policies': [mapped]}) != signature({'policies': [compact]})
    with pytest.raises(ValueError, match='execution_mode'):
        design_policy(tmp_path, 'same', panel, groups, execution_mode='unknown')
