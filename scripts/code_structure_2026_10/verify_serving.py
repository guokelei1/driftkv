#!/usr/bin/env python3
"""Check static hits/misses and structured incremental codes in FP32 on CPU."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]

import numpy as np
import torch
from code_structure_2026_10.structure import StructuredKVAdapter
from code_structure_2026_10.session import StructuredEncoder, ITEM_TABLES
from code_structure_2026_10.calibrate import ORIGINAL
from hstu_kvcache.design_one.lightweight_fullitem_kv import LightweightFullItemKVAdapter
from hstu_kvcache.models import HSTUKVCache


class Items:
    def lookup_item_embeddings(self, ids):
        return torch.sin((ids[..., None] % 991).float() * .017 + torch.arange(192) * .013)


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(17)
    payload = torch.load(ORIGINAL / 'v3/read_only/calibration.pt', weights_only=False)
    original = LightweightFullItemKVAdapter(**payload['config'])
    original.load_state_dict(payload['state_dict'])
    with np.load(ITEM_TABLES) as tables:
        warm = tables['v3_4096'][:3].tolist()
    items = warm + warm + [0, 0, 1, 1]
    cache = HSTUKVCache(torch.randn(6, 1, len(items), 192), torch.randn(6, 1, len(items), 192), len(items))
    unchanged = cache.k.clone(), cache.v.clone()
    reports = []
    for layout in ['group2', 'shared6_private7', 'shared12_private6']:
        adapter = StructuredKVAdapter.from_adapter(original, layout).eval()
        encoder = StructuredEncoder(adapter, Items(), 3)
        actual = encoder.encode(cache, items)
        wanted = adapter.encode_cache(cache, Items().lookup_item_embeddings(torch.tensor(items)[None]))
        error = float((actual.codes - wanted.codes).abs().max())
        if not torch.allclose(actual.codes, wanted.codes, atol=2e-5, rtol=2e-5):
            raise RuntimeError('static projection changed FP32 code function')
        prefix = HSTUKVCache(cache.k[:, :, :6], cache.v[:, :, :6], 6)
        entering = HSTUKVCache(cache.k[:, :, 6:], cache.v[:, :, 6:], 4)
        combined = torch.cat((encoder.encode(prefix, items[:6]).codes,
            encoder.encode(entering, items[6:]).codes), 2)[:, :, -7:]
        if not torch.allclose(combined, actual.codes[:, :, -7:], atol=2e-5, rtol=2e-5):
            raise RuntimeError('incremental encoding/FIFO differs from full encoding')
        if not torch.equal(cache.k, unchanged[0]) or not torch.equal(cache.v, unchanged[1]):
            raise RuntimeError('encoder mutated native K/V')
        expected = 2 * len(items) * adapter.row_kv_flops()
        if encoder.counters['row_kv_flops'] != expected:
            raise RuntimeError('row arithmetic counter differs from actual calls')
        reports.append(dict(layout=layout, max_code_error=error, counters=dict(encoder.counters),
            append_fifo_equivalent=True, native_unchanged=True))
    target = ROOT / 'results/code_structure_2026_10/serving_checks.json'
    target.write_text(json.dumps(dict(status='passed', checks=reports), indent=2) + '\n')
    print(json.dumps(dict(status='passed', max_error=max(r['max_code_error'] for r in reports))))


if __name__ == '__main__':
    main()
