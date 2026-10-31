#!/usr/bin/env python3
"""Match actual GPU code lifecycle counters to the retained cost simulator."""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]

import numpy as np
from experiments.archive.code_structure.evaluate import PANEL
from experiments.archive.code_structure.session import ITEM_TABLES
from hstu_kvcache.evaluation.panels import load_panel, load_history
from experiments.archive.partial_refresh.round2.factored_cost import simulate_factored
from hstu_kvcache.common.artifacts import write_json


def run(args):
    saved = json.loads((args.quality / 'summary.json').read_text())
    users = saved['selected_uids']
    panel, _, requests = load_panel(PANEL)
    histories = load_history(PANEL, requested_uids=users)
    masks = {}
    with np.load(ITEM_TABLES) as tables:
        for v in ['v3', 'v4', 'v5']:
            mask = np.zeros(panel['model_config']['num_items'], dtype=bool)
            mask[tables[f'{v}_4096']] = True
            masks[v] = {4096: mask}
    expected = Counter()
    for uid in users:
        rows = simulate_factored(histories.rows[uid], requests[uid],
            dict(v3=1., v4=1., v5=1.), masks)
        for release in rows.values():
            c, cache = release['counts'], release['cache_counts'][4096]
            for key in ['first_read_encode_rows', 'append_encode_rows', 'corrected_requests',
                'corrected_token_query_pairs']:
                expected[key] += c.get(key, 0)
            expected['encode_calls'] += c.get('first_read_encode_calls', 0) + c.get('append_encode_calls', 0)
            expected['unique_item_hits'] += cache.get('unique_item_hits', 0)
            expected['unique_item_misses'] += cache.get('unique_item_misses', 0)
    actual = saved['actual_service_counters']
    mismatches = {key: dict(expected=value, actual=actual.get(key, 0))
        for key, value in expected.items() if value != actual.get(key, 0)}
    report = dict(status='passed' if not mismatches else 'failed', users=len(users),
        expected=dict(expected), actual={key: actual.get(key, 0) for key in expected}, mismatches=mismatches,
        meaning='Executed GPU initialization/appends/reads and static item hit/miss shapes agree with the same simulator used for fixed30k costing.')
    write_json(args.quality / 'cost_shape_check.json', report)
    print(json.dumps(report))
    if mismatches:
        raise RuntimeError('candidate serving shapes differ from cost ledger')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quality', type=Path, required=True)
    run(p.parse_args())
