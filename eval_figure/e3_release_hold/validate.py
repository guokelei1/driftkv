"""CPU check of the SYNTHETIC fixed-workload cost curve, not quality.

Replay the declared event sequence independently of vectorized accumulation.
This checks accounting, not real model FLOPs, hardware, or deployments.
"""
from pathlib import Path
import json
import numpy as np
from plot import correction_cost, cost_trajectory


def main():
    root = Path(__file__).resolve().parent
    data = json.loads((root/'data.json').read_text())
    n = data['scenario']['window_length']
    users = data['scenario']['active_population']
    query_cycle = data['scenario']['candidate_reads_per_append_cycle']
    ledger = cost_trajectory(data)
    rows, cost = n, ledger['cachetide_added_flops'][0]
    for appends in range(1, data['scenario']['max_appends']+1):
        queries = query_cycle[(appends-1) % len(query_cycle)]
        for _ in range(queries):
            cost += users * float(correction_cost(data, rows, 'score'))
        cost += users * float(correction_cost(data, rows, 'append'))
        rows = max(rows-1, 0)
        assert np.isclose(cost, ledger['cachetide_added_flops'][appends], rtol=1e-11)
    assert correction_cost(data, 0, 'score') == 0
    assert correction_cost(data, 0, 'append') == 0
    assert np.all(np.diff(ledger['added_percent']) >= -1e-10)
    assert np.allclose(ledger['added_percent'][n:], ledger['added_percent'][n])
    assert np.all((ledger['added_percent'] > 100) ==
                  (ledger['cachetide_total_flops'] > ledger['rebuild_total_flops']))
    result = {'synthetic_coefficients': True,
              'fixed_candidate_reads_per_append_cycle': query_cycle,
              'appends_checked': data['scenario']['max_appends'],
              'explicit_sequential_ledger_agrees': True,
              'empty_prefix_bypass_and_cost_parity_checked': True,
              'actual_model_or_quality_validation': False}
    (root/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f'PASS: {result["appends_checked"]} appends at fixed workload; cumulative costs and FIFO agree.')


if __name__ == '__main__':
    main()
