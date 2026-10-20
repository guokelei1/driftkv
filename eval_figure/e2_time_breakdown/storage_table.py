"""Compute the explicitly simulated persistent-storage table; CPU only.

The native LaTeX table is copied into main.tex. This script never edits the
paper, measures GPU allocation, or changes the time-breakdown preview.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def storage_ledger(configuration, scenario):
    shape = configuration['persistent_storage']
    users = configuration['active_users']
    window = scenario['window_length']
    layers = shape['layers']
    heads = shape['heads']
    width = 12 + 6 * heads
    scalar_bytes = scenario['storage_scalar_bytes']
    replicas = scenario['storage_shared_replicas']

    native = users * window * layers * 2 * heads * shape['head_dimension'] * scalar_bytes
    codes = users * window * layers * width * scalar_bytes
    one_item_table = shape['catalogue_items'] * layers * width * scalar_bytes
    shared_tables = replicas * one_item_table * (
        2 + scenario['storage_old_release_tables_retained'])
    parameters = replicas * shape['shared_parameter_bundle_bytes']
    added = codes + shared_tables + parameters
    return {
        'configuration': f"{configuration['dataset']}-{configuration['scale']}",
        'native_kv_bytes': native,
        'single_code_buffer_bytes': codes,
        'fixed_feature_table_bytes': replicas * one_item_table,
        'current_item_table_bytes': replicas * one_item_table,
        'retained_old_item_table_bytes': replicas * one_item_table * scenario['storage_old_release_tables_retained'],
        'shared_parameter_bundle_bytes': parameters,
        'added_bytes': added,
        'total_cache_and_correction_bytes': native + added,
        'added_pct_native_kv': 100 * added / native,
    }


def latex_table(rows):
    lines = [
        r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrr@{}}',
        r'\toprule',
        r'Configuration & Native K/V & Added & Increase\\',
        r' & (TB) & (GB) & (\%)\\',
        r'\midrule',
    ]
    for row in rows:
        lines.append(
            f"{row['configuration']} & {row['native_kv_bytes'] / 1e12:.2f} & "
            f"{row['added_bytes'] / 1e9:.1f} & {row['added_pct_native_kv']:.1f}"
            + r'\\')
    lines.extend([r'\bottomrule', r'\end{tabular*}'])
    return '\n'.join(lines) + '\n'


def main():
    data = json.loads((ROOT / 'data.json').read_text())
    if data['meta']['synthetic'] is not True:
        raise ValueError('This table is a simulated layout preview.')
    rows = [storage_ledger(c, data['scenario']) for c in data['configurations']]
    ledger = {
        'meta': {'synthetic': True, 'scope': data['meta']['storage_scope'],
                 'units': 'Decimal GB=10^9 bytes, TB=10^12 bytes.'},
        'configurations': rows,
    }
    (ROOT / 'storage_ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    (ROOT / 'table_6_5_storage_simulated.tex').write_text(latex_table(rows))
    print(latex_table(rows), end='')


if __name__ == '__main__':
    main()
