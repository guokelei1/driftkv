"""SIMULATED: a single release held fixed through two window turnovers.

Quality values in data.json are invented layout examples, not predictions.
Cost values follow the explicit synthetic coefficients and causal event order.
No model weights, real labels, experiment outputs or GPU are used here.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from style import COLORS, finish, load_data


def correction_cost(data, prefix_rows, kind):
    """Per candidate or per behavior, including nonempty-prefix fixed work.

    The preview uses equal coefficients for the two paths, explicitly, while
    their real execution shapes and operation counts must be charged separately.
    """
    p = data['flop_coefficients']
    rows = np.asarray(prefix_rows)
    return np.where(rows > 0, p[f'correction_{kind}_fixed_flops'] +
                    p[f'correction_{kind}_flops_per_inherited_row'] * rows, 0.0)


def cost_trajectory(data):
    """One up-front rebuild vs one preparation, then matched native events.

    Each group of candidate reads occurs before its next behavior append.
    The initial trigger read has already happened and is charged. At a=0 no
    subsequent reads or appends have occurred; at a=N the last old row leaves.
    All common native work is retained, although it cancels in the crossing.
    """
    case = data['scenario']
    p = data['flop_coefficients']
    n, users = case['window_length'], case['active_population']
    appends = np.arange(case['max_appends'] + 1)
    reads_per_append = np.resize(case['candidate_reads_per_append_cycle'], case['max_appends'])
    prefix_before = np.maximum(n - appends[1:] + 1, 0)
    # Both candidate and behavior reads use the same remaining old prefix.
    recurring = np.r_[0.0, np.cumsum(
        reads_per_append * correction_cost(data, prefix_before, 'score') +
        correction_cost(data, prefix_before, 'append'))]
    fixed = sum(data['fixed_cost_flops'].values())
    catalogue = (case['shared_table_replicas'] * case['catalogue_items'] *
                 p['catalogue_flops_per_item'])
    preparation = p['materialize_fixed_flops'] + p['materialize_flops_per_row'] * n
    rebuild_per_user = (p['rebuild_linear_flops_per_row'] * n +
                        p['rebuild_quadratic_flops_per_row_squared'] * n * n)
    rebuild_budget = users * rebuild_per_user
    trigger = case['initial_trigger_candidate_reads'] * correction_cost(data, n, 'score')
    added = fixed + catalogue + users * (preparation + trigger + recurring)
    score = p['native_score_fixed_flops'] + p['native_score_flops_per_history_row'] * n
    write = p['native_append_fixed_flops'] + p['native_append_flops_per_history_row'] * n
    candidate_reads = np.r_[0, np.cumsum(reads_per_append)]
    common = users * (case['initial_trigger_candidate_reads'] * score +
                      candidate_reads * score + appends * write)
    return {
        'appends': appends,
        'later_candidate_reads': candidate_reads,
        'remaining_inherited_rows': np.maximum(n - appends, 0),
        'rebuild_budget_flops': rebuild_budget,
        'shared_flops': fixed + catalogue,
        'materialization_flops': users * preparation,
        'initial_trigger_correction_flops': float(users * trigger),
        'added_read_write_flops': users * recurring,
        'cachetide_added_flops': added,
        'common_native_flops': common,
        'cachetide_total_flops': common + added,
        'rebuild_total_flops': common + rebuild_budget,
        'added_percent': 100 * added / rebuild_budget,
        'total_ratio': (common + added) / (common + rebuild_budget),
    }


def draw_trend(data, ledger):
    """Pair cost and recovery at each append checkpoint on one trajectory.

    Direct labels identify cumulative appends. The small arrow shows service
    progression, not a budget choice; no cost threshold is visually privileged.
    """
    n = data['scenario']['window_length']
    q = data['quality_preview']
    checkpoints = np.asarray(q['appends'], dtype=int)
    teacher = np.asarray(q['auc']['Rebuild'])
    reuse = np.asarray(q['auc']['Reuse'])
    recovery = 100*(np.asarray(q['auc']['CacheTide'])-reuse)/(teacher-reuse)
    appends = checkpoints/n
    costs = ledger['added_percent'][checkpoints]
    color = COLORS['CacheTide']
    plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.5,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'axes.linewidth': .65})
    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    fig.subplots_adjust(left=.145, right=.98, bottom=.23, top=.96)
    ax.plot(costs, recovery, color=color, lw=1.5, zorder=3)
    # Sparse stage labels replace large parity guides or shaded regions.
    shown = [0, 3, 4, 5, 6, 8]
    offsets = [(5, 6), (4, 6), (5, 5), (5, 5), (5, 3), (5, -7)]
    for i, offset in zip(shown, offsets):
        ax.plot(costs[i], recovery[i], 'o', color=color, ms=3.5,
                mec='white', mew=.6, zorder=4)
        ax.annotate(f'{appends[i]:g}N', (costs[i], recovery[i]),
                    xytext=offset, textcoords='offset points',
                    ha='left', color=color, fontsize=7)
    ax.annotate('', xy=(costs[4], recovery[4]), xytext=(costs[3], recovery[3]),
                arrowprops={'arrowstyle': '-|>', 'lw': 1.2, 'color': color,
                            'shrinkA': 5, 'shrinkB': 5, 'mutation_scale': 8})
    ax.set(xlim=(0, 122), ylim=(0, 100), xticks=[0, 25, 50, 75, 100],
           yticks=[0, 25, 50, 75, 100], xlabel='Added FLOPs (% of Rebuild)',
           ylabel='Recovery (%)')
    ax.tick_params(length=2.5, pad=3, color='#737D87')
    for spine in ax.spines.values():
        spine.set_color('#737D87')
    ax.grid(False)
    # Retain numerical diagnostics in the ledger, without displaying or
    # treating this particular synthetic crossing as a deployment threshold.
    crossings = np.flatnonzero(ledger['added_percent'] >= 100)
    crossing = int(crossings[0]) if len(crossings) else None
    parity_recovery = (float(np.interp(crossing/n, appends, recovery))
                       if crossing is not None else None)
    return fig, {'first_break_even_append': crossing,
                 'appends_per_window_at_parity': crossing/n if crossing is not None else None,
                 'interpolated_mock_recovery_at_parity_percent': parity_recovery,
                 'interpolation_is_not_measured_quality': True}


def main():
    data = load_data(__file__)
    case = data['scenario']
    n = case['window_length']
    ledger = cost_trajectory(data)
    fig, parity = draw_trend(data, ledger)
    finish(fig, __file__, '6-3-hold', 'Recovery and cumulative cost while holding one release fixed',
           layout=False, paper_column=True)
    rows = []
    for a in sorted(set([0, 20, *data['quality_preview']['appends']])):
        row = {'appends': a,
               'remaining_inherited_rows': int(ledger['remaining_inherited_rows'][a])}
        for key in ['shared_flops', 'materialization_flops',
                    'initial_trigger_correction_flops', 'rebuild_budget_flops']:
            row[key] = float(ledger[key])
        for key in ['added_read_write_flops', 'cachetide_added_flops',
                    'common_native_flops', 'cachetide_total_flops',
                    'rebuild_total_flops', 'added_percent', 'total_ratio']:
            row[key] = float(ledger[key][a])
        rows.append(row)
    output = {'synthetic': True, 'configuration': case['configuration'],
              'window_length': n, 'active_population': case['active_population'],
              'quality_is_independent_layout_example': True,
              'fixed_workload': {
                  'candidate_reads_per_append_cycle': case['candidate_reads_per_append_cycle'],
                  'initial_trigger_charged': True,
                  'event_order': case['event_order'],
                  **parity,
                  'scope': 'Complete ADDED work; common native service retained separately in full totals.'},
              'checkpoints': rows}
    Path(__file__).with_name('model_ledger.json').write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
