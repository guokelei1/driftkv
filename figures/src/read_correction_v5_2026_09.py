"""Saved-only 15-panel comparison of v5 candidates and retained Q/H controls."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'figures/src'), str(ROOT / 'scripts')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from read_correction_common import SCALES, EDGES, write_csv, paired, sha256, PAIRED_INTEGERS
from read_correction_v3.summarize import validate_point

RESULTS = ROOT / 'results/read_correction_2026_09'
EXPECTED = {(s, e) for s in SCALES for e in EDGES}
STYLES = {
    'q_v2': dict(color='#7F9BB5', marker='o', line='--', label='Q-v2: retained C128–1024'),
    'h_v1': dict(color='#272727', marker='D', line='none', label='H-v1: C512 low-cost control'),
    'h_v4': dict(color='#9A4D8E', marker='P', line='none', label='H-v4: all tokens, C128'),
    'q_v5': dict(color='#0F4D92', marker='o', line='-', label='Q-v5: selected feature'),
    'h_v5_all': dict(color='#52894E', marker='o', line='-', label='H-v5: all tokens'),
    'h_v5_old': dict(color='#B64342', marker='o', line='-', label='H-v5: inherited tokens'),
}
BUDGET_MARKERS = {128: 'o', 256: 's', 512: '^', 1024: 'D'}


def collect(input_root, q_summary, v1_root, v4_root):
    rows, sources, references, evidence = [], [], {}, []
    plan_path = input_root / 'plan.json'
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else None
    if plan:
        sources.append({'path': str(plan_path.resolve()), 'sha256': sha256(plan_path)})

    def read(path):
        report = json.loads(path.read_text())
        sources.append({'path': str(path.resolve()), 'sha256': sha256(path)})
        return report

    def add(original, series, path):
        point = copy.deepcopy(original)
        validate_point(point)
        key = point['scale'], point['edge']
        if key not in EXPECTED:
            raise ValueError(f'unexpected scale/edge: {key}')
        if key in references:
            paired(point, references[key])
        else:
            references[key] = point
        point.update(series=series, source_path=str(path.resolve()), source_sha256=sha256(path),
            calibration_percent=100 * point['calibration_flops'] / point['full_minus_reuse_flops'],
            inference_percent=100 * point['correction_flops'] / point['full_minus_reuse_flops'])
        rows.append(point)

    old = read(q_summary)
    if old['status'] != 'complete':
        raise ValueError('retained Q-v2 controls must be complete')
    for scale, edge in sorted(EXPECTED):
        query = [p for p in old['points'] if p['kind'] == 'measurement' and p['method'] == 'query_only'
                 and (p['scale'], p['edge']) == (scale, edge)]
        if len(query) != 4 or {p['budget'] for p in query} != {128, 256, 512, 1024}:
            raise ValueError('retain all four original Q-v2 budgets on every edge')
        for point in query:
            add(point, 'q_v2', q_summary)
        path = v1_root / scale / edge / 'summary.json'
        report = read(path)
        history = [p for p in report['points'] if p['kind'] == 'measurement'
                   and p['method'] == 'history_conditioned' and p['budget'] == 512]
        if report['status'] != 'complete' or len(history) != 1:
            raise ValueError('fixed H-v1 C512 control missing')
        add(history[0], 'h_v1', path)
        path = v4_root / 'evaluation' / scale / edge / 'summary.json'
        report = read(path)
        history = [p for p in report['points'] if p['variant'] == 'map_all']
        if report['status'] != 'complete' or report['probe_only'] or report['users'] != 3000 or len(history) != 1:
            raise ValueError('retained H-v4 must be the complete panel')
        add(history[0], 'h_v4', path)

    complete, signatures, point_sets = set(), set(), set()
    for path in sorted((input_root / 'evaluation').glob('*/v*_to_v*/summary.json')):
        report = read(path)
        key = report['scale'], report['edge']
        if key not in EXPECTED or key in complete:
            raise ValueError(f'unexpected or duplicate v5 edge: {key}')
        if report['status'] != 'complete':
            continue
        if report['users'] != 3000 or report['probe_only'] or report['partition'] != 'evaluation':
            raise ValueError('small probes cannot enter the v5 population figure')
        points = report['points']
        point_set = {(p['method'], p['variant'], p['budget']) for p in points}
        if len(point_set) != len(points):
            raise ValueError('duplicate v5 method/variant/budget point')
        if plan:
            query_variant = plan['query']['feature_mode']
            if plan['query'].get('joint_epochs', 0):
                query_variant += f'_joint{plan["query"]["joint_epochs"]}'
            query_variant = plan['query'].get('variant', query_variant)
            desired = {('query_only', query_variant, n) for n in plan['query']['budgets']}
            desired |= {('history_conditioned', v, n) for n in plan['history']['budgets']
                        for v in ('map_all', 'map_old_prefix')}
            if point_set != desired:
                raise ValueError('completed edge does not retain every planned v5 point')
        point_sets.add(tuple(sorted(point_set)))
        for point in points:
            if (point['scale'], point['edge']) != key:
                raise ValueError('v5 point and containing edge disagree')
            series = ('q_v5' if point['method'] == 'query_only' else
                      'h_v5_all' if point['variant'] == 'map_all' else 'h_v5_old')
            add(point, series, path)
            score = report['scores'][point['tag']]
            score_path = Path(score['path'])
            if not score_path.is_absolute():
                score_path = ROOT / score_path
            if sha256(score_path) != score['sha256'] or score['rows'] != point['requests']:
                raise ValueError(f'v5 retained scores changed: {score_path}')
        for n in {p['budget'] for p in points if p['method'] == 'history_conditioned'}:
            pair = [p for p in points if p['method'] == 'history_conditioned' and p['budget'] == n]
            if len(pair) != 2 or any(pair[0][field] != pair[1][field]
                                   for field in ('calibration_flops', 'correction_flops', 'extra_flops')):
                raise ValueError('H all/old-prefix must retain the same actual dense-map cost')
        signatures.add(json.dumps(report['inputs']['execution_sources'], sort_keys=True))
        evidence.append({'scale': key[0], 'edge': key[1], 'input_signature': report['input_signature'],
            'inputs': {k: v for k, v in report['inputs'].items() if k != 'uids'},
            'scores': report['scores'], 'controls': report['controls']})
        complete.add(key)
    if len(signatures) > 1 or len(point_sets) > 1:
        raise ValueError('v5 edges mix execution sources or candidate sets')
    return rows, complete, sources, evidence, plan


def draw(rows, complete, plan):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 1.3,
        'legend.frameon': False, 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    ylim = {}
    for scale in SCALES:
        values = [0., 100.] + [p['recovery_percent'] for p in rows if p['scale'] == scale]
        pad = max(5., .055 * (max(values) - min(values)))
        ylim[scale] = min(values) - pad, max(values) + pad
    # Keep the requested 0..100 axis whenever possible; never hide an over-budget result.
    largest_cost = max((p['relative_flops_percent'] for p in rows), default=0.)
    xmax = 1.03 * largest_cost if largest_cost > 100 else 100.
    fig, axes = plt.subplots(3, 5, figsize=(19, 11.8), sharex=True, sharey='row')
    for r, (scale, label) in enumerate(SCALES.items()):
        for c, edge in enumerate(EDGES):
            ax = axes[r, c]
            panel = [p for p in rows if (p['scale'], p['edge']) == (scale, edge)]
            for series, style in STYLES.items():
                points = sorted((p for p in panel if p['series'] == series),
                                key=lambda p: p['budget'] if isinstance(p['budget'], (int, float)) else 0)
                if not points:
                    continue
                x, y = ([p[field] for p in points] for field in ('relative_flops_percent', 'recovery_percent'))
                if len(points) > 1 and style['line'] != 'none':
                    ax.plot(x, y, color=style['color'], linestyle=style['line'], linewidth=1.65, zorder=2)
                for point in points:
                    marker = BUDGET_MARKERS.get(int(point['budget']), style['marker']) if series.startswith(('q_v5', 'h_v5')) else style['marker']
                    ax.scatter(point['relative_flops_percent'], point['recovery_percent'],
                        color=style['color'] if series != 'h_v5_old' else None,
                        facecolors='white' if series == 'h_v5_old' else style['color'],
                        edgecolors=style['color'], marker=marker, s=64 if series == 'h_v1' else 49,
                        linewidths=1.35, zorder=6 if series == 'h_v1' else 4)
            ax.axhline(0, color='#CFCECE', linewidth=.8, zorder=0)
            ax.axhline(100, color='#767676', linewidth=.8, linestyle=':', zorder=0)
            ax.scatter(0, 0, marker='X', s=44, color='#4D4D4D', clip_on=False, zorder=7)
            ax.scatter(100, 100, marker='*', s=78, color='#4D4D4D', clip_on=False, zorder=7)
            ax.set(xlim=(0, xmax), ylim=ylim[scale], xticks=(0, 25, 50, 75, 100))
            ax.tick_params(direction='out', length=3.5, width=1, labelsize=10)
            if r == 0:
                ax.set_title(edge.replace('_to_', ' → ').upper(), fontsize=13, pad=10)
            if c == 0:
                ax.set_ylabel(label, fontsize=12, labelpad=10)
            if (scale, edge) not in complete:
                ax.text(.97, .035, 'v5 pending', transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=9, color='#B64342', bbox=dict(facecolor='white', edgecolor='none', pad=2))
    state = 'complete' if complete == EXPECTED else 'PARTIAL'
    fig.suptitle(f'Read correction across 15 adjacent release edges — {state}: v5 {len(complete)}/15',
                 fontsize=17, y=.974)
    fig.supylabel('Full−Reuse AUC gap recovered (%)', fontsize=13, x=.009)
    fig.supxlabel('Extra compute / (Full − Reuse) compute (%)', fontsize=13, y=.198)
    handles = [Line2D([], [], color=s['color'], marker=s['marker'], markersize=7,
        markerfacecolor='white' if k == 'h_v5_old' else s['color'], linestyle=s['line'], label=s['label'])
        for k, s in STYLES.items()
        if not (plan and plan['history'].get('reuse_v4') and k.startswith('h_v5'))]
    handles += [Line2D([], [], color='#4D4D4D', marker=m, linestyle='none', markersize=8, label=t)
                for m, t in (('X', 'Reuse (0, 0)'), ('*', 'Full (100, 100)'))]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .105), ncol=4,
               fontsize=10.5, columnspacing=1.7, handlelength=2.3)
    budgets = sorted({int(p['budget']) for p in rows if p['series'].startswith(('q_v5', 'h_v5'))})
    if not budgets and plan:
        budgets = sorted(set(plan['query']['budgets'] + plan['history']['budgets']))
    if budgets:
        fig.legend(handles=[Line2D([], [], color='#555555', marker=BUDGET_MARKERS[n], linestyle='none',
            label=f'v5 C{n}') for n in budgets], loc='lower center', bbox_to_anchor=(.5, .069), ncol=4, fontsize=10)
    text = ''
    if plan:
        q = plan['query']
        joint = f', {q["joint_epochs"]} joint epochs' if q.get('joint_epochs', 0) else ''
        h = plan['history']
        if h.get('reuse_v4'):
            history_text = 'H: retained complete v4 C128 results.'
        else:
            history_state = h['state_mode'] + (' / old-heavy' if h.get('mix_profile') == 'old_heavy' else '')
            history_text = f'H-v5: {history_state}.'
        text = f'Q-v5: {q["feature_mode"]}, {q["state_mode"]}{joint}. {history_text} '
    fig.text(.5, .048, text + 'Calibration + additional inference FLOPs; 3,000 fixed development users per edge.', ha='center', fontsize=10)
    footnote = ('Negative and above-100% recovery remain visible. Prior cheap H-v1 control is retained on every edge.'
                if plan and plan['history'].get('reuse_v4') else
                'Negative and above-100% recovery remain visible. H scopes share dense-map cost; no producer-gating compute discount.')
    fig.text(.5, .027, 'Y-axis shared within each scale row; ranges differ across scales. ' + footnote,
             ha='center', fontsize=9.5)
    fig.tight_layout(rect=(.018, .235, .996, .952), h_pad=1.8, w_pad=1.2)
    return fig, ylim, xmax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=RESULTS / 'v5/population_run')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'figures/out/read_correction_2026_09/v5')
    parser.add_argument('--q-summary', type=Path, default=RESULTS / 'v2/summary.json')
    parser.add_argument('--v1-root', type=Path, default=RESULTS / 'runtime/development/v1')
    parser.add_argument('--v4-root', type=Path, default=RESULTS / 'v4/population_run')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    rows, complete, sources, evidence, plan = collect(args.input_root, args.q_summary, args.v1_root, args.v4_root)
    if args.require_complete and complete != EXPECTED:
        raise ValueError(f'only {len(complete)}/15 v5 edges are complete')
    fig, ylim, xmax = draw(rows, complete, plan)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    means = []
    for scale in SCALES:
        for series in STYLES:
            subset = [p for p in rows if p['scale'] == scale and p['series'] == series]
            for budget in sorted({str(p['budget']) for p in subset}):
                group = [p for p in subset if str(p['budget']) == budget]
                means.append({'scale': scale, 'series': series, 'budget': budget, 'edges': len(group),
                    **{f: sum(p[f] for p in group) / len(group) for f in
                       ('baseline_auc', 'recovery_percent', 'relative_flops_percent', 'calibration_percent', 'inference_percent')}})
    manifest = {'status': 'complete' if complete == EXPECTED else 'partial', 'completed_v5_edges': len(complete),
        'expected_edges': 15, 'measurement_points': len(rows), 'sources': sources, 'v5_evidence': evidence,
        'generator_path': str(Path(__file__).resolve()), 'generator_sha256': sha256(__file__),
        'xlim': [0, xmax], 'ylim_by_scale': ylim, 'shared_axes': {'x': 'all', 'y': 'within scale row'}, 'equal_edge_means': means,
        'paired_controls_checked': [*PAIRED_INTEGERS, 'full_auc', 'reuse_auc'],
        'missing_v5_edges': [dict(scale=s, edge=e) for s, e in sorted(EXPECTED - complete)],
        'evaluation_role': 'outcome-conditioned development exploration',
        'cost_convention': '(complete calibration + additional inference) / (Full-history recompute - rolling Reuse append)',
        'plots': ['overview_15panels.png', 'overview_15panels.pdf'], 'data': 'overview_15panels.csv'}
    with tempfile.TemporaryDirectory(prefix='.v5_plot_', dir=args.output_dir) as directory:
        stage = Path(directory)
        for suffix in ('png', 'pdf'):
            fig.savefig(stage / f'overview_15panels.{suffix}', dpi=300)
        plt.close(fig)
        write_csv(rows, stage / 'overview_15panels.csv')
        manifest['artifacts'] = {p.name: sha256(p) for p in stage.iterdir()}
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        for path in sorted(stage.iterdir(), key=lambda p: p.name == 'manifest.json'):
            path.replace(args.output_dir / path.name)
    print(json.dumps({'status': manifest['status'], 'v5_edges': len(complete), 'measurement_points': len(rows),
                      'output_dir': str(args.output_dir)}))


if __name__ == '__main__':
    main()
