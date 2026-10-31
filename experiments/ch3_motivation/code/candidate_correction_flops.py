"""Two forward-cost curves at a fixed real cache snapshot; no GPU execution."""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT/'experiments/ch3_motivation/runs/real_quality_angles/attempt_1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive(output):
    pass  # imports use the installed package / repository namespace
    from dataclasses import asdict
    from experiments.ch3_motivation.code.probe_variants import load_variant
    from experiments.ch3_motivation.code.probe_cost import cached_preparation_flops, padded_read_extra_flops
    from experiments.ch3_motivation.code.simple_kv_maps import load_maps
    from hstu_kvcache.evaluation.cost import CostModel
    population_path = RUN/'population.json'
    population = json.loads(population_path.read_text())
    checkpoint = ROOT/'experiments/ch3_motivation/runs/insight_mechanisms/attempt_1/fitting/edge_2/f.pt'
    f, fitting = load_variant(checkpoint, 'cpu')
    cfg = f.cfg
    n = 1024
    actual_uid = next(u for u in population['users']
        if population['edges']['2']['user_metadata'][str(u)]['history_count'] >= n)
    assert cfg.num_layers == 10 and cfg.num_heads == 10 and cfg.head_dim == 32
    assert cfg.row_dim == 384 and cfg.hidden_width == 64 and cfg.depth == 1
    once = cached_preparation_flops(f, n)
    rebuild = CostModel(320, 10, 10, 'triton').full_cache(n)
    rows = []
    for q in [1,4,8,16,32,64,230]:
        # Same executable probe geometry. Stage changes only the analytical
        # branch; no weights are fitted and no quality is claimed for h here.
        f.stage = 'fgh'
        gh = padded_read_extra_flops(f, n, q)
        f.stage = 'fg'
        g = padded_read_extra_flops(f, n, q)
        h = gh-g
        rows.append(dict(candidates=q, primary_curve=q<=64, history_rows=n,
            once_mapping_flops=once, repeated_pair_h_flops=h,
            once_mapping_plus_pair_h_flops=once+h, once_rebuild_flops=rebuild,
            once_mapping_ratio_rebuild=once/rebuild,
            once_plus_pair_h_ratio_rebuild=(once+h)/rebuild,
            once_plus_pair_h_ratio_once_mapping=(once+h)/once))
    assert all(r['repeated_pair_h_flops'] == r['candidates']*rows[0]['repeated_pair_h_flops'] for r in rows)
    assert all(r['once_mapping_plus_pair_h_flops'] > once for r in rows)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'points.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    maps_path = RUN/'simple_maps/edge_2/simple_kv_maps.pt'
    linear = load_maps(maps_path,'cpu')['kv_linear']
    info=dict(status='complete',physical_gpu_used=False,format='fixed_snapshot_added_correction_flops_v1',
        geometry=asdict(cfg),native_hidden_size=320,history_rows=n,reference_uid=actual_uid,
        reference_uid_rule='First frozen resident with1024 cutover rows; no quality-based choice.',
        branch_dimensions=dict(f_input=cfg.row_dim,h_input=cfg.row_dim+cfg.head_dim,
            hidden_width=cfg.hidden_width,output=2*cfg.head_dim,heads_times_layers=100),
        curves=['Once candidate-independent F materialization','The same F materialization plus full per-candidate/per-row h'],
        once_mapping_flops=once,once_rebuild_flops=rebuild,
        linear_once_mapping_flops=linear.materialization_flops(n),
        actual_f_shared_fitting_flops=fitting['cumulative_calibration_flops_estimate'],
        fitting_scope='Recorded separately; excluded from both forward curves. No h fitting or quality claim.',
        native_work='Both alternatives also pay identical native reads; excluded from added correction FLOPs.',
        scope='Same frozen1024-row cutover snapshot, varying candidate count only. No appends/FIFO, service throughput, wall-time, or complete-release cost claim. h cost excludes the separate g branch; F includes shared row-feature preparation.',
        reference230='Algebraic extrapolation assuming the same snapshot; not a measured230-request release trajectory.',
        points=rows,source_sha256={str(p):sha(p) for p in [Path(__file__),checkpoint,population_path,maps_path,
            ROOT/'experiments/ch3_motivation/code/probe_cost.py',ROOT/'experiments/ch3_motivation/code/probes.py',
            ROOT/'experiments/ch3_motivation/code/probe_variants.py',ROOT/'src/hstu_kvcache/methods/baselines/protocol/cost.py']})
    (output/'summary.json').write_text(json.dumps(info,indent=2)+'\n')
    (output/'report.md').write_text(
        '# Fixed-snapshot correction FLOPs\n\n'
        'Actual Large geometry:10 layers,10 heads,32 dimensions/head;1024 retained rows. '
        'F uses384→64→64; h adds the32-dimensional query and uses416→64→64. '
        'The h curve uses the existing executable analytical graph at this geometry; it is not quality evidence for a trained h.\n\n'
        f'Once F: {once/1e9:.3f} GFLOPs; once Rebuild: {rebuild/1e9:.3f} GFLOPs. '
        f'Q16 F+h: {rows[3]["once_mapping_plus_pair_h_flops"]/1e9:.3f} GFLOPs '
        f'({rows[3]["once_plus_pair_h_ratio_rebuild"]:.2f}× Rebuild); '
        f'Q64: {rows[5]["once_plus_pair_h_ratio_rebuild"]:.2f}× Rebuild.\n\n'
        'Both curves exclude common native query reads and calibration fitting. '
        'The candidate-pair curve includes the same once-only F preparation plus repeated h, excluding separate g. '
        'This motivates sharing row preparation and keeping per-candidate work small; it does not prove a particular rank or gate. '
        'Q230 is a separate fixed-history extrapolation, not complete release accounting.\n')
    print(json.dumps({k:info[k] for k in ['once_mapping_flops','once_rebuild_flops','linear_once_mapping_flops']}))


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=json.loads((output/'summary.json').read_text())
    rows=[r for r in summary['points'] if r['primary_curve']]
    q=[r['candidates'] for r in rows]
    fig,ax=plt.subplots(figsize=(7.7,4.7))
    ax.plot(q,[r['once_mapping_ratio_rebuild'] for r in rows],'o-',label='Once KV mapping')
    ax.plot(q,[r['once_plus_pair_h_ratio_rebuild'] for r in rows],'s-',label='Once mapping + full candidate-row correction')
    ax.axhline(1,color='.55',ls='--',lw=1,label='Once-Rebuild reference')
    ax.set(xlabel='Candidates reading the same cache snapshot',
        ylabel='Added correction FLOPs / once-Rebuild FLOPs',xticks=q,
        title='Full candidate-row correction repeats expensive work')
    ax.grid(alpha=.18);ax.legend(frameon=False,loc='upper left')
    fig.text(.5,.035,'Large10L / H320;1024 rows; W64. Analytical forward work only.\n'
        'Shared fitting and common native reads excluded; no quality or timing claim.',ha='center',fontsize=9,color='.35')
    fig.tight_layout(rect=(0,.1,1,1))
    for extension in ['png','pdf']:fig.savefig(output/f'candidate_correction_flops.{extension}',dpi=180)
    plt.close(fig)
    (output/'plot_manifest.json').write_text(json.dumps(dict(source_sha256=sha(__file__),
        summary_sha256=sha(output/'summary.json'),
        artifacts={p.name:sha(p) for p in [output/'candidate_correction_flops.png',output/'candidate_correction_flops.pdf']}),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=RUN/'candidate_flops')
    parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args()
    if args.plot_only:plot(args.output)
    else:derive(args.output)
