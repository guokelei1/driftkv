#!/usr/bin/env python3
"""Post-hoc SVD geometry of centered candidate-by-read-output residuals.

This is not the historical E6 h(candidate, history-row) truncation.  These
matrices have 16 candidate rows and32 READ-OUTPUT columns, so centering itself
caps rank at15.  No factor is fitted on calibration users or deployed.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from experiments.ch3_motivation.code.analyze_candidate_residual_probe import blocks, bootstrap_weights

RANKS = (1, 2, 4, 8)
PANELS = ('broad16', 'near16')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def svd_energies(matrices):
    """Return total and best-r retained energies for [...,Q,D] matrices.

    Recenter in float64 only to remove the collector's float32 roundoff.
    Exactly zero centered residuals remain zero and receive undefined ratios.
    """
    x = np.asarray(matrices, dtype=np.float64)
    x = x - x.mean(-2, keepdims=True)
    singular = np.linalg.svd(x, compute_uv=False)
    energy = singular ** 2
    total = energy.sum(-1)
    retained = np.stack([energy[..., :min(rank, energy.shape[-1])].sum(-1)
                         for rank in RANKS], -1)
    return total, retained


def divide(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator>0)


def estimate(values, weights):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    with np.errstate(invalid='ignore', divide='ignore'):
        draws = (weights @ np.where(valid, values, 0.)) / (weights @ valid.astype(float))
    good = draws[np.isfinite(draws)]
    return dict(mean=float(values[valid].mean()) if valid.any() else None,
        ci95=np.quantile(good, [.025, .975]).tolist() if len(good) else None,
        valid_users=int(valid.sum()), undefined_users=int((~valid).sum()),
        quality_use_permitted=bool(valid.sum()>=512), invalid_bootstrap_draws=int(len(draws)-len(good)))


def flat(prefix, s):
    lo, hi = s['ci95'] or (None, None)
    return {prefix:s['mean'], prefix+'_ci95_low':lo, prefix+'_ci95_high':hi,
            prefix+'_valid_users':s['valid_users'], prefix+'_undefined_users':s['undefined_users']}


def analyze(input_root, *, replicates=500, seed=17):
    started = time.perf_counter(); root = Path(input_root).resolve()
    config = json.loads((root/'configuration.json').read_text())
    runner = json.loads((root/'summary.json').read_text())
    if (config['format']!='candidate_residual_probe_v1' or runner['status']!='complete'
            or config['panels']!=list(PANELS) or runner['feedback_labels_used'] or runner['quality_auc_computed']):
        raise ValueError('requires the completed label-free fixed-panel collector')
    geometry=config['geometry']; layers,heads,dim=(geometry[k] for k in ('layers','heads','head_dim'))
    total_parts, retained_parts, original_parts, uids = [], [], [], []
    maximum_recentering_mean = 0.
    for data in blocks(root, 'main', config, runner):
        target = data['read_target']
        expected=(len(data['uid']),2,16,layers,heads,dim)
        if target.shape != expected or not np.isfinite(target).all():
            raise ValueError('read residual shape/nonfinite values differ from geometry')
        matrix = target.transpose(0,1,3,4,2,5)
        maximum_recentering_mean=max(maximum_recentering_mean,float(np.abs(matrix.mean(-2, dtype=np.float64)).max()))
        total, retained = svd_energies(matrix)
        original = data['read_total_mse'].astype(np.float64)*16*dim
        if not np.isfinite(original).all() or (original<0).any() or np.any(total>original*(1+5e-6)+1e-24):
            raise ValueError('centered energy must be bounded by original local read residual energy')
        total_parts.append(total);retained_parts.append(retained);original_parts.append(original)
        uids.extend(map(int,data['uid']))
    total=np.concatenate(total_parts); retained=np.concatenate(retained_parts); original=np.concatenate(original_parts)
    if uids!=config['roles']['main'] or len(set(uids))!=len(uids):
        raise ValueError('main UID identity changed')
    w=bootstrap_weights(len(uids),replicates,seed)
    overall, headwise, per_uid = [], [], {}
    for p,panel in enumerate(PANELS):
        centered_fraction=divide(total[:,p].sum((1,2)),original[:,p].sum((1,2)))
        pooled=divide(retained[:,p].sum((1,2)),total[:,p].sum((1,2))[:,None])
        per_uid[panel]=pooled
        record=dict(panel=panel, unique_users=len(uids),
            teacher_centered_fraction_of_original=estimate(centered_fraction,w),
            rank_energy_retention={str(rank):estimate(pooled[:,k],w) for k,rank in enumerate(RANKS)},
            original_zero_energy_user_heads=int((original[:,p]==0).sum()),
            centered_zero_energy_user_heads=int((total[:,p]==0).sum()),
            centered_zero_energy_users=int((total[:,p].sum((1,2))==0).sum()))
        overall.append(record)
        for l in range(layers):
            for h in range(heads):
                row=dict(panel=panel,layer_zero_based=l,head_zero_based=h, unique_users=len(uids))
                row.update(flat('teacher_centered_fraction_of_original',estimate(divide(total[:,p,l,h],original[:,p,l,h]),w)))
                for k,rank in enumerate(RANKS):
                    row.update(flat(f'rank{rank}_energy_retention',estimate(divide(retained[:,p,l,h,k],total[:,p,l,h]),w)))
                headwise.append(row)
    paired=[dict(rank=rank,near_minus_broad=estimate(per_uid['near16'][:,k]-per_uid['broad16'][:,k],w))
            for k,rank in enumerate(RANKS)]
    permitted=not config['diagnostic'] and runner['roles']==dict(fit=100,validation=16,main=2000)
    summary=dict(format='candidate_residual_oracle_svd_v1', edge=config['edge'], main_users=len(uids),
        quality_use_permitted=permitted, runner_diagnostic=config['diagnostic'], ranks=list(RANKS),
        matrix='16 candidates by32 local-read-output dimensions, separately for each UID/layer/head',
        matrix_geometry=dict(candidates=16,read_output_dim=dim,layers=layers,heads=heads,centered_rank_ceiling=15),
        overall=overall, panel_paired_contrasts=paired, all_head_records=len(headwise),
        centering='targets already centered by collector; recentered in float64 only for float32 roundoff',
        maximum_abs_target_mean_before_recentering=maximum_recentering_mean,
        aggregation='within each UID, sum retained and total centered energy over all100 layer/heads, then take UID mean of ratios; detailed head ratios separately',
        confidence=dict(method='paired UID bootstrap',replicates=replicates,seed=seed,
            scope='both panels share identical UID multiplicities; pointwise conditional on fixed F/panels; not fitting-seed or candidate resampling'),
        seconds=time.perf_counter()-started,
        provenance=dict(input_root=str(root),configuration_sha256=sha(root/'configuration.json'),
            runner_summary_sha256=sha(root/'summary.json'),analysis_source_sha256=sha(Path(__file__)),
            block_loader_source_sha256=sha(Path(__file__).with_name('analyze_candidate_residual_probe.py'))),
        limitations=[
            'Post-hoc per-UID/layer/head oracle SVD; uses teacher residuals, not a fitted/deployable shared predictor.',
            'Q16 candidate centering itself imposes rank<=15. Low rank is conditional on this small panel and read-output width32.',
            'No historical-row axis: this is not E6 h(candidate,history-row) truncation and cannot identify per-row code rank.',
            'Energy retention is neither recovered recommendation quality nor downstream logit/AUC recovery.',
            'Near16 is selected using teacher scores from a64-item pool; this changes the scenario and is teacher-informed/optimistic.',
            'Both panels reuse the same UID/cache/F and can share candidate IDs; they are paired conditions, not independent replication.',
            'Zero-energy residuals have undefined retention and are excluded from ratio means, with counts retained.',
            'All100 heads retained; intervals are pointwise with no multiple-head correction.',
            'No new GPU collection, future labels, final Design intervention or paper edit.'])
    return summary,headwise


def write_outputs(summary, headwise, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    with (output/'headwise.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(headwise[0]));writer.writeheader();writer.writerows(headwise)
    rows=[]
    for point in summary['overall']:
        for rank,s in point['rank_energy_retention'].items():
            rows.append(dict(panel=point['panel'],rank=int(rank),**flat('energy_retention',s)))
    with (output/'rank_energy.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    def fmt(s):
        return 'undefined' if s['mean'] is None else f"{100*s['mean']:.3f}% [{100*s['ci95'][0]:.3f}, {100*s['ci95'][1]:.3f}]"
    lines=[f"# 候选read残差的事后几何：edge{summary['edge']}", '',
        f"固定{summary['main_users']}名main用户；{summary['confidence']['replicates']}次配对UID bootstrap。", '',
        '每个用户/层/头，对16候选×32维局部read输出残差分别做SVD；先在同UID内跨100头汇总能量取比，再作用户平均。', '',
        '| panel | 候选中心化能量/原始残差 | rank1 | rank2 | rank4 | rank8 | 零中心残差UID/head记录 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p in summary['overall']:
        lines.append('| '+' | '.join([p['panel'],fmt(p['teacher_centered_fraction_of_original']),
            *[fmt(p['rank_energy_retention'][str(r)]) for r in RANKS],str(p['centered_zero_energy_user_heads'])])+' |')
    lines += ['', '整体统计对每个UID先按层头池化，全部层头统计见headwise.csv。两panel使用同一组UID抽样权重，不能当成独立重复。', '',
        '**解释边界：** 这是事后oracle分解，Q16去候选均值本身使秩不超过15；32维是局部read输出，没有历史行轴。'
        '因此不能把这些曲线冒称E6交互h截断、质量恢复率或所需compact-code秩。near16用teacher分数选取，本身有teacher-informed场景选择的乐观性。', '',
        *['- '+x for x in summary['limitations']]]
    if not summary['quality_use_permitted']:lines.insert(2,'**仅结构/数值诊断，不能作为人口机制结论。**')
    (output/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--replicates',type=int,default=500);p.add_argument('--seed',type=int,default=17)
    a=p.parse_args();s,h=analyze(a.input,replicates=a.replicates,seed=a.seed);write_outputs(s,h,a.output)
    print(json.dumps(dict(status='complete',main_users=s['main_users'],quality_use_permitted=s['quality_use_permitted'],seconds=s['seconds'])))
