#!/usr/bin/env python3
"""CPU-only descriptions of heterogeneous residuals and relative candidate scores.

This reuses frozen Q16 scores, not future labels.  Candidate-centered logit
error removes a per-user common shift; it is neither AUC nor gate necessity.
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
import numpy as np

METHODS = ('reuse', 'f', 'teacher_forced')
CHECKPOINTS = ('cutover', 'append_8', 'append_32', 'append_128')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights(n, replicates, seed):
    rng = np.random.default_rng(seed)
    return np.asarray([np.bincount(rng.integers(n, size=n), minlength=n)
                       for _ in range(replicates)], dtype=np.uint16)


def stat(point, draws, valid_users):
    draws = np.asarray(draws)
    finite = draws[np.isfinite(draws)]
    return dict(mean=float(point) if np.isfinite(point) else None,
                ci95=np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
                valid_users=int(valid_users), quality_use_permitted=bool(valid_users >= 512),
                invalid_bootstrap_draws=int(len(draws) - len(finite)))


def mean_stat(values, w):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    with np.errstate(invalid='ignore', divide='ignore'):
        draws = (w @ np.nan_to_num(values, nan=0.)) / (w @ valid.astype(float))
    return stat(np.nanmean(values), draws, valid.sum())


def ratio_stat(num, den, w):
    with np.errstate(invalid='ignore', divide='ignore'):
        draws = (w @ num) / (w @ den)
        point = np.mean(num) / np.mean(den)
    return stat(point, draws, len(num))


def flat(prefix, s):
    lo, hi = s['ci95'] or (None, None)
    return {prefix: s['mean'], prefix+'_ci95_low': lo, prefix+'_ci95_high': hi,
            prefix+'_valid_users': s['valid_users']}


def score_metrics(snapshot, method):
    values = np.asarray(snapshot['fixed_panel_scores'][method], dtype=float)
    teacher = np.asarray(snapshot['fixed_panel_scores']['rebuild'], dtype=float)
    error = values - teacher
    centered = error - error.mean()
    total = np.mean(error ** 2)
    offset = error.mean() ** 2
    center = np.mean(centered ** 2)
    if not np.isclose(total, offset + center, rtol=1e-10, atol=1e-16):
        raise ValueError('common-offset and candidate-centered decomposition failed')
    i, j = np.triu_indices(len(values), 1)
    td, sd = teacher[i] - teacher[j], values[i] - values[j]
    valid = td != 0
    disagreement = np.mean(np.sign(td[valid]) != np.sign(sd[valid])) if valid.any() else np.nan
    return dict(total_mse=float(total), common_offset_mse=float(offset),
                candidate_centered_mse=float(center), teacher_pair_disagreement=float(disagreement))


def load(root):
    config = json.loads((root/'configuration.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    if summary['status'] != 'complete' or config['diagnostic'] or summary['users'] != 2000:
        raise ValueError('this analysis requires the complete original 2000-person non-diagnostic case')
    identity = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    users, artifacts = [], []
    for artifact in summary['artifacts']:
        path = root/artifact['path']
        if sha(path) != artifact['sha256']:
            raise ValueError('committed result hash changed')
        block = json.loads(path.read_text())
        if (block['identity_sha256'] != identity or block['status'] != 'complete'
                or block['uids'] != config['users'][len(users):len(users)+len(block['users'])]
                or [u['uid'] for u in block['users']] != block['uids']):
            raise ValueError('committed member/config identity changed')
        users.extend(block['users'])
        artifacts.append(dict(path=str(path), sha256=artifact['sha256']))
    if [u['uid'] for u in users] != config['users']:
        raise ValueError('population is not complete')
    return users, dict(configuration_sha256=sha(root/'configuration.json'), artifacts=artifacts)


def analyze_edge(users, *, replicates, seed):
    snapshots = [{s['checkpoint']: s for s in u['snapshots']} for u in users]
    cut = [s['cutover']['candidate_residual'] for s in snapshots]
    raw = np.asarray([c['raw_mse_by_layer_head'] for c in cut], dtype=float)
    ratios = np.asarray([c['centered_energy_ratio_by_layer_head'] for c in cut], dtype=float)
    centered = raw * np.nan_to_num(ratios, nan=0.)
    if np.any(raw[~np.isfinite(ratios)] != 0):
        raise ValueError('undefined variation ratio only allowed at zero residual energy')
    n, layers, heads = raw.shape
    w = weights(n, replicates, seed)
    r = raw.reshape(n, -1); c = centered.reshape(n, -1); q = ratios.reshape(n, -1)
    rb, cb = w @ r, w @ c
    raw_total, centered_total = r.sum(), c.sum()
    records = []
    for l in range(layers):
        for h in range(heads):
            k = l*heads+h
            row = dict(layer_zero_based=l, head_zero_based=h, unique_users=n,
                       valid_ratio_users=int(np.isfinite(q[:, k]).sum()),
                       users_above_10pct=int((q[:, k] > .1).sum()),
                       mean_raw_read_mse=float(r[:, k].mean()))
            row.update(flat('mean_uid_centered_fraction', mean_stat(q[:, k], w)))
            row.update(flat('pooled_centered_fraction', ratio_stat(c[:, k], r[:, k], w)))
            row.update(flat('raw_energy_share', stat(r[:, k].sum()/raw_total, rb[:, k]/rb.sum(1), n)))
            row.update(flat('candidate_energy_share', stat(c[:, k].sum()/centered_total, cb[:, k]/cb.sum(1), n)))
            records.append(row)
    global_metrics = dict(
        mean_uid_pooled_centered_fraction=mean_stat(c.sum(1)/r.sum(1), w),
        all_uid_head_energy_pooled_centered_fraction=ratio_stat(c.sum(1), r.sum(1), w),
        mean_uid_equal_head_centered_fraction=mean_stat(np.nanmean(q, axis=1), w))
    temporal = []
    for checkpoint in CHECKPOINTS:
        selected = [s for s in snapshots if checkpoint in s]
        m = len(selected); wt = weights(m, replicates, seed)
        current = {method: [score_metrics(s[checkpoint], method) for s in selected] for method in METHODS}
        initial = {method: [score_metrics(s['cutover'], method) for s in selected] for method in METHODS}
        for method in METHODS:
            row = dict(checkpoint=checkpoint, method=method, unique_users=m,
                       quality_use_permitted=m>=512)
            for metric in current[method][0]:
                a = np.asarray([s[metric] for s in current[method]])
                b = np.asarray([s[metric] for s in initial[method]])
                row.update(flat(metric, mean_stat(a, wt)))
                row.update(flat(metric+'_own_cutover', mean_stat(b, wt)))
                row.update(flat(metric+'_paired_change', mean_stat(a-b, wt)))
            if method == 'f':
                for metric in ('total_mse', 'candidate_centered_mse'):
                    a = np.asarray([s[metric] for s in current['f']])
                    b = np.asarray([s[metric] for s in current['teacher_forced']])
                    row.update(flat(metric+'_teacher_new_rows_improvement', mean_stat(a-b, wt)))
            temporal.append(row)
    return dict(unique_users=n, head_residuals=records, global_metrics=global_metrics,
                temporal_candidate_scores=temporal)


def write_csv(path, rows):
    names = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=names); writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, default=Path('experiments/ch3_motivation/runs/insight_mechanisms/attempt_1'))
    p.add_argument('--output', type=Path)
    p.add_argument('--replicates', type=int, default=500)
    p.add_argument('--seed', type=int, default=17)
    args = p.parse_args(); out = args.output or args.run_root/'residual_angles'; out.mkdir(parents=True, exist_ok=True)
    results, provenance = {}, {}
    for edge in [2, 3]:
        users, binding = load(args.run_root/f'main_edge{edge}')
        results[str(edge)] = analyze_edge(users, replicates=args.replicates, seed=args.seed)
        provenance[str(edge)] = binding
    summary = dict(format='residual_angles_v1', exploratory=True, edges=results,
                   confidence=dict(method='paired UID bootstrap; conditional on frozen single-seed F',
                                   replicates=args.replicates, seed=args.seed),
                   provenance=provenance, analysis_source_sha256=sha(Path(__file__)),
                   limitations=['Q16 panels and single fitted seed; no feedback labels or AUC.',
                    'Head inspection is exploratory; all100 heads retained, not independent confirmation.',
                    'Candidate variation of the mapped-path same-query read residual cannot prove item gates needed.',
                    'Across checkpoints query_delta changes with actual recent behavior; not a fixed-time causal isolation.',
                    'Centered logit error concerns relative scores, not necessarily ordering or observed feedback quality.',
                    'Teacher-new-row replacement is an oracle trajectory diagnostic, not deployable Design.'])
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    write_csv(out/'head_residuals.csv', [dict(edge=e, **row) for e,a in results.items() for row in a['head_residuals']])
    write_csv(out/'temporal_candidate_scores.csv', [dict(edge=e, **row) for e,a in results.items() for row in a['temporal_candidate_scores']])
    lines = ['# 简单F残差的两个补充观察', '',
      '使用原有两边各2,000人冻结结果；不启动GPU、不改变用户或权重。所有层头和检查点保留。', '',
      '## 1. 总体低候选能量掩盖层头异质性', '',
      '| 边 | 全层头能量先汇总、再取用户均值 | 层头等权后用户均值 | L0/H1用户平均比例[95%] | L0/H1占总候选变化能量 |',
      '|---|---:|---:|---:|---:|']
    for e,a in results.items():
        g=a['global_metrics'];h=a['head_residuals'][1]
        lines.append(f"| {e} | {g['mean_uid_pooled_centered_fraction']['mean']*100:.3f}% | "
                     f"{g['mean_uid_equal_head_centered_fraction']['mean']*100:.3f}% | "
                     f"{h['mean_uid_centered_fraction']*100:.2f}% "
                     f"[{h['mean_uid_centered_fraction_ci95_low']*100:.2f}, {h['mean_uid_centered_fraction_ci95_high']*100:.2f}] | "
                     f"{h['candidate_energy_share']*100:.2f}% |")
    lines += ['', 'L/H均从零编号。L0/H1是在全100头检查后发现的探索性案例，不能称独立确认。'
      '全局比例低主要描述总体残差能量构成，不代表每个头都对候选不敏感；同样，少数头有变化不证明增加门控必有收益。'
      '可用于提出“不同头的剩余误差形态不同，校准和校正输入应关注局部差异”的研究问题。', '',
      '## 2. 总MSE下降可同时伴随候选相对分数误差增加', '',
      '令e_c为F与teacher的logit差，严格分解为mean(e)^2 + mean((e−mean(e))^2)。'
      '第二项消除了每用户共同偏置，反映固定候选之间的相对分数差，不使用标签。'
      '它乘以32/15等于Q16所有无序候选对的相对logit差MSE；并非AUC。', '',
      '| 边 | checkpoint | 同UID N | F centered MSE: cutover→当前 | 配对变化[95%] | 教师新行额外减少centered MSE[95%] |',
      '|---|---|---:|---:|---:|---:|']
    for e,a in results.items():
        for row in a['temporal_candidate_scores']:
            if row['method']!='f':continue
            k='candidate_centered_mse';d=k+'_paired_change';f=k+'_teacher_new_rows_improvement'
            lines.append(f"| {e} | {row['checkpoint']} | {row['unique_users']} | "
                         f"{row[k+'_own_cutover']:.4g}→{row[k]:.4g} | "
                         f"{row[d]:.4g} [{row[d+'_ci95_low']:.4g}, {row[d+'_ci95_high']:.4g}] | "
                         f"{row[f]:.4g} [{row[f+'_ci95_low']:.4g}, {row[f+'_ci95_high']:.4g}] |")
    lines += ['', '所有追加点均与该点可用用户自己的cutover配对；无人口构成混淆。'
      '但候选ID固定不等于query上下文固定：query_delta随实际行为改变。该发现不能独立归因历史长度或写入传播，宜再做固定delta交叉。', '',
      '重要反向检查：Q16内teacher成对排序不一致率并未在两边稳定上升；append128相对cutover变化edge2约−0.058个百分点、edge3约+0.046个百分点，均区间跨零（完整各点在CSV）。'
      '因此可说“全局MSE掩盖了相对分数残差”，不能说推荐排序或真实AUC必然恶化。', '',
      *['- '+s for s in summary['limitations']]]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='complete', output=str(out), edges=[2,3], users_each=2000)))


if __name__ == '__main__':
    main()
