#!/usr/bin/env python3
"""Matched teacher-variance scale check for existing fixed-Q16 F errors."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.ch3_motivation.code.analyze_residual_angles import load, weights, mean_stat, stat


def components(snapshot):
    t=np.asarray(snapshot['fixed_panel_scores']['rebuild'], dtype=float)
    f=np.asarray(snapshot['fixed_panel_scores']['f'], dtype=float)
    e=f-t
    return np.mean((e-e.mean())**2), np.mean((t-t.mean())**2)


def compare(users, checkpoint, *, replicates=500, seed=17):
    arrays=[]
    for u in users:
        points={s['checkpoint']:s for s in u['snapshots']}
        if checkpoint in points:
            arrays.append([*components(points[checkpoint]),*components(points['cutover'])])
    x=np.asarray(arrays);n=len(x);w=weights(n,replicates,seed)
    current_error,current_variance,initial_error,initial_variance=x.T
    sums=w@x
    with np.errstate(invalid='ignore',divide='ignore'):
        current_ratio=current_error.mean()/current_variance.mean()
        initial_ratio=initial_error.mean()/initial_variance.mean()
        current_draw=sums[:,0]/sums[:,1];initial_draw=sums[:,2]/sums[:,3]
    return dict(checkpoint=checkpoint,matched_users=n,quality_use_permitted=n>=512,
        current=dict(centered_f_error_mse=mean_stat(current_error,w),teacher_candidate_variance=mean_stat(current_variance,w),
            ratio_of_means=stat(current_ratio,current_draw,n),zero_teacher_variance_users=int((current_variance==0).sum())),
        same_users_cutover=dict(centered_f_error_mse=mean_stat(initial_error,w),teacher_candidate_variance=mean_stat(initial_variance,w),
            ratio_of_means=stat(initial_ratio,initial_draw,n),zero_teacher_variance_users=int((initial_variance==0).sum())),
        paired_change=dict(centered_f_error_mse=mean_stat(current_error-initial_error,w),
            teacher_candidate_variance=mean_stat(current_variance-initial_variance,w),
            normalized_ratio=stat(current_ratio-initial_ratio,current_draw-initial_draw,n)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,default=Path('experiments/ch3_motivation/runs/insight_mechanisms/attempt_1'))
    p.add_argument('--replicates',type=int,default=500);p.add_argument('--seed',type=int,default=17)
    a=p.parse_args();out=a.run_root/'residual_angles';out.mkdir(exist_ok=True)
    cases,provenance={},{}
    for edge in [2,3]:
        users,binding=load(a.run_root/f'main_edge{edge}')
        cases[str(edge)]=[compare(users,c,replicates=a.replicates,seed=a.seed) for c in ('cutover','append_8','append_32','append_128')]
        provenance[str(edge)]=binding
    result=dict(format='teacher_candidate_variance_scale_check_v1',cases=cases,
        definition='mean_UID[mean_candidate(centered(F-teacher)^2)] / mean_UID[var_candidate(teacher logits)]',
        pairing='each available checkpoint cohort compared with exactly those UID own cutover; same UID multiplicities used across every numerator/denominator and phase',
        confidence=dict(method='paired UID bootstrap',replicates=a.replicates,seed=a.seed),
        provenance=provenance,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Ratio of means, not mean of per-user ratios; individual zero-variance teachers remain in the aggregate with counts.',
          'This controls one overall teacher-score amplitude reference, not all time/history/candidate confounds.',
          'Absolute centered error growth need not imply worse relative teacher discrimination or task quality.',
          'Q16 teacher agreement only; no labels or AUC. Existing teacher-pair discord changes remain non-significant at append128.',
          'Conditional on existing F weights and selected cases; no new GPU computation or independent confirmation.'])
    (out/'teacher_scale_check.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    def fmt(s):return f"{s['mean']:.6g} [{s['ci95'][0]:.6g}, {s['ci95'][1]:.6g}]"
    lines=['# 固定Q16误差的teacher方差尺度核对','',
        '每点只用该点可用UID，并与相同UID自身cutover比较。归一化定义为用户平均centered F logit MSE / 用户平均teacher候选方差；不是逐UID小分母比率的平均。', '',
        '| 边 | checkpoint | matched UID | teacher方差：cutover→当前 | normalized error：cutover→当前 | normalized paired变化[UID95%] |',
        '|---|---|---:|---:|---:|---:|']
    for edge,points in cases.items():
        for c in points:
            old,new=c['same_users_cutover'],c['current']
            lines.append(f"| {edge} | {c['checkpoint']} | {c['matched_users']} | "
                f"{old['teacher_candidate_variance']['mean']:.6g}→{new['teacher_candidate_variance']['mean']:.6g} | "
                f"{old['ratio_of_means']['mean']:.6g}→{new['ratio_of_means']['mean']:.6g} | {fmt(c['paired_change']['normalized_ratio'])} |")
    lines+=['','teacher方差本身的均值、配对变化及500次UID区间均在JSON内。'
        '归一化后的变化才可用于讨论相对teacher信号尺度上的残差，不能仅凭absolute centered MSE增长宣称判别恶化。'
        '即使归一化改善，也不等于真实AUC改善；此前Q16成对排序不一致率在append128没有稳定上升。','',
        *['- '+x for x in result['limitations']]]
    (out/'teacher_scale_check.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='complete',output=str(out/'teacher_scale_check.json'))))


if __name__=='__main__':main()
