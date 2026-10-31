#!/usr/bin/env python3
"""Candidate-varying teacher logit residuals; no feedback labels or AUC.

The existing candidate probe saves F and teacher scores for two16-candidate
panels, not Reuse or full64 scores.  Keep that distinction explicit.
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def estimate(point, samples):
    valid=np.isfinite(samples)
    return dict(value=float(point) if np.isfinite(point) else None,
        ci95=np.percentile(np.asarray(samples)[valid],[2.5,97.5]).tolist() if valid.any() else None)


def candidate_statistics(teacher, method, *, bootstrap=500, seed=17):
    teacher,method=np.asarray(teacher,dtype=float),np.asarray(method,dtype=float)
    if teacher.shape!=method.shape or teacher.ndim!=2 or not np.isfinite(teacher).all() or not np.isfinite(method).all():
        raise ValueError('requires finite paired[UID,candidate] scores')
    users,candidates=teacher.shape
    if candidates<2 or len(teacher)<1:
        raise ValueError('needs users and multiple candidates')
    delta=teacher-method
    mean=delta.mean(1)
    centered=delta-mean[:,None]
    per_uid=dict(total_energy=np.mean(delta**2,axis=1),mean_energy=mean**2,
        centered_energy=np.mean(centered**2,axis=1))
    np.testing.assert_allclose(per_uid['total_energy'],per_uid['mean_energy']+per_uid['centered_energy'],rtol=1e-12,atol=1e-15)
    rng=np.random.default_rng(seed)
    draws=rng.integers(users,size=(bootstrap,users))
    energy={key:(value.mean(),value[draws].mean(1)) for key,value in per_uid.items()}
    result=dict(users=users,candidates_per_user=candidates,
        mechanism_support_permitted=users>=512,
        teacher_minus_method_mean=float(delta.mean()),
        total_residual_rms=estimate(np.sqrt(energy['total_energy'][0]),np.sqrt(energy['total_energy'][1])),
        per_uid_mean_rms=estimate(np.sqrt(energy['mean_energy'][0]),np.sqrt(energy['mean_energy'][1])),
        centered_candidate_rms=estimate(np.sqrt(energy['centered_energy'][0]),np.sqrt(energy['centered_energy'][1])))
    with np.errstate(divide='ignore',invalid='ignore'):
        for key in ['mean_energy','centered_energy']:
            point=energy[key][0]/energy['total_energy'][0] if energy['total_energy'][0] else np.nan
            result[key+'_fraction']=estimate(point,energy[key][1]/energy['total_energy'][1])
    left,right=np.triu_indices(candidates,1)
    order=np.argsort(teacher,axis=1,kind='stable')
    sorted_teacher=np.take_along_axis(teacher,order,axis=1)
    sorted_method=np.take_along_axis(method,order,axis=1)
    comparisons=dict(all_pairs=(teacher[:,right]-teacher[:,left],method[:,right]-method[:,left]),
        adjacent_teacher_pairs=(np.diff(sorted_teacher,axis=1),np.diff(sorted_method,axis=1)))
    result['ranking']={}
    for name,(target_diff,actual_diff) in comparisons.items():
        comparable=target_diff!=0.
        flip=(target_diff*actual_diff<0.)&comparable
        tied=(actual_diff==0.)&comparable
        denominators=comparable.sum(1)
        numerators=flip.sum(1)
        method_ties=tied.sum(1)
        with np.errstate(divide='ignore',invalid='ignore'):
            rate=numerators.sum()/denominators.sum() if denominators.sum() else np.nan
            samples=numerators[draws].sum(1)/denominators[draws].sum(1)
        result['ranking'][name]=dict(teacher_nontied_pairs=int(denominators.sum()),
            strict_reversals=int(numerators.sum()),method_ties_on_teacher_nonties=int(method_ties.sum()),
            users_with_teacher_nontied_pairs=int((denominators>0).sum()),
            mechanism_support_permitted=int((denominators>0).sum())>=512,
            disagreement=estimate(rate,samples))
    return result,dict(per_uid_mean=mean,**per_uid)


def analyze_existing(folder, output):
    folder,output=Path(folder),Path(output)
    config=json.loads((folder/'configuration.json').read_text())
    summary=json.loads((folder/'summary.json').read_text())
    if summary['status']!='complete' or config['diagnostic']:
        raise ValueError('needs complete existing main2000 candidate probe')
    users=[];teacher=[];fitted=[];gaps=[];lengths=[];artifacts=[]
    for artifact in summary['artifacts']:
        path=folder/artifact['path']
        if path.parent.name!='main':continue
        if sha(path)!=artifact['sha256']:
            raise ValueError('candidate artifact changed')
        with np.load(path,allow_pickle=False) as pack:
            if 'reuse_scores' in pack.files:
                raise ValueError('this explicitly F-only loader needs review for the new schema')
            users.extend(map(int,pack['uid']))
            teacher.append(pack['teacher_scores']);fitted.append(pack['f_scores'])
            gaps.extend(map(int,pack['query_delta']));lengths.extend(map(int,pack['history_length']))
            if pack['pool_ids'].shape[1]!=64 or pack['panel_indices'].shape[1:]!=(2,16):
                raise ValueError('not the frozen broad16/near16 protocol')
        artifacts.append(dict(path=str(path),sha256=artifact['sha256']))
    if users!=config['roles']['main'] or len(users)!=len(set(users)) or len(users)<512:
        raise ValueError('not the fixed role-separated main panel of at least512 unique UID')
    teacher,fitted=np.concatenate(teacher),np.concatenate(fitted)
    if teacher.shape!=(len(users),2,16) or fitted.shape!=teacher.shape:
        raise ValueError('candidate scores differ from recorded panel geometry')
    result=dict(format='candidate_logit_angles_existing_v1',edge=config['edge'],users=len(users),
        residual='native cutover teacher logit minus mapped-F logit; NOT teacher minus Reuse',
        source=str(folder.resolve()),configuration_sha256=sha(folder/'configuration.json'),
        summary_sha256=sha(folder/'summary.json'),analysis_sha256=sha(__file__),
        artifacts=artifacts,models=config['models'],f_checkpoint_sha256=config['checkpoint_sha256'],
        panels={},protocol=config['panel_rules'],bootstrap='500 UID-cluster bootstrap draws, seed17; each UID shares fixed history/release time/gap across candidates',
        reuse_scores_available=False,full64_scores_available=False,real_labels_available=False,
        quality_auc_computed=False,
        decomposition='d_ui=teacher_ui-F_ui; mu_u=mean_i d_ui; centered_ui=d_ui-mu_u. Mean_u mean_i d_ui² = mean_u mu_u² + mean_u mean_i centered_ui².',
        limitations=['Teacher agreement is not true-label prediction accuracy or AUC.',
            'near16 is selected using the teacher-smallest score span within a64-ID pool; it is a conditional oracle diagnostic.',
            'pool_ids stores64 candidates, but saved scores cover two16-candidate panels only; panels may overlap.',
            'No Reuse scores are saved, so candidate dependence of uncorrected Reuse requires new scoring.',
            'Candidate-varying scalar residual does not prove a particular gate or compact code is necessary.',
            'Intervals condition on frozen foundation and F weights; no independent confirmation or training-seed uncertainty.'])
    saved=dict(uid=np.asarray(users,dtype=np.uint64),query_delta=np.asarray(gaps,dtype=np.int64),history_length=np.asarray(lengths,dtype=np.int32))
    rows=[]
    for index,panel in enumerate(config['panels']):
        stats,per_uid=candidate_statistics(teacher[:,index],fitted[:,index])
        result['panels'][panel]=stats
        saved.update({panel+'_'+key:value for key,value in per_uid.items()})
        row=dict(panel=panel,users=len(users))
        for key in ['total_residual_rms','per_uid_mean_rms','centered_candidate_rms','mean_energy_fraction','centered_energy_fraction']:
            row[key]=stats[key]['value']
        row['all_pair_disagreement']=stats['ranking']['all_pairs']['disagreement']['value']
        row['adjacent_teacher_pair_disagreement']=stats['ranking']['adjacent_teacher_pairs']['disagreement']['value']
        rows.append(row)
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    np.savez_compressed(output/'per_uid_decomposition.npz',**saved)
    with (output/'candidate_angles.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=[f"# edge{config['edge']}: fixed-history candidate logit residual",'',
        'Existing main2000, same cutover history/time/gap. Residual is teacher minus F; no Reuse scores, real labels or AUC.','',
        '| Panel | UIDs | Total RMS | Per-UID mean RMS | Centered RMS | Centered energy | Pair reversals | Adjacent-teacher reversals |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['panel']} | {row['users']} | {row['total_residual_rms']:.7f} | {row['per_uid_mean_rms']:.7f} | {row['centered_candidate_rms']:.7f} | {100*row['centered_energy_fraction']:.4f}% | {100*row['all_pair_disagreement']:.4f}% | {100*row['adjacent_teacher_pair_disagreement']:.4f}% |")
    lines+=['','The mean/centered split is an algebraic description, not a fitted correction. A constant score offset leaves within-UID ranking unchanged. The centered fraction measures the share of squared logit residual beyond this offset.','']
    lines+=['- '+item for item in result['limitations']]
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(edge=config['edge'],users=len(users),output=str(output),panels=rows)),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    analyze_existing(args.input,args.output)
