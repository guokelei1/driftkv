#!/usr/bin/env python3
"""Fixed-history candidate logit residuals for native K/V-source controls.

This panel has no real labels.  Its ordering disagreement is teacher
agreement, never true-label AUC.  The full64 and fixed broad16 panels are
primary; the teacher-selected near16 is a conditional diagnostic.
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
import json
from pathlib import Path
import sys

from experiments.ch3_motivation.code.analyze_candidate_logit_angles import candidate_statistics, sha
import numpy as np

METHODS=('oldKoldV','teacherKoldV','oldKteacherV','teacherKteacherV')
TEACHER='teacherKteacherV'


def panel_indices(teacher):
    """PerUID near16 uses this run's teacher64; stable tie order."""
    users,width=teacher.shape
    if width!=64:raise ValueError('full64 native-cross panel required')
    order=np.argsort(teacher,axis=1,kind='stable')
    ranked=np.take_along_axis(teacher,order,axis=1)
    start=np.argmin(ranked[:,15:]-ranked[:,:-15],axis=1)
    near=np.take_along_axis(order,start[:,None]+np.arange(16)[None],axis=1)
    return dict(full64=np.broadcast_to(np.arange(64),(users,64)),
        broad16=np.broadcast_to(np.r_[0:8,56:64],(users,16)),near16=near)


def summarize_arrays(uids,scores,methods):
    if tuple(methods)!=METHODS or scores.shape!=(len(uids),4,64):
        raise ValueError('four native K/V-source controls in the fixed method order required')
    if len(uids)!=len(set(map(int,uids))) or len(uids)<512:
        raise ValueError('main mechanism panels require at least512 distinct UID')
    teacher=scores[:,methods.index(TEACHER)]
    panels=panel_indices(teacher)
    statistics={};per_uid=dict(uid=uids);rows=[]
    for panel,indices in panels.items():
        target=np.take_along_axis(teacher,indices,axis=1)
        statistics[panel]={}
        for index,method in enumerate(methods):
            actual=np.take_along_axis(scores[:,index],indices,axis=1)
            point,decomposition=candidate_statistics(target,actual)
            statistics[panel][method]=point
            per_uid.update({panel+'_'+method+'_'+key:value for key,value in decomposition.items()})
            row=dict(panel=panel,method=method,users=len(uids),
                primary_panel=panel in ['full64','broad16'])
            for key in ['total_residual_rms','per_uid_mean_rms','centered_candidate_rms','mean_energy_fraction','centered_energy_fraction']:
                value=point[key];row[key]=value['value']
                row[key+'_ci_low'],row[key+'_ci_high']=value['ci95'] or [None,None]
            for name in ['all_pairs','adjacent_teacher_pairs']:
                value=point['ranking'][name]['disagreement']
                row[name+'_strict_reversal_rate']=value['value']
                row[name+'_ci_low'],row[name+'_ci_high']=value['ci95'] or [None,None]
            rows.append(row)
    return statistics,per_uid,rows


def source_metadata(folder):
    """Read only the old candidate IDs, gaps and lengths; no old scores."""
    config=json.loads((folder/'configuration.json').read_text())
    summary=json.loads((folder/'summary.json').read_text())
    metadata={}
    for artifact in summary['artifacts']:
        path=folder/artifact['path']
        if path.parent.name!='main':continue
        if sha(path)!=artifact['sha256']:raise ValueError('original candidate input changed')
        with np.load(path,allow_pickle=False) as pack:
            for i,uid in enumerate(pack['uid']):
                metadata[int(uid)]=(pack['pool_ids'][i].copy(),int(pack['query_delta'][i]),int(pack['history_length'][i]))
    if set(metadata)!=set(config['roles']['main']):raise ValueError('original main candidate input incomplete')
    return config,metadata


def run(folder,candidate_source,output):
    folder,candidate_source,output=Path(folder),Path(candidate_source),Path(output)
    config=json.loads((folder/'configuration.json').read_text())
    summary=json.loads((folder/'summary.json').read_text())
    if summary['status']!='complete' or config.get('diagnostic',False):
        raise ValueError('requires complete non-diagnostic native-cross collection')
    previous,original=source_metadata(candidate_source)
    if config['edge']!=previous['edge']:raise ValueError('candidate source edge differs')
    methods=list(config.get('candidate_method_names',METHODS))
    uids=[];scores=[];artifacts=[]
    for artifact in summary['candidate_artifacts']:
        path=folder/artifact['path']
        if sha(path)!=artifact['sha256']:raise ValueError('native candidate block changed')
        with np.load(path,allow_pickle=False) as pack:
            for i,uid in enumerate(pack['uid']):
                uid=int(uid)
                if uid not in original:raise ValueError('candidate UID outside original main role')
                pool,gap,length=original[uid]
                if (not np.array_equal(pack['pool_ids'][i],pool)
                        or int(pack['query_delta'][i])!=gap or int(pack['history_length'][i])!=length):
                    raise ValueError('native panel changed the fixed pool/history/time gap')
            uids.extend(map(int,pack['uid']));scores.append(pack['scores'])
        artifacts.append(dict(path=str(path.resolve()),sha256=artifact['sha256']))
    expected=config.get('candidate_users')
    if expected is not None and uids!=expected:raise ValueError('native candidate users differ from fixed plan')
    scores=np.concatenate(scores)
    points,per_uid,rows=summarize_arrays(np.asarray(uids,dtype=np.uint64),scores,methods)
    result=dict(format='native_cross_candidate_logit_angles_v1',edge=config['edge'],users=len(uids),
        residual='teacherKteacherV logit minus each current-reader native K/V-source cross logit; oldKoldV is Reuse at cutover',
        scope='Cutover-only native-source counterfactual, fixed pool/history/time/gap; original main users with observed feedback, not all4000 residents.',
        source=str(folder.resolve()),configuration_sha256=sha(folder/'configuration.json'),
        summary_sha256=sha(folder/'summary.json'),analysis_sha256=sha(__file__),
        candidate_source=str(candidate_source.resolve()),candidate_source_configuration_sha256=sha(candidate_source/'configuration.json'),
        candidate_artifacts=artifacts,methods=methods,panels=points,
        same_original_pool_gap_history_verified=True,real_labels_used=False,quality_auc_computed=False,
        primary_panels=['full64','broad16'],conditional_diagnostic_panels=['near16'],
        limitations=['Candidate panels have no true labels; ordering disagreement is not actual AUC or accuracy.',
            'near16 is selected by the smallest teacher64 score span, so results condition on teacher-near scores.',
            'Mean/centered split describes residual geometry; perUID teacher-dependent mean is not an available deployable calibrator.',
            'Four K/V-source states are transient read controls, not four persistent write trajectories.',
            'Teacher−Reuse here and prior teacher−F16 are separately collected comparisons; do not substitute their names or mix their panels.',
            '500 UID bootstrap draws condition on frozen backbones; this is exploratory, not independent confirmation.'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    np.savez_compressed(output/'per_uid_decomposition.npz',**per_uid)
    with (output/'candidate_residuals.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    def fmt(value,percent=False):
        return 'NA' if value is None else f'{value*(100 if percent else 1):.6f}'
    lines=[f"# Native K/V-source candidate residual: edge{config['edge']}",'',result['scope'],'',
        'Every row uses current-reader logits for the same history, time gap and candidate pool. No candidate true labels or AUC. full64/broad16 are primary; near16 is teacher-conditioned.','',
        '| Panel | Method | UIDs | Total RMS | PerUID mean RMS | Centered RMS | Centered energy% | Pair reversal% | Teacher-neighbor reversal% |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| '+f"{row['panel']} | {row['method']} | {row['users']} | "+' | '.join(
            fmt(row[key],key.endswith('fraction') or key.endswith('rate')) for key in [
                'total_residual_rms','per_uid_mean_rms','centered_candidate_rms','centered_energy_fraction',
                'all_pairs_strict_reversal_rate','adjacent_teacher_pairs_strict_reversal_rate'])+' |')
    lines+=['','- '+'\n- '.join(result['limitations'])]
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(edge=config['edge'],users=len(uids),output=str(output))),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--candidate-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.input,args.candidate_source,args.output)
