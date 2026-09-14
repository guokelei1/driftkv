"""Metadata-only reconstruction of the frozen detector's preparation dependencies."""

import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from design.data import ROOT, DAY, histories
from design.report_native_flops import V, prefix_literal, append_ops, query_ops, value, read_cost, prepare_cost
from design.run import write_json
from design2.run_lifecycle import sha, summary_build_cost
from hstu_kvcache.design2.conditional import arithmetic
from insight_two.common import CUTOVER_DAYS

OUT=ROOT/'results/design2/lifecycle_01/preparation'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((ROOT/'configs/design2/detection_01.json').read_text())
    uids=protocol['groups']['residual_calibration']; life=set(protocol['lifetime_uids']['residual_calibration'])
    history=histories(uids,288)
    cal=pd.read_parquet(ROOT/'results/design2/analysis/detection_01/states.parquet',filters=[('role','==','residual_calibration')])
    assert set(cal.uid)==set(uids) and len(uids)==512
    counts=Counter(); costs=Counter(); specials=Counter(); built={}; source_summaries=set()
    def end(u,stamp):return int(np.searchsorted(history.rows[u][0],stamp,side='left'))
    def charge(label,ops):
        costs[label]+=value(ops)
        for k,v in ops.items():
            if k not in ('gemm','scalar'):specials[k]+=v
    def build(u,t,producer,label,summary):
        key=(u,t,producer); n=min(1024,end(u,t)); assert n>0
        if key not in built:
            built[key]=n; counts[f'{label}_prefix_n{n}']+=1
            charge(label+'_prefix',prefix_literal(n))
        if summary and key not in source_summaries:
            source_summaries.add(key);costs['source_summary_initial']+=summary_build_cost(n)
        return n
    def replay(n,events,label):
        for i in range(0,events,128):
            m=min(128,events-i); counts[f'{label}_append_n{n}_m{m}']+=1
            charge(label+'_append',append_ops(n,m))
            evicted=max(0,n+m-1024)
            costs['source_summary_maintenance']+=V*(m+evicted)
            n=min(1024,n+m)
        return n
    lengths,early={},{}; start=CUTOVER_DAYS[0]*DAY
    for u in uids:
        n=build(u,start,0,'source',True); lengths[u]=n
        ts=history.rows[u][0]; k=end(u,start)
        snap=int(ts[k-4]) if n>4 and ts[k-4]>ts[k-n] else start
        old=build(u,snap,0,'source',True); early[u]=(old,k-end(u,snap),snap)
    validations=[]
    for target in range(1,6):
        cut=CUTOVER_DAYS[target-1]*DAY
        if target!=2:
            sizes=[]
            for u in uids:
                old,e,snap=early[u]
                sizes.extend([lengths[u],replay(old,e,'early_source')])
                if target>1:sizes.append(build(u,cut,target-1,'adjacent_source',True))
                points=[]
                if u in life:
                    ts=history.rows[u][0]; k=end(u,cut)
                    points=list(dict.fromkeys(int(ts[i]) for tail in (64,256,1024,2048,4096,6144) if (i:=max(1024,k-tail))<k))
                for stamp in points:
                    old=build(u,stamp,target-1,'lifetime_source',True)
                    sizes.append(replay(old,end(u,cut)-end(u,stamp),'lifetime_source'))
                # One actual Current reference per UID/release, reused by every alternative scene.
                n=build(u,cut,target,'residual_current_teacher',u in life)
                charge('residual_current_teacher_read',query_ops(n,16))
                if u in life:sizes.append(n)
            actual=cal[cal.target==target]
            assert Counter(sizes)==Counter(actual['count']), (target,len(sizes),len(actual))
            for n in sizes:
                p=prepare_cost(target,8)
                costs['residual_source_prepare_bound']+=sum(p[k] for k in ('pack','features','pca','view'))+2*((target+1)*(V+1)+1)+8
                charge('residual_C_read',query_ops(n,16))
                costs['residual_C_correction']+=read_cost(16,1)
            validations.append(dict(target=target,states=len(sizes),length_histogram=dict(Counter(sizes))))
        if target<5:
            stop=CUTOVER_DAYS[target]*DAY
            for u in uids:
                ts=history.rows[u][0]; left,right=end(u,cut),end(u,stop)
                snap=int(ts[max(left,right-4)]) if right>left else stop
                middle=end(u,snap)
                n=replay(lengths[u],middle-left,'lineage')
                early[u]=(n,right-middle,snap)
                lengths[u]=replay(n,right-middle,'lineage')
    # Calibration precedes the fitted affine short-circuits: all scene scores needed.
    # Same exact225D algorithm; no extra teacher and no change to the stored u values.
    costs['residual_geometry_all_scenes']=len(cal)*arithmetic(16)['panel_flops']
    # Explicit generous arithmetic allowance for the fixed2-parameter NNLS and UID cross calibration.
    costs['residual_affine_scalar_allowance']=100_000_000
    original=json.loads((ROOT/'results/design/analysis/native_flops_01/ledger.json').read_text())['native']
    hcost=json.loads((ROOT/'results/design2/analysis/detection_01/cost.json').read_text())['arithmetic']
    inverse=json.loads((ROOT/'results/design2/conditional225_01/inverse/summary.json').read_text())
    # C's normal equations already contain the same Gram. Do not reacquire C inputs
    # or add its Gram twice; only retain/factor/symmetrize the additional H.
    detector=dict(residual_calibration=sum(costs.values()),H_cholesky=hcost['cholesky_leading_flops'],
        H_scalar_allowance=144*64*1281*1281,inverse_preparation=inverse['total_arithmetic_charge'])
    record=dict(status='complete',source='existing512UID timestamps/scene metadata only; zero model forwards or fitting',
        residual_users=512,residual_states=len(cal),residual_components=dict(costs),residual_special_calls=dict(specials),
        residual_scene_validation=validations,source_prefixes=len(built),source_summaries=len(source_summaries),
        source_replay_histogram=dict(counts),design1_components=original['shared_FLOPs'],design1_shared=original['K'],
        detector_components=detector,design2_shared=original['K']+sum(detector.values()),
        reuse_of_H_gram=dict(omitted_duplicate_gram=hcost['gram_main_contraction_flops'],
            note='Frozen C fit_joint already produces the weighted joint Gram with the same feature coordinates. Reuse it; original standalone recovery and research rehydration are not required twice.'),
        residual_calibration_schedule='One512UID source replay; Current KV and16 Exact outputs per UID/release reused across scenes; all scene adapted reads and exact225D scores;8-segment source preparation bound. No teacher historical-append branch substituted for Current rebuild.',
        original_dependency_scope='Retain historical C256 and A64 coordinate/PCA preparation algorithms once each, as recorded by Design1. Do not multiply shared preparation by serving population or methods.',
        scalar_and_special_scope='Ordinary arithmetic plus ELU/SiLU/rsqrt/sin/cos at convention1; calls also listed. Affine fit100M-operation allowance is an explicit accounting budget, not measured solver instruction count.',
        shared_storage=dict(cholesky_bytes=1890397716,inverse_bytes=inverse['storage_bytes']),
        hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'configs/design2/scale_calibration_01_fitted.json',ROOT/'results/design/analysis/native_flops_01/ledger.json']})
    write_json(OUT/'summary.json',record)
    print(json.dumps(dict(design1_shared_TF=record['design1_shared']/1e12,design2_shared_TF=record['design2_shared']/1e12,
        residual_TF=detector['residual_calibration']/1e12,states=len(cal))))


if __name__=='__main__':main()
