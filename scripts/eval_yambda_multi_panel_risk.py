#!/usr/bin/env python3
"""Compute Q_main multi-panel cutover risk with one Full/Reuse hidden per state."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.stats import spearmanr
from train_yambda_theta0_medium import MAX_HISTORY, build_foundation_data
from train_yambda_two_edges import compact_history_tensors, load_checkpoint
from eval_yambda_cutover_probe_validity import EDGE

PANELS=32; DEV=np.arange(16); HELD=np.arange(16,32); K=10

def regret(full,reuse):
    order_f=np.argsort(-full,axis=-1,kind='stable')[...,:K]; order_r=np.argsort(-reuse,axis=-1,kind='stable')[...,:K]
    ftop=np.take_along_axis(full,order_f,axis=-1).mean(-1); rtop=np.take_along_axis(full,order_r,axis=-1).mean(-1)
    std=np.maximum(full.std(-1),1e-8)
    return np.maximum(0.0,(ftop-rtop)/std), (full.std(-1)<=1e-8)

def top_recall(a,b,f=.1):
    n=max(1,int(np.ceil(len(a)*f))); x=set(np.argsort(-a,kind='stable')[:n]); y=set(np.argsort(-b,kind='stable')[:n]); return len(x&y)/n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge", choices=sorted(EDGE), default=None)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    root=Path('results/data_audit/yambda50m_v2'); raw=Path('data/raw/yambda/flat/50m/listens.parquet')
    device=torch.device(args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    _,_,item_map,_=build_foundation_data(raw,set()); result={'status':'q_main_multi_panel_risk_development','distribution':'Q_main_rank_decay_v1','primary':'mean_current_model_topk_regret','companion':'cvar90_current_model_topk_regret','edges':{}}
    edges = {args.edge: EDGE[args.edge]} if args.edge else EDGE
    for edge,(release,_,parent_path,current_path) in edges.items():
        snap=pq.read_table(f'data/manifests/yambda50m_v2_release_snapshot_{edge}.parquet').to_pydict(); states={int(u):{k:v[i] for k,v in snap.items()} for i,u in enumerate(snap['uid'])}
        panels=pq.read_table(f'data/manifests/yambda50m_v2_qmain32_v2_{edge}.parquet').to_pylist(); panel_map={}
        for r in panels: panel_map.setdefault(int(r['uid']),[]).append(r)
        for rows in panel_map.values(): rows.sort(key=lambda x:x['panel_id'])
        parent,_=load_checkpoint(Path(parent_path),device); current,_=load_checkpoint(Path(current_path),device)
        inputs={}; cur=None; prefix=[]
        def consume(uid,events):
            if uid not in states:return
            eff=[e for e in events if e[0] in item_map][-MAX_HISTORY:]
            if eff: inputs[uid]=eff
        for batch in pq.ParquetFile(raw).iter_batches(batch_size=262144,columns=['uid','timestamp','item_id','is_organic']):
            for u,t,i,o in zip(batch.column('uid').to_numpy(zero_copy_only=False),batch.column('timestamp').to_numpy(zero_copy_only=False),batch.column('item_id').to_numpy(zero_copy_only=False),batch.column('is_organic').to_numpy(zero_copy_only=False)):
                u,t,i,o=int(u),int(t),int(i),int(o)
                if cur is not None and u!=cur: consume(cur,prefix); prefix=[]
                cur=u
                if t<release: prefix.append((i,t,1+(1-o)))
        if cur is not None:consume(cur,prefix)
        if args.max_users is not None:
            selected = set(sorted(inputs)[: args.max_users])
            inputs = {uid: history for uid, history in inputs.items() if uid in selected}
        records=[]; groups={}
        for u,h in inputs.items():groups.setdefault(len(h),[]).append((u,h))
        for group in groups.values():
            for start in range(0,len(group),16):
                chunk=group[start:start+16]; prefixes=[x[1] for x in chunk]; uids=[x[0] for x in chunk]
                readouts=[(0,release,0)]*len(chunk)
                fullparts=[compact_history_tensors(h+[r],item_map,device) for h,r in zip(prefixes,readouts)]; preparts=[compact_history_tensors(h,item_map,device) for h in prefixes]; rparts=[compact_history_tensors([r],item_map,device,previous_timestamp=h[-1][1]) for h,r in zip(prefixes,readouts)]
                fi,fb,fd,fl=[torch.cat([x[j] for x in fullparts]) for j in range(4)]; pi,pb,pd,pl=[torch.cat([x[j] for x in preparts]) for j in range(4)]; ri,rb,rd,_=[torch.cat([x[j] for x in rparts]) for j in range(4)]
                candidate=torch.tensor([[item_map[int(x)] for row in panel_map[u] for x in row['candidate_item_ids']] for u in uids],device=device)
                with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                    hidden,_=current(fi,fb,fd,lengths=fl); fs=current.score_candidates(hidden,candidate,fl).float().cpu().numpy().reshape(len(chunk),PANELS,100)
                    cache=parent.compute_kv(pi,pb,pd,pl); rh,_=current.forward_with_cache(cache,ri,rb,rd); rs=current.score_hidden(rh[:,-1,:],candidate).float().cpu().numpy().reshape(len(chunk),PANELS,100)
                values,floors=regret(fs,rs)
                for j,u in enumerate(uids):
                    v=values[j]
                    tail_count=max(1,int(np.ceil(.1*PANELS)))
                    records.append({'edge_id':edge,'uid':u,'effective_prefix_length':len(prefixes[j]),'exact_token_layer_work':states[u]['exact_token_layer_work'],'panel_regrets':v.tolist(),'dev_mean':float(v[DEV].mean()),'heldout_mean':float(v[HELD].mean()),'cvar90':float(np.sort(v)[-tail_count:].mean()),'within_panel_variance':float(v.var()),'score_std_floor_panels':int(floors[j].sum())})
        dev=np.array([r['dev_mean'] for r in records]); held=np.array([r['heldout_mean'] for r in records]); allv=np.array([r['panel_regrets'] for r in records])
        convergence={str(m):{'spearman':float(spearmanr(allv[:,:m].mean(1),allv[:,16:16+m].mean(1)).statistic),'top10_recall':top_recall(allv[:,:m].mean(1),allv[:,16:16+m].mean(1))} for m in (1,2,4,8,16)}
        result['edges'][edge]={'states':len(records),'split_half_spearman':float(spearmanr(dev,held).statistic),'split_half_top10_recall':top_recall(dev,held),'between_user_variance':float(allv.mean(1).var()),'mean_within_user_panel_variance':float(allv.var(1).mean()),'panel_induced_variance_fraction':float(allv.var(1).mean()/(allv.mean(1).var()+allv.var(1).mean()+1e-12)),'convergence':convergence,'score_std_floor_state_fraction':float(np.mean([r['score_std_floor_panels']>0 for r in records]))}
        pq.write_table(pa.Table.from_pylist(records),root/f'multi_panel_risk_v1{args.output_suffix}_{edge}.parquet',compression='zstd')
    json_suffix = f"{args.output_suffix}_{args.edge}" if args.edge else args.output_suffix
    (root/f'multi_panel_risk_v1{json_suffix}.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__':main()
