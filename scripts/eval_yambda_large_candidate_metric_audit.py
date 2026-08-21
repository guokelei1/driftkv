#!/usr/bin/env python3
"""Large-candidate audit for frozen Q_main and its declared 10K extension.

The 1K exact Top-K endpoint is the hard same-support audit: it uses precisely
the Q_main support already frozen for panel sampling.  The 10K endpoint keeps
the same pre-release seen filter and rank-decay rule but is explicitly an
extended-support companion, not a redefinition of Q_main.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.stats import spearmanr
from build_yambda_release_snapshot import prepare_catalog_and_popularity
from eval_yambda_cutover_probe_validity import EDGE
from eval_yambda_panel_free_score_distortion import RAW, ROOT, inputs_and_pools, score_batch
from eval_yambda_release_budget_oracle import greedy_indices, summarize
from train_yambda_theta0_medium import build_foundation_data
from train_yambda_two_edges import load_checkpoint

K=10; SIZES=(1000,10000); BUDGETS=(.25,.5,.75)
def regret(full,reuse):
 f=np.argsort(-full,axis=1,kind='stable')[:,:K]; r=np.argsort(-reuse,axis=1,kind='stable')[:,:K]
 return np.maximum(0,(np.take_along_axis(full,f,1).mean(1)-np.take_along_axis(full,r,1).mean(1))/np.maximum(full.std(1),1e-8))
def top_recall(a,b,p=.1):
 n=max(1,int(np.ceil(len(a)*p))); return len(set(np.argsort(-a)[:n])&set(np.argsort(-b)[:n]))/n
def main():
 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); _,popular=prepare_catalog_and_popularity(RAW); _,_,item_map,_=build_foundation_data(RAW,set()); result={'status':'large_candidate_metric_audit_development','same_support_hard_endpoint':'exact Top-K over frozen Q_main top1000 support','extended_support_companion':'exact Top-K over rank-decay top10000 extension; not Q_main primary','target_injection':False,'edges':{}}
 for edge,(release,_,parent_path,current_path) in EDGE.items():
  subset=pq.read_table(f'data/manifests/yambda50m_v2_large_candidate_audit_subset_{edge}.parquet').to_pydict(); keep=set(map(int,subset['uid'])); components={int(u):c for u,c in zip(subset['uid'],subset['component'])}
  snap=pq.read_table(f'data/manifests/yambda50m_v2_release_snapshot_{edge}.parquet').to_pydict(); states={int(u):{k:v[i] for k,v in snap.items()} for i,u in enumerate(snap['uid']) if int(u) in keep}
  risk=pq.read_table(ROOT/f'multi_panel_risk_v1_{edge}.parquet').to_pydict(); held={int(u):float(v) for u,v in zip(risk['uid'],risk['heldout_mean'])}
  parent,_=load_checkpoint(Path(parent_path),device); current,_=load_checkpoint(Path(current_path),device)
  # Reconstruct with the same Q_main filtering; request 10K once and take its prefix for 1K.
  import eval_yambda_panel_free_score_distortion as pfree
  old=pfree.POOL; pfree.POOL=10000
  try: inputs=inputs_and_pools(edge,states,item_map,popular)
  finally: pfree.POOL=old
  records=[]; groups={}
  for uid,(history,pool) in inputs.items(): groups.setdefault(len(history),[]).append((uid,history,pool))
  for group in groups.values():
   for start in range(0,len(group),8):
    chunk=group[start:start+8]; histories=[x[1] for x in chunk]; pools=[x[2] for x in chunk]
    full,reuse=score_batch(parent,current,histories,pools,release,item_map,device)
    for j,(uid,h,pool) in enumerate(chunk):
     row={'uid':uid,'component':components[uid],'heldout_multi_panel_regret':held[uid],'effective_prefix_length':len(h),'exact_cost':states[uid]['exact_token_layer_work']}
     for size in SIZES: row[f'exact_topk_regret_{size}']=float(regret(full[j:j+1,:size],reuse[j:j+1,:size])[0])
     records.append(row)
  metrics={}
  for size in SIZES:
   val=np.asarray([r[f'exact_topk_regret_{size}'] for r in records]); target=np.asarray([r['heldout_multi_panel_regret'] for r in records]); metrics[str(size)]={'spearman_vs_multi_panel':float(spearmanr(val,target).statistic),'top10_recall_vs_multi_panel':top_recall(val,target),'top20_recall_vs_multi_panel':top_recall(val,target,.2)}
  uniform=[r for r in records if r['component']=='uniform']; frontier={}
  for size in SIZES:
   losses=np.asarray([r[f'exact_topk_regret_{size}'] for r in uniform]); costs=np.asarray([r['exact_cost'] for r in uniform],float); total=float(costs.sum()); points=[]
   for b in BUDGETS:
    selected=greedy_indices(np.argsort(-(losses/costs),kind='stable'),costs,total*b); points.append({'budget':b,**summarize('subset_metric_priority',selected,losses,costs,total)})
   frontier[str(size)]=points
  pq.write_table(pa.Table.from_pylist(records),ROOT/f'large_candidate_metric_audit_v1_{edge}.parquet',compression='zstd')
  result['edges'][edge]={'states':len(records),'uniform_states':len(uniform),'metrics':metrics,'uniform_subset_oracle_like_frontier':frontier}
 (ROOT/'large_candidate_metric_audit_v1.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__':main()
