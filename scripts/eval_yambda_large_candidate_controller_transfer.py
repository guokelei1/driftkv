#!/usr/bin/env python3
"""Evaluate frozen-development scheduler selections on large-candidate audits."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from eval_yambda_metadata_risk_ranker import load, models, selected_from_prediction
from eval_yambda_release_budget_oracle import BUDGETS, greedy_indices, summarize
ROOT=Path('results/data_audit/yambda50m_v2'); EDGES=('theta0_theta1','theta1_theta2')
def main():
 out={'status':'large_candidate_controller_transfer_development','selection':'GBDT trained on opposite development edge; budget-matched scheduling within preregistered uniform audit subset','edges':{}}
 for edge,other in [(EDGES[0],EDGES[1]),(EDGES[1],EDGES[0])]:
  train,test=load(other),load(edge); m=models()['hgb_80_leaf8']; m.fit(train['X'],train['label_dev']); pred=np.maximum(m.predict(test['X']),0); total=test['cost'].sum()
  table=pq.read_table(ROOT/f'large_candidate_metric_audit_v1_{edge}.parquet').to_pylist(); uniform=[r for r in table if r['component']=='uniform']; index={int(u):i for i,u in enumerate(test['uid'])}; subset_idx=np.asarray([index[int(r['uid'])] for r in uniform]); costs=np.asarray([r['exact_cost'] for r in uniform],float)
  points={}
  sub_cost=test['cost'][subset_idx]; sub_total=sub_cost.sum()
  for b in (.25,.5,.75):
   cap=sub_total*b; selects={'gbdt':selected_from_prediction(pred[subset_idx],sub_cost,cap),'longest_prefix':greedy_indices(np.argsort(-test['prefix'][subset_idx],kind='stable'),sub_cost,cap),'recent_active':greedy_indices(np.argsort(test['age'][subset_idx],kind='stable'),sub_cost,cap),'highest_activity':greedy_indices(np.argsort(-test['activity'][subset_idx],kind='stable'),sub_cost,cap)}
   values={}
   for size in (1000,10000):
    losses=np.asarray([r[f'exact_topk_regret_{size}'] for r in uniform]); values[str(size)]={}
    for name,selected in selects.items():
     values[str(size)][name]=summarize(name,selected,losses,costs,costs.sum())
   points[str(b)]=values
  out['edges'][edge]={'train_edge':other,'uniform_states':len(uniform),'points':points}
 print(json.dumps(out,indent=2)); (ROOT/'large_candidate_controller_transfer_v1.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
