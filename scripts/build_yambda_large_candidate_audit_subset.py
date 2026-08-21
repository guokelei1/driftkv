#!/usr/bin/env python3
"""Pre-register release states for large-candidate metric audits.

The uniform component is the population-valid reporting subset.  The other
components deliberately cover risk, prefix/activity and recency strata for
diagnostics only; they must not be pooled unweighted into a population claim.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path('data/manifests'); RISK=Path('results/data_audit/yambda50m_v2')
EDGES=('theta0_theta1','theta1_theta2'); SEED=20260818

def bins(values,n):
 out=np.empty(len(values),dtype=int); order=np.argsort(values,kind='stable')
 for i,g in enumerate(np.array_split(order,n)): out[g]=i
 return out

def choose(rng, pool, n, taken):
 options=np.asarray([x for x in pool if x not in taken],dtype=int)
 return rng.choice(options,size=min(n,len(options)),replace=False).tolist()

def main():
 ROOT.mkdir(parents=True,exist_ok=True); result={'status':'preregistered_large_candidate_audit_subset_v1','seed':SEED,'components':{'uniform':128,'risk_decile':128,'prefix_activity':128,'recency':128},'edges':{}}
 for edge in EDGES:
  snap=pq.read_table(f'data/manifests/yambda50m_v2_release_snapshot_{edge}.parquet').to_pydict(); risk=pq.read_table(RISK/f'multi_panel_risk_v1_{edge}.parquet').to_pydict()
  byuid={int(u):i for i,u in enumerate(snap['uid'])}; uids=np.asarray(risk['uid'],dtype=np.int64); idx=np.asarray([byuid[int(u)] for u in uids])
  values=np.asarray(risk['heldout_mean'],float); prefix=np.asarray(snap['effective_prefix_length'],float)[idx]; activity=np.asarray(snap['events_last_7d'],float)[idx]; recency=np.asarray(snap['last_activity_age_seconds'],float)[idx]
  rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'{SEED}:{edge}'.encode()).digest()[:8],'little')); taken=set(); rows=[]
  for position in choose(rng,range(len(uids)),128,taken): taken.add(position); rows.append({'edge_id':edge,'uid':int(uids[position]),'component':'uniform','population_weight':len(uids)/128})
  for decile in range(10):
   for position in choose(rng,np.flatnonzero(bins(values,10)==decile),13 if decile<8 else 12,taken): taken.add(position); rows.append({'edge_id':edge,'uid':int(uids[position]),'component':'risk_decile','risk_decile':decile+1,'population_weight':None})
  pbin,abin=bins(prefix,4),bins(activity,4)
  for cell in range(16):
   for position in choose(rng,np.flatnonzero((pbin*4+abin)==cell),8,taken): taken.add(position); rows.append({'edge_id':edge,'uid':int(uids[position]),'component':'prefix_activity','prefix_quartile':int(cell//4+1),'activity_quartile':int(cell%4+1),'population_weight':None})
  rbin=bins(recency,4)
  for quartile in range(4):
   for position in choose(rng,np.flatnonzero(rbin==quartile),32,taken): taken.add(position); rows.append({'edge_id':edge,'uid':int(uids[position]),'component':'recency','recency_quartile':quartile+1,'population_weight':None})
  # Components are allowed to overlap only through the explicitly uniform set;
  # de-duplication above makes the state sample exactly 512 where possible.
  pq.write_table(pa.Table.from_pylist(rows),ROOT/f'yambda50m_v2_large_candidate_audit_subset_{edge}.parquet',compression='zstd')
  result['edges'][edge]={'states':len(rows),'uniform_states':sum(r['component']=='uniform' for r in rows),'uid_hash':hashlib.sha256(','.join(map(str,sorted(r['uid'] for r in rows))).encode()).hexdigest()}
 print(json.dumps(result,indent=2)); (RISK/'large_candidate_audit_subset_v1.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__': main()
