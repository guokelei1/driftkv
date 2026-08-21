#!/usr/bin/env python3
"""Add cap-stratified and timing audits to completed dilution curves."""
from __future__ import annotations
import json
from pathlib import Path
import pyarrow.parquet as pq
import numpy as np
ROOT=Path('results/data_audit/yambda50m_v2'); KS=(0,1,2,4,8,16)
def s(rows):
 if not rows:return {'states':0}
 return {'states':len(rows),'mean_regret':float(np.mean([r['mean_regret'] for r in rows])),'mean_panel_free_score_distortion':float(np.mean([r['panel_free_score_distortion'] for r in rows])),'mean_evicted_stale_tokens':float(np.mean([r['evicted_stale_prefix_tokens'] for r in rows]))}
def main():
 base=json.loads((ROOT/'dilution_curve_v2.json').read_text()); out={'status':'dilution_curve_cap_timing_audit_v1','source':'dilution_curve_v2','edges':{}}
 for edge in base['edges']:
  rows=pq.read_table(ROOT/f'dilution_curve_v2_{edge}.parquet').to_pylist(); initial={r['uid']:bool(r['history_cap_hit']) for r in rows if r['append_count']==0}
  edgeout={'initial_cap_hit_strata':{},'time_to_append_seconds':{}}
  for hit in (False,True):edgeout['initial_cap_hit_strata'][str(hit)]={str(k):s([r for r in rows if r['append_count']==k and initial[r['uid']]==hit]) for k in KS}
  for k in KS:
   v=np.asarray([r['append_delay_seconds'] for r in rows if r['append_count']==k and k>0],float)
   edgeout['time_to_append_seconds'][str(k)]={} if not len(v) else {'p50':float(np.quantile(v,.5)),'p90':float(np.quantile(v,.9)),'p95':float(np.quantile(v,.95))}
  out['edges'][edge]=edgeout
 (ROOT/'dilution_curve_cap_timing_audit_v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
