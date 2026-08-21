#!/usr/bin/env python3
"""Measure append dilution with explicit 512-token physical-cache eviction.

Reuse starts from the materialized parent cache.  When the state cap is
exceeded, old cache entries are physically evicted (sliced) before current
model append; they are never silently regenerated under the parent model.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from build_yambda_release_snapshot import prepare_catalog_and_popularity
from eval_yambda_cutover_probe_validity import EDGE, state_hash
from eval_yambda_multi_panel_risk import PANELS, regret
from train_yambda_theta0_medium import FOUNDATION_END, MAX_HISTORY, build_foundation_data
from train_yambda_two_edges import compact_history_tensors, load_checkpoint
from hstu_kvcache.models import HSTUKVCache

ROOT=Path('results/data_audit/yambda50m_v2'); RAW=Path('data/raw/yambda/flat/50m/listens.parquet'); KS=(0,1,2,4,8,16)
def cache_tail(cache,n):
 if n==cache.seq_len:return cache
 return HSTUKVCache(cache.k[:,:, -n:,:],cache.v[:,:, -n:,:],n)
def js_and_overlap(full,reuse):
 pf=torch.softmax(torch.from_numpy(full),-1); pr=torch.softmax(torch.from_numpy(reuse),-1); mid=(pf+pr)/2
 js=(.5*(pf*(pf.clamp_min(1e-12).log()-mid.clamp_min(1e-12).log())).sum(-1)+.5*(pr*(pr.clamp_min(1e-12).log()-mid.clamp_min(1e-12).log())).sum(-1)).numpy()
 a=np.argsort(-full,-1)[:,:,:10]; b=np.argsort(-reuse,-1)[:,:,:10]; overlap=np.asarray([[len(set(x)&set(y))/10 for x,y in zip(xr,yr)] for xr,yr in zip(a,b)])
 return js,1-overlap
def summary(rows):
 if not rows:return {'states':0}
 out={'states':len(rows)}
 for key in ('mean_regret','cvar90_regret','mean_js','mean_top10_overlap_loss','panel_free_score_distortion','stale_token_fraction'):
  v=np.asarray([r[key] for r in rows]);out[key]={'mean':float(v.mean()),'p50':float(np.median(v)),'p95':float(np.quantile(v,.95))}
 for key in ('retained_stale_prefix_tokens','evicted_stale_prefix_tokens','current_version_append_tokens','history_cap_hit') :out[key]=float(np.mean([r[key] for r in rows]))
 return out
def build_inputs(edge,states,item_map,popular):
 release,next_release,*_=EDGE[edge]; ans={}; cur=None; pre=[]; suf=[]; seen=set()
 def done(uid):
  if uid not in states:return
  eff=[x for x in pre if x[0] in item_map][-MAX_HISTORY:]
  if not eff or state_hash(eff)!=states[uid]['state_hash']:raise ValueError('snapshot mismatch')
  pool=[int(x) for x in popular if int(x) not in seen][:1000]
  ans[uid]=(eff,[x for x in suf if x[0] in item_map],pool)
 for batch in pq.ParquetFile(RAW).iter_batches(batch_size=262144,columns=['uid','timestamp','item_id','is_organic','played_ratio_pct']):
  for u,t,i,o,p in zip(*[batch.column(x).to_numpy(zero_copy_only=False) for x in ['uid','timestamp','item_id','is_organic','played_ratio_pct']]):
   u,t,i,o,p=map(int,(u,t,i,o,p))
   if cur is not None and u!=cur:done(cur);pre=[];suf=[];seen=set()
   cur=u
   if t<release:pre.append((i,t,1+(1-o)))
   elif release<t<next_release:suf.append((i,t,1+(1-o)))
   if t<FOUNDATION_END and p>50:seen.add(i)
 if cur is not None:done(cur)
 return ans
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--max-users',type=int,default=None);parser.add_argument('--output-suffix',default='');args=parser.parse_args()
 dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); _,popular=prepare_catalog_and_popularity(RAW);_,_,imap,_=build_foundation_data(RAW,set()); result={'status':'multi_panel_dilution_development','append_counts':list(KS),'target_injection':False,'physical_cache_eviction':'parent cache tail is sliced at MAX_HISTORY before current-model append','edges':{}}
 for edge,(release,_,pp,cp) in EDGE.items():
  snap=pq.read_table(f'data/manifests/yambda50m_v2_release_snapshot_{edge}.parquet').to_pydict(); states={int(u):{k:v[j] for k,v in snap.items()} for j,u in enumerate(snap['uid'])}; panels=pq.read_table(f'data/manifests/yambda50m_v2_qmain32_v2_{edge}.parquet').to_pylist(); pm={}
  for r in panels:pm.setdefault(int(r['uid']),[]).append(r)
  for v in pm.values():v.sort(key=lambda x:x['panel_id'])
  inputs=build_inputs(edge,states,imap,popular)
  if args.max_users is not None: inputs={u:inputs[u] for u in sorted(inputs)[:args.max_users]}
  parent,_=load_checkpoint(Path(pp),dev);current,_=load_checkpoint(Path(cp),dev);records=[]
  for k in KS:
   eligible=[(u,*x) for u,x in inputs.items() if len(x[1])>=k]; groups={}
   for x in eligible:groups.setdefault(len(x[1]),[]).append(x)
   for group in groups.values():
    for start in range(0,len(group),8):
     chunk=group[start:start+8];uids=[x[0] for x in chunk];prefix=[x[1] for x in chunk];suffix=[x[2][:k] for x in chunk];retain=[min(len(h),MAX_HISTORY-k) for h in prefix]
     # Each group has equal parent cache length; retain is therefore equal too.
     read=[(0,release+(s[-1][1]-release if s else 0),0) for s in suffix]
     fullhist=[h[-r:]+s for h,s,r in zip(prefix,suffix,retain)]; fp=[compact_history_tensors(h+[q],imap,dev) for h,q in zip(fullhist,read)]; ppv=[compact_history_tensors(h,imap,dev) for h in prefix]; ap=[compact_history_tensors(s+[q],imap,dev,previous_timestamp=h[-1][1]) for h,s,q in zip(prefix,suffix,read)]
     fi,fb,fd,fl=[torch.cat([x[i] for x in fp]) for i in range(4)];pi,pb,pd,pl=[torch.cat([x[i] for x in ppv]) for i in range(4)];ai,ab,ad,_=[torch.cat([x[i] for x in ap]) for i in range(4)]
     cand=torch.tensor([[imap[int(z)] for row in pm[u] for z in row['candidate_item_ids']] for u in uids],device=dev); qpool=torch.tensor([[imap[int(z)] for z in x[3]] for x in chunk],device=dev)
     with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
      fh,_=current(fi,fb,fd,lengths=fl); fs=current.score_candidates(fh,cand,fl).float().cpu().numpy().reshape(len(chunk),PANELS,100); fq=current.score_candidates(fh,qpool,fl).float().cpu().numpy()
      old=parent.compute_kv(pi,pb,pd,pl); old=cache_tail(old,retain[0]); rh,_=current.forward_with_cache(old,ai,ab,ad); rs=current.score_hidden(rh[:,-1],cand).float().cpu().numpy().reshape(len(chunk),PANELS,100); rq=current.score_hidden(rh[:,-1],qpool).float().cpu().numpy()
     reg,_=regret(fs,rs);js,ol=js_and_overlap(fs,rs); w=np.arange(1,1001,dtype=float)**-.5;w/=w.sum();dist=np.sqrt(((fq-rq)**2*w).sum(1))/np.maximum(np.sqrt((fq**2*w).sum(1)),1e-8)
     for j,u in enumerate(uids):
      records.append({'edge_id':edge,'uid':u,'append_count':k,'complete_case':len(inputs[u][1])>=16,'mean_regret':float(reg[j].mean()),'cvar90_regret':float(np.sort(reg[j])[-4:].mean()),'mean_js':float(js[j].mean()),'mean_top10_overlap_loss':float(ol[j].mean()),'panel_free_score_distortion':float(dist[j]),'retained_stale_prefix_tokens':retain[j],'evicted_stale_prefix_tokens':len(prefix[j])-retain[j],'current_version_append_tokens':k,'stale_token_fraction':retain[j]/max(1,retain[j]+k),'history_cap_hit':int(len(prefix[j])+k>=MAX_HISTORY),'append_delay_seconds':0 if not k else suffix[j][-1][1]-release})
  points={};
  for cohort,rows in [('all_available',records),('complete_case',[r for r in records if r['complete_case']])]:points[cohort]={str(k):summary([r for r in rows if r['append_count']==k]) for k in KS}
  complete=[r for r in records if r['complete_case']]; mono=[]
  for u in {r['uid'] for r in complete}:
   v=[next(r['mean_regret'] for r in complete if r['uid']==u and r['append_count']==k) for k in KS];mono.append(all(b<=a+1e-9 for a,b in zip(v,v[1:])))
  result['edges'][edge]={'points':points,'complete_case_individual_monotonic_fraction':float(np.mean(mono)),'complete_case_nonmonotonic_fraction':float(1-np.mean(mono))}
  pq.write_table(pa.Table.from_pylist(records),ROOT/f'dilution_curve_v2{args.output_suffix}_{edge}.parquet',compression='zstd')
 (ROOT/f'dilution_curve_v2{args.output_suffix}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
