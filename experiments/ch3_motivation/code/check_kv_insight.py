#!/usr/bin/env python3
"""Actual-model checks before the two-case KV-map mechanism study."""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

from experiments.ch3_motivation.code.common import ROOT, load_edge, load_models, write_json
from experiments.ch3_motivation.code.probe_variants import load_variant
from experiments.ch3_motivation.code.kv_evolution import (
    clone_native_trajectory, teacher_forced_cache, checkpoint_metadata,
)
from experiments.ch3_motivation.code.candidate_residual import candidate_residual
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.methods.read_correction.calibration.v2.calibrate import mixed_candidates
from hstu_kvcache.common.artifacts import sha256
import numpy as np
import torch


@torch.no_grad()
def run(args):
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    torch.cuda.set_device(0)
    started=time.perf_counter()
    parent,current,models=load_models(args.edge,torch.device('cuda:0'))
    probe,metadata=load_variant(args.checkpoint,'cuda:0')
    probe.eval().requires_grad_(False)
    if probe.stage!='f' or metadata['identity']['models']!=models:
        raise ValueError('fresh fitted F and backbone identity differ')
    zero=copy.deepcopy(probe)
    for module in zero.f:
        module.weights[-1].zero_();module.biases[-1].zero_()
    binding,histories,_=load_edge(ROOT/f'experiments/ch3_motivation/inputs/edge_{args.edge}')
    cfg=binding['settings'];cutover=cfg['release_day']*86400
    users=json.loads((ROOT/'experiments/ch3_motivation/inputs/population.json').read_text())['main'][:8]
    max_score=max_kv=0.;counts=0
    for uid in users:
        events=histories.rows[uid]
        initial=NativeTrajectory(parent,args.edge-1,events,cutover,max_length=1024,append_band_size=128)
        reuse=clone_native_trajectory(initial);reuse.switch(current,args.edge,cutover)
        null=clone_native_trajectory(reuse)
        initial_ids=torch.as_tensor(initial.retained_events[1],device='cuda:0',dtype=torch.long)[None]
        null.cache=zero.prepare_cache(current,initial.cache,initial_ids)
        mapped=clone_native_trajectory(reuse)
        mapped.cache=probe.prepare_cache(current,initial.cache,initial_ids)
        initial_mapped=mapped.cache
        teacher=clone_native_trajectory(initial);teacher.switch(current,args.edge,cutover,rebuild=True)
        candidates=mixed_candidates(uid,initial.retained_events[1],current.cfg.num_items-256,count=16,seed=17)
        candidates=torch.as_tensor(candidates,device='cuda:0',dtype=torch.long)[None]
        times=[cutover]
        for count in (8,128):
            index=initial.cursor+count-1
            if index<len(events[0]) and events[0][index]+1<cfg['end_day']*86400:
                times.append(int(events[0][index])+1)
        for timestamp in sorted(set(times)):
            for trajectory in (reuse,null,mapped,teacher):trajectory.advance(timestamp)
            delta=torch.tensor([timestamp-reuse.last_timestamp],device='cuda:0',dtype=torch.float32)
            a,_=score(current,reuse.cache,candidates,delta)
            b,_=score(current,null.cache,candidates,delta)
            max_score=max(max_score,float((a-b).abs().max()))
            max_kv=max(max_kv,float((reuse.cache.k-null.cache.k).abs().max()),float((reuse.cache.v-null.cache.v).abs().max()))
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(reuse.cache.k,null.cache.k,atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(reuse.cache.v,null.cache.v,atol=1e-6,rtol=1e-5)
            forced=teacher_forced_cache(initial_mapped,teacher.cache,teacher.evicted_tokens)
            inherited=checkpoint_metadata(teacher,initial_mapped.seq_len)['surviving_inherited_rows']
            torch.testing.assert_close(forced.k[:,:,inherited:],teacher.cache.k[:,:,inherited:],atol=0,rtol=0)
            actual=candidate_residual(current,mapped.cache,teacher.cache,candidates,delta)
            direct,_=score(current,mapped.cache,candidates,delta)
            torch.testing.assert_close(torch.tensor(actual['mapped_scores'],device='cuda:0'),direct[0],atol=1e-6,rtol=1e-5)
            if not np.isfinite(actual['summary']['raw_read_mse']):raise ValueError('nonfinite read residual')
            counts+=1
    sources=[Path(__file__),Path(__file__).with_name('kv_evolution.py'),Path(__file__).with_name('candidate_residual.py')]
    result=dict(status='passed',users=8,checkpoints=counts,max_zero_f_score_error=max_score,
        max_zero_f_kv_error=max_kv,seconds=time.perf_counter()-started,physical_gpu=args.gpu,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),checkpoint_sha256=sha256(args.checkpoint),
        source_sha256={str(p):sha256(p) for p in sources},
        scope='Actual Large numerical/performance diagnostic only; no quality or mechanism conclusion.')
    write_json(args.output/'summary.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--edge',type=int,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=1)
    run(p.parse_args())
