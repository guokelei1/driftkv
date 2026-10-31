#!/usr/bin/env python3
"""Actual Large parity/speed diagnostic for independent cache-control batches."""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
import json
import os
from pathlib import Path
import sys
import time
from experiments.ch3_motivation.code.common import ROOT, load_edge, load_models, write_json
from experiments.ch3_motivation.code.probe_variants import load_variant
from experiments.ch3_motivation.code.kv_evolution import clone_native_trajectory, teacher_forced_cache
from experiments.ch3_motivation.code.kv_score_batch import score_cache_group
from hstu_kvcache.evaluation.trajectory import NativeTrajectory
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.methods.read_correction.calibration.v2.calibrate import mixed_candidates
from hstu_kvcache.common.artifacts import sha256
import torch


@torch.no_grad()
def run(args):
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu);torch.cuda.set_device(0)
    parent,current,models=load_models(2,torch.device('cuda:0'))
    probe,meta=load_variant(args.checkpoint,'cuda:0');probe.eval()
    b,h,_=load_edge(ROOT/'experiments/ch3_motivation/inputs/edge_2');cfg=b['settings']
    users=json.loads((ROOT/'experiments/ch3_motivation/inputs/population.json').read_text())['main'][:8]
    error=separate_time=batched_time=0.;cases=0
    for uid in users:
        native=NativeTrajectory(parent,1,h.rows[uid],cfg['release_day']*86400,max_length=1024,append_band_size=128)
        reuse=clone_native_trajectory(native);reuse.switch(current,2,native.time)
        teacher=clone_native_trajectory(native);teacher.switch(current,2,native.time,rebuild=True)
        mapped=clone_native_trajectory(reuse)
        ids=torch.as_tensor(native.retained_events[1],device='cuda:0',dtype=torch.long)[None]
        mapped.cache=probe.prepare_cache(current,native.cache,ids);initial=mapped.cache
        candidates=torch.as_tensor(mixed_candidates(uid,native.retained_events[1],current.cfg.num_items-256,count=16,seed=17),device='cuda:0',dtype=torch.long)[None]
        forced=teacher_forced_cache(initial,teacher.cache,0)
        controls=[reuse.cache,teacher.cache,mapped.cache,forced]
        for delta in (torch.tensor([0.],device='cuda:0'),torch.arange(16,device='cuda:0',dtype=torch.float32)[None]):
            # Warm both layouts before the small observed-input timing.
            score_cache_group(current,controls,candidates,delta)
            torch.cuda.synchronize();start=time.perf_counter()
            expected=torch.cat([score(current,c,candidates,delta)[0] for c in controls])
            torch.cuda.synchronize();separate_time+=time.perf_counter()-start
            start=time.perf_counter();actual=score_cache_group(current,controls,candidates,delta)
            torch.cuda.synchronize();batched_time+=time.perf_counter()-start
            error=max(error,float((actual-expected).abs().max()))
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
            cases+=1
    result=dict(status='passed',users=8,cases=cases,max_absolute_score_error=error,
        separate_seconds=separate_time,batched_seconds=batched_time,
        helper_sha256=sha256(Path(__file__).with_name('kv_score_batch.py')),
        checkpoint_sha256=sha256(args.checkpoint),physical_gpu=args.gpu,
        scope='Numerical parity and bounded warm execution diagnostic, not controlled serving speed or quality evidence.')
    write_json(args.output/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=1)
    run(p.parse_args())
