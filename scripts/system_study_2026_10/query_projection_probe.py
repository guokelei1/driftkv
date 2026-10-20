"""Paired synthetic-shape numerical/timing check; no quality/users inference."""
import argparse
from pathlib import Path
import sys
import time
import statistics
import json
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
import torch
from implementation_2026_10.backend import InferenceAdapter, ImplementationConfig
from query_relay_2026_10.adapter import UnifiedRelayAdapter
from hstu_kvcache.design_one.lightweight import LightweightValueState
from system_study_2026_10.fused_query_projection import QueryProjectionAdapter


@torch.no_grad()
def run(output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = ROOT/'results/query_relay_2026_10/probes/A_itemgate6_rms_roll4/calibration/v4/calibration.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    adapter = UnifiedRelayAdapter(**payload['config']).cuda().eval().requires_grad_(False)
    adapter.load_state_dict(payload['state_dict'])
    old, new = InferenceAdapter(adapter), QueryProjectionAdapter(adapter)
    attention = SimpleNamespace(position_bias=None, block_variant='legacy', training=False,
                                cfg=SimpleNamespace(activation='elu_plus1'), scale=32**-.5)
    shapes = [(1,1,13,0), (1,1,13,7), (1,4,1024,255), (1,16,1024,1024),
              (8,16,1024,511), (8,32,1024,1024), (3,8,127,65)]
    results = []
    for b,m,n,c in shapes:
        q = torch.randn(b,6,m,32,device='cuda')*.1
        gate = torch.randn_like(q)
        k = torch.randn(b,n,192,device='cuda')*.1
        v = torch.randn_like(k)*.1
        codes = torch.randn(6,b,n,6,8,device='cuda')*.1
        codes[:,:,c:] = 0
        state = LightweightValueState(codes,n)
        # Include a nontrivial per-query causal-style mask, same for both paths.
        mask = torch.arange(n,device='cuda')[None,:] <= n-m+torch.arange(m,device='cuda')[:,None]
        callbacks = [a.make_history_read(state,code_count=c,gate_query=gate,valid_mask=mask) for a in (old,new)]
        reference,actual = [fn(0,attention,q,k,v) for fn in callbacks]
        torch.testing.assert_close(actual,reference,atol=3e-5,rtol=3e-5)
        error=float((actual-reference).abs().max())
        times=[[],[]]
        for fn in callbacks:
            for _ in range(3): fn(0,attention,q,k,v)
        torch.cuda.synchronize()
        for rep in range(10):
            for index in ([0,1] if rep%2==0 else [1,0]):
                torch.cuda.synchronize()
                begin=time.perf_counter()
                callbacks[index](0,attention,q,k,v)
                torch.cuda.synchronize()
                times[index].append((time.perf_counter()-begin)*1000)
        row=dict(batch=b,queries=m,history=n,coded=c,max_abs=error,
                 original_ms=times[0],fused_projection_ms=times[1],
                 medians=[statistics.median(t) for t in times])
        results.append(row)
        print(json.dumps(row),flush=True)
    result=dict(status='passed_query_projection_numerics',shapes=results,
        checkpoint=str(checkpoint),command=sys.argv,seed=17,
        scope='Synthetic shape diagnostic with retained trained read parameters; one-layer callback, prepared gates outside timing. Not real-user service, fitting, or end-to-end throughput.',
        precision='FP32 states; Triton matrix TF32x3; vector projection FP32 reduction; Torch reference TF32 off')
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    run(parser.parse_args().output)
