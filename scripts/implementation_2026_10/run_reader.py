#!/usr/bin/env python3
"""Standalone GPU numerical/engineering probe; no fitting or population run."""
from __future__ import annotations
import argparse
import hashlib
import gc
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import torch
from implementation_2026_10.reader import fused_relay_read,reader_work_model
from query_relay_2026_10.adapter import UnifiedRelayAdapter,operation_gate_query


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def attention():
    return SimpleNamespace(position_bias=None,block_variant='legacy',
        cfg=SimpleNamespace(activation='elu_plus1'),training=False,
        attn_dropout=torch.nn.Identity(),scale=32**-.5,
        _activate=lambda s:torch.nn.functional.elu(s)+1)


def difference(a,b):
    delta=(a-b).double()
    return dict(max_abs=float(delta.abs().max()) if a.numel() else 0.,
        rms=float(delta.square().mean().sqrt()) if a.numel() else 0.,
        relative_l2=float(delta.norm()/a.double().norm().clamp_min(1e-30)),
        reference_abs_max=float(a.abs().max()) if a.numel() else 0.,
        max_abs_over_reference_peak=float(delta.abs().max()/a.double().abs().max().clamp_min(1e-30)) if a.numel() else 0.,
        finite=bool(torch.isfinite(b).all()))


def oracle_read64(layer,attn,q,k,v,codes,count,gate_q):
    """High-precision diagnostic with the same FP32 prepared projections/gates.

    Native/correction dot products, ELU+1 and both aggregates use FP64. The
    mathematical target is separated from either FP32 reduction order.
    """
    b,h,m,d=q.shape;n=k.shape[1]
    keys=k.reshape(b,n,h,d).transpose(1,2).double()
    values=v.reshape(b,n,h,d).transpose(1,2).double()
    scores=q.double()@keys.transpose(-2,-1)
    if count:
        z=layer.accessible_codes(codes[:,:count]).double()
        gk,gv=layer.query_gates(gate_q)
        projected=((q@layer.key_decoder[:,layer.read_slots].transpose(-2,-1)[None])*gk).double()
        scores=torch.cat((scores[...,:count]+projected@z.transpose(-2,-1),scores[...,count:]),-1)
    weights=torch.nn.functional.elu(scores*attn.scale)+1
    output=weights@values
    if count:
        output=output+((weights[...,:count]@z)*gv.double())@layer.decoder[:,layer.read_slots].double()[None]
    return output


@torch.no_grad()
def synthetic_grid(device):
    torch.manual_seed(177)
    layer=UnifiedRelayAdapter(1,mode='A',read_shared=3,read_private=3,
        gate_normalization='rms',gate_source='item').to(device).eval().layers[0]
    for p in (layer.key_decoder,layer.decoder,layer.key_gate_weight,layer.value_gate_weight):
        torch.nn.init.normal_(p,std=.1)
    attn=attention();rows=[]
    for n in (0,1,17,63,64,65,127,513,1024):
        for queries in (1,2,4,5,16,32):
            for count in sorted(set((0,min(1,n),n//2,n))):
                for mask_name in ('none','exclusive_fifo','empty' if queries==1 else 'numeric_batch'):
                    b=2 if mask_name=='numeric_batch' else 1
                    q=torch.randn(b,queries,6,32,device=device).transpose(1,2)
                    k=torch.randn(b,n+9,192,device=device)[:,9:]
                    v=torch.randn_like(k)
                    # Allocate only the active prefix: zero suffix cannot be read by accident.
                    z=torch.randn(b,count,6,8,device=device)
                    gate_q=torch.randn_like(q)
                    mask=None
                    if mask_name=='exclusive_fifo':
                        pos=torch.arange(queries,device=device)+max(0,n-queries)
                        col=torch.arange(n,device=device)
                        mask=(col[None,:]<pos[:,None])&(col[None,:]>pos[:,None]-71)
                    elif mask_name=='empty':mask=torch.zeros(queries,n,device=device,dtype=torch.bool)
                    elif mask_name=='numeric_batch':mask=torch.rand(b,queries,n,device=device)
                    expected=fused_relay_read(layer,attn,q,k,v,z,valid_mask=mask,
                        code_count=count,gate_q=gate_q,backend='torch')
                    actual,stats=fused_relay_read(layer,attn,q,k,v,z,valid_mask=mask,
                        code_count=count,gate_q=gate_q,return_stats=True)
                    err=difference(expected,actual)
                    torch.testing.assert_close(actual,expected,atol=3e-4,rtol=3e-5)
                    if count:
                        # Poison every unused persistent coordinate. Direct selected-coordinate
                        # addressing must ignore these values without a per-head materialization.
                        poisoned=z.clone().flatten(-2)
                        unused=[3,4,5,6,7,8,9,10,11]
                        unused += [12+h*6+r for h in range(6) for r in (3,4,5)]
                        poisoned[...,unused]=10000
                        again=fused_relay_read(layer,attn,q,k,v,poisoned,valid_mask=mask,
                            code_count=count,gate_q=gate_q)
                        torch.testing.assert_close(again,actual,atol=0,rtol=0)
                    rows.append(dict(batch=b,history=n,queries=queries,code_count=count,
                        mask=mask_name,error=err,backend=stats['backend'],kernel=stats['kernel'],fallback=stats['fallback']))
    return rows


@torch.no_grad()
def real_values(device,users):
    from evaluate_yambda500m_foundation_raw import load_model
    from hstu_kvcache.adaptation.reader import score
    from hstu_kvcache.design_one.lightweight import LightweightValueState
    source_path=ROOT/'results/query_relay_2026_10/probes/A_itemgate6_rms_roll4/source_v4/source.pt'
    cal_path=ROOT/'results/query_relay_2026_10/probes/A_itemgate6_rms_roll4/calibration/v3/calibration.pt'
    model_path=ROOT/'results/unified_training_2026_09/medium/seed17/checkpoints/v3/checkpoint_100.pt'
    payload=torch.load(cal_path,map_location='cpu',weights_only=False)
    adapter=UnifiedRelayAdapter(**payload['config']).to(device).eval().requires_grad_(False)
    adapter.load_state_dict(payload['state_dict'])
    model,_=load_model(model_path,device)
    model.eval().requires_grad_(False)
    source=torch.load(source_path,map_location='cpu',weights_only=False,mmap=True)
    records={row['uid']:row for row in source['records']}
    results=[];worst_inputs=None;worst_error=-1
    source_cases=[(uid,row,qcount) for uid,row in list(source['rows'].items())[:users]
                  for qcount in (1,4,16)]
    for uid,row,qcount in source_cases:
        cache=row['parent'].to(device)
        state=LightweightValueState(row['previous_codes'].to(device),cache.seq_len)
        candidates=row['candidates'][:qcount].to(device)[None]
        gate_q=operation_gate_query(adapter,model,candidates)
        count=int(records[uid]['coded_rows'])
        errors=[]
        def reference(index,attn,q,k,v):
            return adapter.layers[index].read(attn,q,k,v,state.codes[index],
                code_count=count,gate_q=gate_q)
        def optimized(index,attn,q,k,v):
            nonlocal worst_inputs,worst_error
            l=adapter.layers[index]
            original=l.read(attn,q,k,v,state.codes[index],code_count=count,gate_q=gate_q)
            actual=fused_relay_read(l,attn,q,k,v,state.codes[index],code_count=count,gate_q=gate_q)
            errors.append(dict(layer=index,**difference(original,actual)))
            # Raw unnormalized ELU+1 reads can be >1e5 and contain cancelled
            # near-zero coordinates. Judge whole-read numerical drift and
            # the final frozen-model logits separately, not a fixed raw .01.
            err=errors[-1]
            print(json.dumps(dict(status='real_layer',uid=int(uid),**err)),flush=True)
            if not err['finite'] or err['relative_l2']>2e-6 or err['max_abs_over_reference_peak']>2e-6:
                raise AssertionError(f"real reader normalized discrepancy {err}")
            if err['max_abs_over_reference_peak']>worst_error:
                worst_error=err['max_abs_over_reference_peak']
                worst_inputs=(int(uid),index,attn,q.detach().clone(),k,v,state.codes[index],
                              count,gate_q,original.detach().clone(),actual.detach().clone())
            return actual
        delta=torch.tensor([row['query_delta']],device=device,dtype=torch.float32)
        old=score(model,cache,candidates,delta,history_read_fn=reference)[0]
        new=score(model,cache,candidates,delta,history_read_fn=optimized)[0]
        score_error=difference(old,new)
        print(json.dumps(dict(status='real_score',uid=int(uid),**score_error)),flush=True)
        torch.testing.assert_close(new,old,atol=2e-5,rtol=2e-5)
        results.append(dict(uid=int(uid),history=cache.seq_len,code_count=count,
            kernel='vector_fp32' if qcount<=4 else 'matrix_tf32x3',
            queries=candidates.numel(),reader_errors=errors,logit_error=score_error))
    bindings={str(p):digest(p) for p in (source_path,cal_path,model_path)}
    uid,index,attn,q,k,v,codes,count,gq,old,new=worst_inputs
    oracle=oracle_read64(adapter.layers[index],attn,q,k,v,codes,count,gq)
    oracle_result=dict(uid=uid,layer=index,selected_by='Largest max-absolute discrepancy normalized by Torch output peak',
        torch_vs_fp64=difference(oracle,old.double()),triton_vs_fp64=difference(oracle,new.double()),
        fixed_preparation='Query projection and gates computed in the same FP32 form; history dot/activation/aggregation FP64')
    return dict(scope='Numerical equivalence only; first frozen calibration source users, no labels or fitting',
                source_service_version=3,source_at_next_release=True,users=results,bindings=bindings,
                worst_layer_oracle=oracle_result)


@torch.no_grad()
def memory_probe(device):
    """Peak live PyTorch CUDA allocations after warmup, excluding retained input.

    This does not count on-chip registers/shared memory or measure DRAM bytes.
    Output and operation-sized gate/projection work are included in both paths.
    """
    layer=UnifiedRelayAdapter(1,mode='A',read_shared=3,read_private=3,
        gate_normalization='rms',gate_source='item').to(device).eval().layers[0]
    for p in (layer.key_decoder,layer.decoder,layer.key_gate_weight,layer.value_gate_weight):
        torch.nn.init.normal_(p,std=.1)
    rows=[]
    for queries in (1,4,16,32):
        for count in (0,512,1024):
            q=torch.randn(1,6,queries,32,device=device)
            k=torch.randn(1,1024,192,device=device);v=torch.randn_like(k)
            z=torch.randn(1,count,6,8,device=device);gate_q=torch.randn_like(q)
            peaks={}
            for backend in ('torch','triton'):
                for _ in range(2):
                    out=fused_relay_read(layer,attention(),q,k,v,z,
                        code_count=count,gate_q=gate_q,backend=backend)
                    del out
                torch.cuda.synchronize(device);gc.collect()
                start=torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                out=fused_relay_read(layer,attention(),q,k,v,z,
                    code_count=count,gate_q=gate_q,backend=backend)
                torch.cuda.synchronize(device)
                peaks[backend]=dict(peak_added_allocated_bytes=torch.cuda.max_memory_allocated(device)-start,
                    live_output_bytes=out.numel()*out.element_size())
                del out
            rows.append(dict(queries=queries,history=1024,code_count=count,backends=peaks))
    return dict(scope='Warm live PyTorch CUDA tensor allocations incl output; not CUDA reserved memory or on-chip scratch, not DRAM traffic',rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/implementation_2026_10/reader')
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--real-users',type=int,default=8)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    began=time.time();torch.set_num_threads(2)
    device=torch.device('cuda',args.gpu)
    torch.cuda.set_device(device)
    config=dict(seed=177,gpu=args.gpu,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        device=torch.cuda.get_device_name(device),torch=torch.__version__,
        fp32_input_storage_accumulation=True,
        kernel_precision=dict(query_le_4='FP32 multiply and reductions, no MMA',
                              query_gt_4='TF32x3 multiplication with FP32 accumulation'),
        inference_only=True,threads=2,real_users=args.real_users,
        synthetic_tolerance=dict(atol=3e-4,rtol=3e-5),
        initial_development_failures=[
            dict(issue='Unused HSTUAttention symbol imported in ad-hoc smoke test',fix='Removed unused import; no kernel executed'),
            dict(issue='Triton z not defined after dynamic conditional',fix='Initialize register tile before conditional code load'),
            dict(issue='Source inspection lacked repository PYTHONPATH',fix='Use explicit scripts:src for standalone scripts')])
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    synthetic=synthetic_grid(device)
    (args.output/'synthetic.json').write_text(json.dumps(synthetic,indent=2)+'\n')
    print(json.dumps(dict(status='synthetic_complete',cases=len(synthetic)),ensure_ascii=False),flush=True)
    real=real_values(device,args.real_users)
    memory=memory_probe(device)
    report=dict(status='complete',config=config,synthetic=synthetic,real=real,memory=memory,
        synthetic_cases=len(synthetic),all_finite=all(x['error']['finite'] for x in synthetic),
        worst_synthetic_max_abs=max(x['error']['max_abs'] for x in synthetic),
        worst_synthetic_relative_l2=max(x['error']['relative_l2'] for x in synthetic),
        work_models=[dict(queries=q,code_count=c,**reader_work_model(batch=1,queries=q,
            history=1024,code_count=c,has_mask=False)) for q in (1,4,16,32) for c in (0,512,1024)],
        source_hashes={str(ROOT/name):digest(ROOT/name) for name in (
            'scripts/implementation_2026_10/reader.py','scripts/implementation_2026_10/reader_kernel.py',
            'scripts/implementation_2026_10/run_reader.py','tests/implementation_2026_10/test_reader.py')},
        elapsed_seconds=time.time()-began,
        limits=['Logical FLOPs unchanged for active pairs; fusion removes global intermediates, not arithmetic.',
            'No BF16/backward/dropout/relative bias support; unsupported Triton inputs raise, no fallback.',
            'No loaded throughput or p99 benchmark, no measured DRAM traffic claim.',
            'FP32 reduction order differs from PyTorch; numerical tolerance is not bitwise equality.',
            'Query<=4 uses one Query/head vector FP32 kernel; Query>4 uses tile16/history64/rank16 TF32x3 matrix kernel.',
            'Precision-lowered/padded instruction work is not the logical FLOPs ledger.'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','synthetic_cases','worst_synthetic_max_abs',
        'worst_synthetic_relative_l2','elapsed_seconds')}),flush=True)


if __name__=='__main__':main()
