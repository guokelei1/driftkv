#!/usr/bin/env python3
"""Native-pair calibration followed by chained predicted-state inference.

Only the calibration-input protocol changes. The ordinary W64/256 row mapper,
target KV objective, users, sampled rows and seed are unchanged. Direct and
stage2 are copied bit-for-bit from the already verified W64/256 reference.
This control diagnoses input-distribution mismatch, not inevitable KV loss.
"""
from __future__ import annotations

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

ROOT=Path(__file__).resolve().parents[3]
import torch
from experiments.ch3_motivation.code.common import write_json
from experiments.ch3_motivation.code.state_lineage_probe import fit_map, load_bank
from hstu_kvcache.common.artifacts import sha256

SWEEP=ROOT/'experiments/ch3_motivation/runs/simple_config_sweep/attempt_1'
LINEAGE=ROOT/'experiments/ch3_motivation/runs/lineage_candidate/attempt_1'


def mse(actual,teacher):
    with torch.no_grad():return float((actual-teacher).square().mean())


def fit_native_pairs(rows,reference_bank,reference_reports,*,target,hidden=64,steps=256,
                     batch_rows=512,lr=1e-3,seed=17):
    """Native inputs select weights; actual-chain inputs are diagnostics only."""
    bank={2:{'chain':copy.deepcopy(reference_bank[2]['chain'])}}
    reports={2:{'chain':dict(copy.deepcopy(reference_reports['2']['chain']),exact_reference_weight_copy=True)}}
    original,original_val=rows[1]['fit'],rows[1]['validation']
    with torch.no_grad():
        previous=bank[2]['chain'](original,original)
        previous_val=bank[2]['chain'](original_val,original_val)
    diagnostics={}
    for version in range(3,target+1):
        native,native_val=rows[version-1]['fit'],rows[version-1]['validation']
        teacher,teacher_val=rows[version]['fit'],rows[version]['validation']
        model,report=fit_map(native,native,teacher,native_val,native_val,teacher_val,
            hidden=hidden,steps=steps,batch_rows=batch_rows,lr=lr,seed=seed,validate_every=32)
        with torch.no_grad():
            native_prediction=model(native_val,native_val)
            actual_prediction=model(previous_val,previous_val)
            recomputed=mse(native_prediction,teacher_val)
            if abs(recomputed-report['validation_kv_mse'])>1e-7:
                raise ValueError('Selected checkpoint must match native-input validation MSE')
            diagnostics[version]=dict(native_input_validation_mse=recomputed,
                actual_chain_input_validation_mse=mse(actual_prediction,teacher_val),
                actual_minus_native_validation_mse=mse(actual_prediction,teacher_val)-recomputed,
                fit_input_native_minus_actual_chain_mse=mse(native,previous),
                validation_input_native_minus_actual_chain_mse=mse(native_val,previous_val),
                actual_chain_diagnostics_used_for_selection=False)
            previous=model(previous,previous)
            previous_val=actual_prediction
        bank[version]={'chain':model};reports[version]={'chain':dict(report,
            calibration_input=f'native V{version-1} rows, duplicated',
            inference_input='Actual predicted predecessor from this native-pair chain, duplicated',
            checkpoint_selection='Native-input validation teacher KV MSE only')}
        print(json.dumps(dict(stage='native_pair_fit',version=version,selected_step=report['selected_step'],
            **diagnostics[version])),flush=True)
    bank[target]['direct']=copy.deepcopy(reference_bank[target]['direct'])
    reports[target]['direct']=dict(copy.deepcopy(reference_reports[str(target)]['direct']),exact_reference_weight_copy=True)
    return bank,reports,diagnostics


def assert_exact_reference(reference,parity,target):
    """The reused stage2/direct weights must match both retained references."""
    checked={}
    for version,kind in ((2,'chain'),(target,'direct')):
        a,b=reference['maps'][version][kind],parity['maps'][version][kind]
        if set(a)!=set(b) or any(not torch.equal(a[name],b[name]) for name in a):
            raise ValueError('Retained old/cached-row reference weights are not bitwise equal')
        checked[f'v{version}_{kind}']=dict(all_tensors_bitwise_equal=True,tensors=len(a),elements=sum(x.numel() for x in a.values()))
    return checked


def run(args):
    if args.gpu!=1:raise ValueError('Physical GPU1 only')
    cache_path=args.cache_dir/'rows.pt';cache_summary_path=args.cache_dir/'summary.json'
    cached=json.loads(cache_summary_path.read_text())
    if cached['status']!='complete' or sha256(cache_path)!=cached['checkpoint_sha256']:
        raise ValueError('Completed immutable native-row capture required')
    capture=torch.load(cache_path,map_location='cpu',weights_only=True,mmap=True)
    ci=capture['identity']
    if capture['format']!='state_lineage_native_rows_v1' or ci!=cached['identity']:
        raise ValueError('Capture payload identity mismatch')
    population=json.loads(args.population.read_text())
    if (ci['source_version']!=1 or ci['final_target']!=args.final_target or ci['row_budget']!=128
            or ci['population_sha256']!=sha256(args.population) or len(ci['fit_uids'])!=100
            or len(ci['validation_users'])!=16 or set(population['users'])&set(ci['fit_uids']+ci['validation_users'])):
        raise ValueError('Fixed source/target/history/fit100/val16/population required')
    reference_path=args.reference/'maps.pt';reference_summary=json.loads((args.reference/'summary.json').read_text())
    parity_path=args.parity/'maps.pt';parity_summary=json.loads((args.parity/'summary.json').read_text())
    for path,summary in ((reference_path,reference_summary),(parity_path,parity_summary)):
        if summary['status']!='complete' or sha256(path)!=summary['checkpoint_sha256']:
            raise ValueError('Immutable complete reference checkpoint required')
    reference=torch.load(reference_path,map_location='cpu',weights_only=True,mmap=True)
    parity=torch.load(parity_path,map_location='cpu',weights_only=True,mmap=True)
    ri=reference['identity']
    keys=('source_version','final_target','cutoff','history_cap','fit_uids','validation_users',
        'population_sha256','fit_population_sha256','input_binding_sha256','history_sha256','models','row_budget')
    if any(ri[k]!=ci[k] or parity['identity'][k]!=ci[k] for k in keys):
        raise ValueError('Native rows and reused weights must share exact snapshots, models and roles')
    if (reference['format']!='state_lineage_maps_v1' or reference['hidden']!=64
            or any(ri[k]!=v for k,v in dict(hidden=64,steps=256,batch_rows=512,lr=.001,seed=17).items())):
        raise ValueError('This bounded control reuses W64/256/seed17 only')
    parity_checks=assert_exact_reference(reference,parity,args.final_target)
    files=[Path(__file__),Path(__file__).with_name('state_lineage_probe.py'),
        Path(__file__).with_name('simple_kv_maps.py'),Path(__file__).with_name('common.py')]
    source_hashes={str(p.resolve()):sha256(p) for p in files}
    # Verify the existing captures/reused functions without changing their files.
    for input_identity in (ci,ri,parity['identity']):
        if any(sha256(Path(p))!=h for p,h in input_identity['sources'].items()):
            raise ValueError('A frozen capture/reference numerical source changed')
    identity={key:copy.deepcopy(ci[key]) for key in keys}
    identity.update(format='state_lineage_native_pair_fit_v1',versions=ci['versions'],hidden=64,steps=256,
        batch_rows=512,lr=.001,seed=17,sources=source_hashes,torch_version=str(torch.__version__),physical_gpu=1,
        native_row_cache=str(cache_path.resolve()),native_row_cache_sha256=sha256(cache_path),
        reference_checkpoint=str(reference_path.resolve()),reference_checkpoint_sha256=sha256(reference_path),
        reference_parity_checkpoint=str(parity_path.resolve()),reference_parity_checkpoint_sha256=sha256(parity_path),
        exact_reused_maps=['v2_chain',f'v{args.final_target}_direct'],reference_parity=parity_checks,
        objective='Raw native teacher KV MSE; same immutable ordinary row MLP; no candidate/label data',
        calibration_protocol='Stages>=3 fit independent native(t−1)→native(t) pairs, not predicted previous states',
        inference_protocol='Chain still applies every selected map to its own predicted predecessor',
        checkpoint_selection='Native-input validation MSE only; actual-chain validation is a held-out shift diagnostic, never selection',
        comparison_scope='Input-protocol sensitivity; not an inevitable failure of rewritten KV or evidence independent state is necessary',
        snapshot_scope='Same final-target cutoff/history at all captured model versions; unchanged cached-prefix-gap scorer')
    args.output.mkdir(parents=True,exist_ok=True);cp=args.output/'configuration.json'
    if cp.exists() and json.loads(cp.read_text())!=identity:raise ValueError('Frozen native-pair attempt changed')
    write_json(cp,identity)
    if (args.output/'maps.pt').exists():raise ValueError('Completed checkpoint already exists; no duplicate fit')
    os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['EVOKV_ATTENTION_BACKEND']='triton'
    torch.cuda.set_device(0);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda:0');torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    rows={version:{split:x.to(device) for split,x in values.items()} for version,values in capture['rows'].items()}
    input_versions={(version,split):x._version for version,values in rows.items() for split,x in values.items()}
    reference_bank,_=load_bank(reference_path,device)
    bank,reports,diagnostics=fit_native_pairs(rows,reference_bank,reference_summary['stages'],target=args.final_target)
    if any(x._version!=input_versions[version,split] for version,values in rows.items() for split,x in values.items()):
        raise ValueError('Fit mutated captured native rows')
    if any(sha256(Path(p))!=h for p,h in source_hashes.items()):raise ValueError('Numerical source changed during fit')
    checkpoint=dict(format='state_lineage_maps_v1',pairs=reference['pairs'],width=reference['width'],hidden=64,identity=identity,
        maps={v:{kind:{name:value.cpu() for name,value in model.state_dict().items()} for kind,model in stage.items()}
            for v,stage in bank.items()})
    assert_exact_reference(checkpoint,reference,args.final_target)
    temporary=args.output/'maps.pt.writing';torch.save(checkpoint,temporary);temporary.replace(args.output/'maps.pt')
    summary=dict(status='complete',identity=identity,checkpoint_sha256=sha256(args.output/'maps.pt'),stages=reports,short_stages={},
        native_vs_actual_chain_validation=diagnostics,history_lengths=cached['history_lengths'],
        sampled_fit_rows=cached['sampled_fit_rows'],sampled_validation_rows=cached['sampled_validation_rows'],
        new_fitted_stages=list(range(3,args.final_target+1)),new_optimizer_steps=256*(args.final_target-2),
        exact_copied_stage2_and_final_direct=True,backbone_capture_reused=True,quality_auc_computed=False,
        diagnostic_did_not_select_checkpoint=True,physical_gpu=1,seconds=time.perf_counter()-started,
        peak_reserved_bytes=torch.cuda.max_memory_reserved())
    write_json(args.output/'summary.json',summary);return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final-target',type=int,choices=(3,4),required=True)
    parser.add_argument('--population',type=Path,default=LINEAGE/'population.json')
    parser.add_argument('--cache-dir',type=Path);parser.add_argument('--reference',type=Path);parser.add_argument('--parity',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--gpu',type=int,default=1)
    args=parser.parse_args()
    if args.cache_dir is None:args.cache_dir=SWEEP/f'captures/final_{args.final_target}'
    if args.reference is None:args.reference=LINEAGE/f'state/final_{args.final_target}'
    if args.parity is None:args.parity=SWEEP/f'parity/w64_s256/final_{args.final_target}'
    try:run(args);write_json(args.output/'exit.json',dict(exit_code=0))
    except BaseException as error:write_json(args.output/'exit.json',dict(exit_code=1,error=str(error)));raise
