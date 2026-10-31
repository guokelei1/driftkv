#!/usr/bin/env python3
"""Actual Large/edge2 numerical check, eight fit + two validation users only.

No AUC or fitted checkpoint is written. This checks parameter freezing and
optimizer arithmetic, not whether the protocol improves held-out quality.
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

ROOT=Path(__file__).resolve().parents[3]
import torch
from experiments.ch3_motivation.code import calibration_sampling, probes
from experiments.ch3_motivation.code.common import capture_samples, load_models, restore_pack, write_json
from experiments.ch3_motivation.code.frozen_predecessor_fitting import fit_probe_new_term, protocol_binding
from experiments.ch3_motivation.code.probe_variants import load_variant
from experiments.ch3_motivation.code.check_resident_probe_fitting import (
    compare_reports, compare_states, attach_gradient_trace, compare_gradients,
)
from hstu_kvcache.common.artifacts import sha256


def run(args):
    output=args.output.resolve()
    if (output/'summary.json').exists():raise ValueError('use a new diagnostic attempt')
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    torch.cuda.set_device(0)
    device=torch.device('cuda:0')
    free,total=torch.cuda.mem_get_info(device)
    if free<32*1024**3:raise ValueError('need measured32GiB free for the additional bounded diagnostic')
    base=ROOT/'experiments/ch3_motivation/runs/e5/calibration_followup/attempt_1/reference/edge_2/seed_17'
    identity=json.loads((base/'configuration.json').read_text())
    source_hashes={str(p):sha256(p) for p in (Path(__file__),Path(probes.__file__),
        ROOT/'experiments/ch3_motivation/code/frozen_predecessor_fitting.py',ROOT/'experiments/ch3_motivation/code/check_resident_probe_fitting.py')}
    input_root=ROOT/'experiments/ch3_motivation/inputs_feedback_pool/fit256'
    directory=input_root/'edge_2'
    binding_path=directory/'inputs.json'; population_path=input_root/'population.json'
    binding=json.loads(binding_path.read_text());population=json.loads(population_path.read_text())
    if sha256(binding_path)!=identity['input_binding_sha256'] or sha256(population_path)!=identity['population_sha256']:
        raise ValueError('reference calibration input/population changed')
    for kind in ('history','requests'):
        if sha256(directory/binding[kind]['path'])!=binding[kind]['sha256']:raise ValueError('input changed')
    views,provenance=calibration_sampling.load_sampling_manifest(Path(identity['sampling_manifest']),input_root,2,.75,'gap_mix50',fit_users=256)
    if sha256(Path(identity['sampling_manifest']))!=identity['sampling_manifest_sha256']:raise ValueError('sampling changed')
    fit=population['fit'][:8];validation=population['validation'][:2]
    if set(fit)&set(validation):raise ValueError('roles overlap')
    parent,model,models=load_models(2,device)
    if models!=identity['models']:raise ValueError('reference models differ')
    cfg=dict(binding['settings'],calibration_queries_per_user=16,seed=17)
    histories=restore_pack(directory/binding['history']['path'])
    raw=capture_samples(parent,model,histories,fit+validation,cfg,device)
    for uid,sample in raw.items():sample['uid']=int(uid)
    del parent,histories
    # Explicit diagnostic subset; candidate/time values still come from the
    # original fixed256 source. Never label these ten captures as fit256.
    subset_provenance=dict(provenance,fit_uids=fit,validation_uids=validation)
    sampled=calibration_sampling.apply_sampling_view(raw,
        {uid:views[uid] for uid in fit+validation},subset_provenance)
    train=probes.prepare_training_samples(model,[sampled[uid] for uid in fit])
    val=probes.prepare_training_samples(model,[sampled[uid] for uid in validation])
    results=[]
    for stage,predecessor,term in (('fg','f','g'),('fgh','fg','h')):
        path=base/f'{predecessor}.pt';source_hashes[str(path)]=sha256(path)
        previous,metadata=load_variant(path)
        if metadata['identity']!=identity:raise ValueError('checkpoint identity changed')
        initial=previous.expanded(stage)
        del previous
        reports={};states={};gradients={}
        for mode in ('manual','helper'):
            probe=copy.deepcopy(initial).to(device)
            original={name:value.detach().cpu().clone() for name,value in probe.named_parameters() if not name.startswith(term+'.')}
            if mode=='manual':
                for name,value in probe.named_parameters():value.requires_grad_(name.startswith(term+'.'))
            trace,handles=attach_gradient_trace(probe)
            kwargs=dict(steps=4,validation_every=1,patience=0,learning_rate=3e-5,seed=17,weight_decay=.0001,gradient_clip=5.)
            fitter=probes.fit_probe if mode=='manual' else fit_probe_new_term
            report=fitter(model,probe,train,val,**kwargs)
            for handle in handles:handle.remove()
            if mode=='helper':report.pop('training_protocol')
            for name,value in probe.named_parameters():
                if not name.startswith(term+'.'):
                    if not torch.equal(original[name],value.detach().cpu()) or value.grad is not None:
                        raise ValueError('frozen predecessor changed or received gradient')
                    if trace.get(name):raise ValueError('frozen predecessor backward hook fired')
            gradients[mode]={name:values for name,values in trace.items() if name.startswith(term+'.')}
            states[mode]={name:value.detach().cpu().clone() for name,value in probe.state_dict().items()}
            reports[mode]=report
            del probe
        numeric=dict(stage=stage,predecessor=str(path),frozen_predecessors_exact=True,
            gradients=compare_gradients(gradients['manual'],gradients['helper'],4,rtol=0.,atol=0.),
            state=compare_states(states['manual'],states['helper'],rtol=0.,atol=0.),
            report=compare_reports(reports['manual'],reports['helper'],rtol=0.,atol=0.))
        if not all(numeric[key]['passed'] for key in ('gradients','state','report')):raise ValueError('manual/helper numerical mismatch')
        results.append(numeric)
        write_json(output/f'{stage}.json',numeric)
    for path,digest in source_hashes.items():
        if sha256(Path(path))!=digest:raise ValueError('diagnostic source/checkpoint changed')
    write_json(output/'summary.json',dict(status='passed_actual_large_numerical_check',fit_users=8,validation_users=2,
        diagnostic_only=True,quality_auc_computed=False,quality_use_permitted=False,formal_weights_written=False,
        stages=[r['stage'] for r in results],protocol=protocol_binding(),models=models,verified_sha256=source_hashes,
        initial_free_bytes=free,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        scope='four optimizer steps per stage versus manually frozen original fitter; same actual Large checkpoint and causal Q16 inputs; shared GPU runtime is not controlled timing; no population quality claim'))
    print(json.dumps(dict(status='passed_actual_large_numerical_check',output=str(output))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,choices=(0,1),required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
