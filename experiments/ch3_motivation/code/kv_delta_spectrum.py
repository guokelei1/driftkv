#!/usr/bin/env python3
"""Fresh native-KV and version-delta spectra, pooled within each layer/head.

All residents/rows are retained. Exact-length native prefills are batched;
short prefixes retain the canonical Torch backend and full1024 uses Triton.
Streaming FP64 Gram matrices give squared singular values without retaining
the population's rows. K/V and all100 layer/head pairs stay separate.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
import numpy as np
import torch
from experiments.ch3_motivation.code.common import load_models, restore_pack, write_json
from hstu_kvcache.evaluation.trajectory import NativeTrajectory, _short_prefix_backend
from hstu_kvcache.models import HSTUKVCache
from hstu_kvcache.common.artifacts import sha256

INPUT=ROOT/'experiments/ch3_motivation/inputs'
OLD=ROOT/'experiments/ch3_motivation/runs/candidate_robustness/attempt_1'
NAMES=('source_K','source_V','rebuilt_K','rebuilt_V','delta_K','delta_V')
KINDICES={'k':(0,2,4),'v':(1,3,5)}
ORIGINAL_NUMERIC_SOURCE_SHA='adb0c96209dbc12d794b4f181ca8f7690b0513ea4c99005a81f9f146b750c3df'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def head_matrix(value,heads):
    """[L,B,N,H*D] -> [L*H,B*N,D]; only rows/users are pooled."""
    layers,batch,rows,width=value.shape
    if width%heads:raise ValueError('Head width does not divide native width')
    return value.reshape(layers,batch,rows,heads,width//heads).permute(0,3,1,2,4).reshape(layers*heads,batch*rows,width//heads)


class Moments:
    def __init__(self,pairs,width,device):
        self.gram=torch.zeros(6,pairs,width,width,dtype=torch.float64,device=device)
        self.total=torch.zeros(6,pairs,width,dtype=torch.float64,device=device)
        self.rows=0

    @torch.no_grad()
    def add(self,source,rebuilt,heads,chunk_rows=2048):
        if source.k.shape!=rebuilt.k.shape or source.v.shape!=rebuilt.v.shape or source.seq_len!=rebuilt.seq_len:
            raise ValueError('Native pair row geometry differs')
        rows=source.k.shape[1]*source.seq_len
        for field,indices in KINDICES.items():
            old,new=head_matrix(getattr(source,field),heads),head_matrix(getattr(rebuilt,field),heads)
            for start in range(0,rows,chunk_rows):
                a=old[:,start:start+chunk_rows].double();b=new[:,start:start+chunk_rows].double()
                # Subtract after FP64 promotion: exact difference of the stored
                # FP32 native values, not a prior FP32 subtraction round trip.
                for index,x in zip(indices,(a,b,b-a),strict=True):
                    self.gram[index].add_(torch.bmm(x.transpose(1,2),x))
                    self.total[index].add_(x.sum(1))
        self.rows+=rows

    def state(self):
        return dict(gram=self.gram.cpu(),total=self.total.cpu(),rows=self.rows)

    def restore(self,state):
        self.gram.copy_(state['gram']);self.total.copy_(state['total']);self.rows=int(state['rows'])


def spectrum(gram,total,rows,centered=False):
    """All32 squared singular values; global row centering within each head."""
    work=gram.double()
    if centered and rows:work=work-total.double()[...,None]*total.double()[...,None,:]/rows
    work=(work+work.transpose(-2,-1))/2
    eigen=torch.linalg.eigvalsh(work).flip(-1)
    scale=gram.diagonal(dim1=-2,dim2=-1).sum(-1).abs().clamp_min(1.)
    if bool((eigen.amin(-1)<-1e-10*scale).any()):raise ValueError('Gram has negative eigenvalues beyond FP64 roundoff tolerance')
    minimum=float(eigen.min());eigen=eigen.clamp_min(0.)
    energy=eigen.sum(-1);valid=energy>0
    curve=torch.full_like(eigen,float('nan'))
    curve[valid]=eigen[valid].cumsum(-1)/energy[valid,None]
    ranks={}
    for threshold in (.8,.9,.95):
        rank=torch.zeros_like(energy,dtype=torch.long)
        rank[valid]=(curve[valid]<threshold).sum(-1)+1
        ranks[str(threshold)]=rank
    return dict(eigenvalues=eigen,energy=energy,curve=curve,valid=valid,ranks=ranks,
        minimum_unclipped_eigenvalue=minimum)


def native_batches(histories,users,cutoff,batch_users):
    groups=defaultdict(list);metadata={}
    for uid in users:
        events=histories.rows[uid];end=int(np.searchsorted(events[0],cutoff,side='left'));length=min(1024,end)
        groups[length].append(uid);metadata[str(uid)]=dict(history_rows=length,last_cached_timestamp=int(events[0][end-1]) if end else None)
    batches=[]
    for length,members in sorted(groups.items(),reverse=True):
        for start in range(0,len(members),batch_users):batches.append((length,members[start:start+batch_users]))
    return groups,batches,metadata


def batch_inputs(histories,uids,cutoff,device):
    times=[];items=[];behaviors=[];gaps=[]
    for uid in uids:
        events=histories.rows[uid];end=int(np.searchsorted(events[0],cutoff,side='left'));start=max(0,end-1024)
        t=events[0][start:end];delta=np.zeros(len(t),dtype=np.float32)
        if len(t)>1:delta[1:]=np.diff(t)
        times.append(t);items.append(events[1][start:end]);behaviors.append(events[2][start:end]);gaps.append(delta)
    if len({len(t) for t in times})!=1:raise ValueError('Only exact-length prefills are batched')
    return tuple(torch.as_tensor(np.stack(values),device=device,dtype=dtype)
        for values,dtype in ((items,torch.long),(behaviors,torch.long),(gaps,torch.float32)))


@torch.no_grad()
def prefill(model,inputs):
    length=inputs[0].shape[1]
    if not length:
        shape=(len(model.blocks),inputs[0].shape[0],0,model.blocks[0].attn.inner)
        reference=next(model.parameters())
        return HSTUKVCache(reference.new_empty(shape),reference.new_empty(shape),0)
    with _short_prefix_backend(model,length,1024):return model.compute_kv(*inputs)


def source_hashes():
    files=[Path(__file__),Path(__file__).with_name('common.py'),ROOT/'src/hstu_kvcache/evaluation/yambda.py',
        ROOT/'src/hstu_kvcache/evaluation/trajectory.py',ROOT/'src/hstu_kvcache/models/hstu.py',
        ROOT/'src/hstu_kvcache/models/attention.py',ROOT/'src/hstu_kvcache/models/state_transition.py',
        ROOT/'src/hstu_kvcache/models/embeddings.py',ROOT/'src/hstu_kvcache/models/triton_attention.py']
    return {str(p.resolve()):sha256(p) for p in files}


def check_sources(sources):
    if any(sha256(Path(p))!=h for p,h in sources.items()):raise ValueError('Frozen numerical source changed')


def bind(args):
    pop=json.loads(args.population.read_text());users=pop['users'];roles=json.loads((INPUT/'population.json').read_text())
    reference_path=OLD/f'quality/edge_{args.edge}/configuration.json';reference=json.loads(reference_path.read_text())
    if (len(users)!=len(set(users)) or len(users)<7000 or not set(reference['users'])<=set(users)
            or set(users)&set(roles['fit']+roles['validation']+roles['development']+roles['confirmation'])
            or pop['original_roles_modified'] or pop['score_values_read']):
        raise ValueError('Frozen original7000 or its authorized expanded development union required')
    entry=pop['edges'][str(args.edge)];refs=entry['inputs']
    for ref in refs.values():
        if sha256(Path(ref['path']))!=ref['sha256']:raise ValueError('Frozen edge population input changed')
    bp=Path(refs['binding']['path']);binding=json.loads(bp.read_text());history=bp.parent/binding['history']['path']
    if (sha256(bp)!=reference['input_binding_sha256'] or binding['population_sha256']!=sha256(INPUT/'population.json')
            or sha256(history)!=binding['history']['sha256'] or binding['settings']['history_length']!=1024
            or binding['settings']['release_day']!=reference['settings']['release_day']):
        raise ValueError('Must use the original candidate-study native models/history/cutoff')
    metadata=dict(edge=args.edge,population=str(args.population.resolve()),population_sha256=sha256(args.population),
        population_users=users,input_binding=str(bp.resolve()),input_binding_sha256=sha256(bp),
        history=str(history.resolve()),history_sha256=sha256(history),models=reference['models'],
        reference_configuration=str(reference_path.resolve()),reference_configuration_sha256=sha256(reference_path),
        original7000_population_sha256=sha256(OLD/'population.json'),
        settings=binding['settings'],cutoff=binding['settings']['release_day']*86400,
        layers=10,heads=10,head_width=32,quantities=list(NAMES),batch_users=args.batch_users,chunk_rows=args.chunk_rows,
        accumulation='FP64 per-head pooled-row Gram and sum; delta computed after FP64 promotion',
        weighting='All retained rows of every fixed resident; pooled rows, not equal UID weights',
        centering='Uncentered primary; centered sensitivity subtracts the single pooled-row mean per layer/head',
        head_summary='Every layer/head retained; median/q10/q90 across heads are descriptive heterogeneity, not confidence intervals',
        ranks=list(range(1,33)),highlight_ranks=[4,6,8],energy_thresholds=[.8,.9,.95],
        labels_or_candidates_used=False,feedback_user_filter=False,training=False,paper_changed=False,
        sources=source_hashes(),physical_gpu=1,diagnostic=args.diagnostic)
    if args.plan:metadata.update(plan=str(args.plan.resolve()),plan_sha256=sha256(args.plan))
    metadata['numeric_precision_evidence']=precision_evidence(args.precision_evidence,metadata)
    metadata['native_numeric_gate']=dict(per_head_relative_rms=1e-5,spectrum_curve_max_abs=1e-5,
        target_ranks_unchanged=True,original_absolute_allclose_is_disclosure_only=True,
        scope='Spectral study only: production FP32 native arithmetic is unchanged; absolute cache errors and original allclose failures remain recorded')
    return pop,metadata,history


def precision_evidence(path,config):
    proof=json.loads(path.read_text())
    if (proof.get('status')!='complete' or not proof.get('evidence_passed') or not proof.get('diagnostic_only')
            or proof.get('quality_use_permitted') or proof.get('original_source_sha256')!=ORIGINAL_NUMERIC_SOURCE_SHA
            or not proof.get('all_target_ranks_unchanged') or not proof['max_fp32_per_head_relative_rms']<1e-5
            or not proof['max_spectrum_curve_abs_difference']<1e-5 or not proof['max_fp64_abs']<1e-10):
        raise ValueError('Completed same-input FP64 and per-head spectral evidence required')
    for name in ('edge','population_sha256','input_binding_sha256','history_sha256','models','batch_users','cutoff','quantities','layers','heads','head_width'):
        if proof['configuration'][name]!=config[name]:raise ValueError('Precision evidence binding differs: '+name)
    for name in ('original_source','inspector_source'):
        if sha256(Path(proof[name]))!=proof[name+'_sha256']:raise ValueError('Archived precision evidence source changed')
    for source,digest in proof['configuration']['sources'].items():
        if Path(source).resolve()==Path(__file__).resolve():continue
        if sha256(Path(source))!=digest:raise ValueError('Native numerical source differs from FP64 evidence')
    return dict(path=str(path.resolve()),sha256=sha256(path),original_source_sha256=proof['original_source_sha256'],
        inspector_source_sha256=proof['inspector_source_sha256'],max_fp64_abs=proof['max_fp64_abs'],
        max_fp32_per_head_relative_rms=proof['max_fp32_per_head_relative_rms'],
        max_spectrum_curve_abs_difference=proof['max_spectrum_curve_abs_difference'])


def check_readiness(path,config):
    proof=json.loads(path.read_text())
    if proof['status']!='complete' or not proof['diagnostic'] or not proof['native_parity_passed'] or not proof['direct_svd_passed']:
        raise ValueError('Completed actual native batch and direct-SVD proof required')
    for name in ('edge','population_sha256','input_binding_sha256','history_sha256','models','batch_users','sources','numeric_precision_evidence','native_numeric_gate'):
        if proof['configuration'][name]!=config[name]:raise ValueError('Numerical readiness binding differs: '+name)


@torch.no_grad()
def native_spectral_check(source,rebuilt,source_single,rebuilt_single,heads):
    """Readiness for the measured spectra; disclose original cache allclose too."""
    pairs=source.k.shape[0]*heads;width=source.k.shape[-1]//heads
    actual=Moments(pairs,width,source.k.device);actual.add(source,rebuilt,heads)
    reference=Moments(pairs,width,source.k.device);reference.add(source_single,rebuilt_single,heads)
    per_head=[];curves=[]
    for field,indices in KINDICES.items():
        a=head_matrix(getattr(source,field),heads).double();b=head_matrix(getattr(rebuilt,field),heads).double()
        ar=head_matrix(getattr(source_single,field),heads).double();br=head_matrix(getattr(rebuilt_single,field),heads).double()
        for index,value,expected in zip(indices,(a,b,b-a),(ar,br,br-ar),strict=True):
            rms=expected.square().mean((1,2)).sqrt();difference=value-expected
            relative=difference.square().mean((1,2)).sqrt()/rms.clamp_min(1e-30)
            normalized_abs=difference.abs().amax((1,2))/rms.clamp_min(1e-30)
            if not bool((relative<1e-5).all()):raise ValueError('Native per-head relative RMS exceeds the spectral readiness limit')
            per_head.append(dict(quantity=NAMES[index],head_order='Layer-major, head-minor',reference_rms=rms.cpu().tolist(),
                relative_rms=relative.cpu().tolist(),max_abs_over_reference_rms=normalized_abs.cpu().tolist()))
    for centered in (False,True):
        a=spectrum(actual.gram,actual.total,actual.rows,centered)
        b=spectrum(reference.gram,reference.total,reference.rows,centered)
        if not bool(a['valid'].all() and b['valid'].all()):raise ValueError('Actual readiness case needs nonzero head spectra')
        for index,name in enumerate(NAMES):
            delta=(a['curve'][index]-b['curve'][index]).abs()
            changed={t:int((a['ranks'][t][index]!=b['ranks'][t][index]).sum()) for t in a['ranks']}
            if float(delta.max())>=1e-5 or any(changed.values()):raise ValueError('Native batch changes measured spectral curves/target ranks')
            curves.append(dict(quantity=name,centered=centered,max_curve_abs_difference=float(delta.max()),
                highlighted={str(r):float(delta[:,r-1].max()) for r in (4,6,8)},changed_target_rank_heads=changed))
    return dict(per_head=per_head,spectra=curves,passed=True)


@torch.no_grad()
def direct_svd_check(source,rebuilt,heads):
    reports=[];n=min(64,source.seq_len)
    source=HSTUKVCache(source.k[:,:1,:n],source.v[:,:1,:n],n)
    rebuilt=HSTUKVCache(rebuilt.k[:,:1,:n],rebuilt.v[:,:1,:n],n)
    moment=Moments(source.k.shape[0]*heads,source.k.shape[-1]//heads,source.k.device)
    moment.add(source,rebuilt,heads,chunk_rows=11)
    for field in ('k','v'):
        a=head_matrix(getattr(source,field),heads).double();b=head_matrix(getattr(rebuilt,field),heads).double()
        for index,name,x in zip(KINDICES[field],('source','rebuilt','delta'),(a,b,b-a),strict=True):
            gram=moment.gram[index];total=moment.total[index]
            for centered in (False,True):
                matrix=x-x.mean(1,keepdim=True) if centered else x
                sv=torch.linalg.svdvals(matrix).square()
                if sv.shape[1]<x.shape[-1]:sv=torch.nn.functional.pad(sv,(0,x.shape[-1]-sv.shape[1]))
                result=spectrum(gram,total,n,centered);actual=result['eigenvalues']
                scale=sv.amax(1,keepdim=True).clamp_min(1.)
                passed=bool(((actual-sv).abs()<=1e-10*scale+1e-10*sv.abs()).all())
                if not passed:raise ValueError('Actual input Gram spectrum differs from direct FP64 SVD')
                reports.append(dict(quantity=name+'_'+field.upper(),centered=centered,rows=n,
                    max_eigenvalue_abs_error=float((actual-sv).abs().max()),
                    max_error_over_reference_peak=float(((actual-sv).abs()/scale).max()),passed=passed))
    return reports


@torch.no_grad()
def diagnostic(args,config,histories,groups,parent,current):
    full=groups[1024][:args.batch_users]
    short_lengths=[n for n,uids in groups.items() if 0<n<1024 and len(uids)>=args.batch_users]
    if len(full)!=args.batch_users or not short_lengths:raise ValueError('Need natural full/short batches for readiness')
    short=max(short_lengths,key=lambda n:(len(groups[n]),-n))
    cases=[(1024,full),(short,groups[short][:args.batch_users])];records=[];svd=[];started=time.perf_counter()
    for length,uids in cases:
        inputs=batch_inputs(histories,uids,config['cutoff'],'cuda:0');torch.cuda.synchronize();tick=time.perf_counter()
        old=prefill(parent,inputs);new=prefill(current,inputs);torch.cuda.synchronize();seconds=time.perf_counter()-tick
        errors=[];singles=[];single_seconds=0.
        for version,model,batch in ((args.edge-1,parent,old),(args.edge,current,new)):
            references=[]
            for j,uid in enumerate(uids):
                torch.cuda.synchronize();single_started=time.perf_counter()
                reference=NativeTrajectory(model,version,histories.rows[uid],config['cutoff'],max_length=1024,append_band_size=128).cache
                torch.cuda.synchronize();single_seconds+=time.perf_counter()-single_started
                references.append(reference)
                for field in ('k','v'):
                    a=getattr(batch,field)[:,j:j+1];b=getattr(reference,field)
                    difference=(a-b).abs()
                    errors.append(dict(version=version,uid=int(uid),field=field,max_abs=float(difference.max()),
                        original_allclose_failed_elements=int((difference>1e-5+1e-5*b.abs()).sum()),elements=a.numel()))
            singles.append(HSTUKVCache(torch.cat([r.k for r in references],dim=1),torch.cat([r.v for r in references],dim=1),length))
        torch.cuda.synchronize();tick=time.perf_counter();parity=native_spectral_check(old,new,*singles,10)
        torch.cuda.synchronize();spectral_seconds=time.perf_counter()-tick;tick=time.perf_counter()
        svd.extend(direct_svd_check(old,new,10))
        torch.cuda.synchronize();svd_seconds=time.perf_counter()-tick
        records.append(dict(prefix_rows=length,uids=uids,batch_users=len(uids),batched_native_seconds=seconds,
            independent_native_prefill_seconds=single_seconds,spectral_check_seconds=spectral_seconds,direct_svd_check_seconds=svd_seconds,
            checks=errors,spectral_parity=parity))
    evidence=json.loads(Path(config['numeric_precision_evidence']['path']).read_text())
    if [(r['prefix_rows'],r['uids']) for r in records]!=[(r['prefix_rows'],r['uids']) for r in evidence['fp32_cases']]:
        raise ValueError('Actual readiness cases differ from the same-input FP64 diagnostic')
    check_sources(config['sources'])
    proof=dict(status='complete',diagnostic=True,quality_use_permitted=False,configuration=config,
        native_parity_passed=True,direct_svd_passed=True,native_gate=config['native_numeric_gate'],
        original_absolute_allclose=dict(atol=1e-5,rtol=1e-5,used_as_gate=False),
        native_cases=records,direct_svd=svd,seconds=time.perf_counter()-started,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),physical_gpu=1,
        scope='Two actual batch8 cases, full1024 Triton and natural exact-length short Torch. Numerical/performance evidence only.')
    write_json(args.output/'summary.json',proof);return proof


def write_spectra(moment,config,output):
    state=moment.state();results={};head_rows=[];curves=[]
    for centered in (False,True):
        mode='centered' if centered else 'uncentered';decomp=spectrum(state['gram'],state['total'],state['rows'],centered)
        entries={}
        for qi,name in enumerate(NAMES):
            curve=decomp['curve'][qi].numpy();valid=decomp['valid'][qi].numpy();energy=decomp['energy'][qi].numpy()
            ranks={t:x[qi].numpy() for t,x in decomp['ranks'].items()}
            for pair in range(100):
                item=dict(edge=config['edge'],quantity=name,centering=mode,layer=pair//10+1,head=pair%10+1,
                    pooled_rows=state['rows'],total_energy=float(energy[pair]),nonzero_energy=bool(valid[pair]),
                    **{'rank_'+t: int(x[pair]) if valid[pair] else None for t,x in ranks.items()},
                    **{f'energy_r{r}':float(curve[pair,r-1]) if valid[pair] else None for r in range(1,33)})
                head_rows.append(item)
            summaries={}
            for scope,mask in [('all100heads',np.ones(100,dtype=bool)),*[(f'layer_{l+1}',np.arange(100)//10==l) for l in range(10)]]:
                keep=mask&valid
                def quantiles(values):
                    return None if not keep.any() else dict(zip(('q10','median','q90'),np.quantile(values[keep],[.1,.5,.9],axis=0).tolist(),strict=True))
                summaries[scope]=dict(heads=int(mask.sum()),nonzero_energy_heads=int(keep.sum()),curve=quantiles(curve),
                    required_rank={t:quantiles(x) for t,x in ranks.items()})
                for r in range(1,33):
                    q=summaries[scope]['curve']
                    curves.append(dict(edge=config['edge'],quantity=name,centering=mode,scope=scope,rank=r,
                        q10=q['q10'][r-1] if q else None,median=q['median'][r-1] if q else None,q90=q['q90'][r-1] if q else None))
            entries[name]=summaries
        results[mode]=dict(quantities=entries,minimum_unclipped_eigenvalue=decomp['minimum_unclipped_eigenvalue'])
    for name,rows in [('per_head.csv',head_rows),('rank_curves.csv',curves)]:
        with (output/name).open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    torch.save(dict(format='kv_spectrum_final_moments_v1',configuration_sha256=digest(config),**state),output/'moments_final.pt')
    summary=dict(status='complete',diagnostic=False,configuration=config,pooled_rows_per_head=state['rows'],
        spectra=results,quality_auc_or_recovery_computed=False,
        limitations=['Pooled row SVD is oracle geometry, not a learned deployable projection or evidence of task-quality preservation.',
            'K/V and every layer/head are separate. No layer/head or user was selected using spectra.',
            'Centered sensitivity removes the global per-head pooled-row mean, not each user mean.',
            'Median/q10/q90 describe fixed-model head heterogeneity and are not confidence intervals.',
            'Concentrated delta energy cannot by itself establish correction-code dimension or justify skipping heads.'])
    write_json(output/'summary.json',summary)
    lines=[f'# Native and version-delta spectra: edge{config["edge"]}','','All fixed residents and all retained rows enter each layer/head Gram separately. K and V remain separate.','',
        '| Centering | Quantity | Median rank80 | Median rank90 | Median rank95 | Median energy r4 | r6 | r8 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for mode,entry in results.items():
        for name,scopes in entry['quantities'].items():
            q=scopes['all100heads'];rank=q['required_rank'];curve=q['curve']
            values=[rank[str(t)]['median'] if rank[str(t)] else None for t in (.8,.9,.95)]
            energy=[curve['median'][r-1] if curve else None for r in (4,6,8)]
            def fmt(x):return 'NA' if x is None else f'{x:.4f}'
            lines.append('| '+' | '.join([mode,name,*map(fmt,values),*map(fmt,energy)])+' |')
    lines+=['',*['- '+s for s in summary['limitations']]]
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return summary


def run(args):
    if args.gpu!=1:raise ValueError('Physical GPU1 only')
    pop,config,history=bind(args)
    if not args.diagnostic:
        if args.readiness is None:raise ValueError('--readiness actual batch/SVD proof required')
        check_readiness(args.readiness,config);config.update(readiness=str(args.readiness.resolve()),readiness_sha256=sha256(args.readiness))
    os.environ['CUDA_VISIBLE_DEVICES']='1';torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    parent,current,models=load_models(args.edge,torch.device('cuda:0'))
    if models!=config['models'] or current.cfg.num_heads!=10 or current.blocks[0].attn.head_dim!=32:
        raise ValueError('Native model/head geometry changed')
    histories=restore_pack(history);groups,batches,metadata=native_batches(histories,pop['users'],config['cutoff'],args.batch_users)
    args.output.mkdir(parents=True,exist_ok=True);cp=args.output/'configuration.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Frozen spectrum attempt changed')
    write_json(cp,config)
    if args.diagnostic:return diagnostic(args,config,histories,groups,parent,current)
    write_json(args.output/'row_manifest.json',dict(configuration_sha256=digest(config),users=metadata,
        exact_length_counts={str(n):len(uids) for n,uids in groups.items()},native_batches=len(batches)))
    moment=Moments(100,32,'cuda:0');checkpoint=args.output/'moments_running.pt';first=0;completed=0
    if checkpoint.exists():
        previous=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if previous['configuration_sha256']!=digest(config):raise ValueError('Running moments belong to a different frozen config')
        moment.restore(previous['moments']);first=previous['next_batch'];completed=previous['completed_users']
    started=time.perf_counter()
    for index in range(first,len(batches)):
        length,uids=batches[index];inputs=batch_inputs(histories,uids,config['cutoff'],'cuda:0')
        old=prefill(parent,inputs);new=prefill(current,inputs);moment.add(old,new,10,args.chunk_rows)
        completed+=len(uids);del old,new,inputs
        if (index+1)%32==0 or index+1==len(batches):
            check_sources(config['sources']);state=dict(configuration_sha256=digest(config),next_batch=index+1,
                completed_users=completed,moments=moment.state())
            temporary=checkpoint.with_suffix('.pt.writing');torch.save(state,temporary);temporary.replace(checkpoint)
            progress=dict(status='running',edge=args.edge,completed_users=completed,users=len(pop['users']),
                completed_batches=index+1,batches=len(batches),rows_per_head=moment.rows,seconds=time.perf_counter()-started)
            write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    if completed!=len(pop['users']):raise ValueError('Population coverage incomplete')
    result=write_spectra(moment,config,args.output);result.update(seconds=time.perf_counter()-started,
        resident_users=completed,peak_reserved_bytes=torch.cuda.max_memory_reserved(),physical_gpu=1)
    write_json(args.output/'summary.json',result);write_json(args.output/'progress.json',dict(progress,status='complete'))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--edge',type=int,choices=[2,3],required=True);parser.add_argument('--population',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--plan',type=Path)
    parser.add_argument('--diagnostic',action='store_true');parser.add_argument('--readiness',type=Path)
    parser.add_argument('--precision-evidence',type=Path,required=True,help='Bound same-input FP64/per-head spectrum diagnostic; does not change native FP32 arithmetic')
    parser.add_argument('--batch-users',type=int,choices=[8,16],default=8);parser.add_argument('--chunk-rows',type=int,default=2048)
    parser.add_argument('--gpu',type=int,default=1);args=parser.parse_args()
    try:run(args);write_json(args.output/'exit.json',dict(exit_code=0))
    except BaseException as error:write_json(args.output/'exit.json',dict(exit_code=1,error=str(error)));raise
