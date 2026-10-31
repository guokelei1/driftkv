#!/usr/bin/env python3
"""CPU arithmetic ledger for completed snapshot fits; no serving-cost claim."""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code.common import write_json
from experiments.ch3_motivation.code.state_lineage_probe import RowResidualMap
from hstu_kvcache.methods.baselines.protocol.cost import CostModel
from hstu_kvcache.common.artifacts import sha256
import pyarrow.parquet as pq

RUN=ROOT/'experiments/ch3_motivation/runs/lineage_candidate/attempt_1'


def main():
    population_path=RUN/'population.json';population=json.loads(population_path.read_text());results={};sources=[]
    dense,fused=CostModel(320,10,10,'torch'),CostModel(320,10,10,'triton')
    native=lambda n:(fused if n>=1024 else dense).full_cache(n)
    for target in (3,4):
        folder=RUN/f'state/final_{target}';sp=folder/'summary.json';summary=json.loads(sp.read_text())
        if summary['status']!='complete':raise ValueError('Completed fit required')
        identity=summary['identity'];model=RowResidualMap(100,64,identity['hidden'])
        per_version_native=sum(native(n) for n in summary['history_lengths'].values())
        sampled_rows=summary['sampled_fit_rows']+summary['sampled_validation_rows']
        maps={}
        def row(name,report):
            train=model.forward_flops(report['sampled_rows_per_step'])*report['actual_optimizer_steps']
            val=model.forward_flops(report['validation_rows'])*len(report['trace'])
            maps[name]=dict(parameters=report['parameter_count'],optimizer_steps=report['actual_optimizer_steps'],
                selected_step=report['selected_step'],training_forward_flops=train,validation_forward_flops=val,
                validation_calls=len(report['trace']),training_plus_backward_3x_estimate=3*train,
                training_plus_backward_and_validation_estimate=3*train+val)
        for version,stage in summary['stages'].items():
            for kind,report in stage.items():
                if int(version)==2 and kind!='chain':continue
                row(f'v{version}_{kind}',report)
        for kind,report in summary['short_stages'].items():row('short_v4_'+kind,report)
        endpoint={}
        for name,names in [('direct',[f'v{target}_direct']),('chain',[f'v{v}_chain' for v in range(2,target+1)])]:
            endpoint[name]=dict(map_fit_names=names,map_fits=len(names),total_optimizer_steps=sum(maps[n]['optimizer_steps'] for n in names),
                map_training_validation_estimate=sum(maps[n]['training_plus_backward_and_validation_estimate'] for n in names),
                native_capture_reference=per_version_native*(2 if name=='direct' else target),
                native_versions=[1,target] if name=='direct' else list(range(1,target+1)))
        entry=population['edges'][str(target)];request_path=Path(entry['inputs']['requests']['path'])
        observed=set(map(int,pq.read_table(request_path,columns=['uid'])['uid'].to_pylist()))
        feedback=[u for u in population['users'] if u in observed]
        lengths={u:int(entry['user_metadata'][str(u)]['history_count']) for u in population['users']}
        material={}
        for scope,users in [('all2048',population['users']),('observed_feedback',feedback)]:
            once=sum(model.forward_flops(lengths[u]) for u in users)
            material[scope]=dict(users=len(users),direct_once_map_flops=once,chain_all_maps_flops=once*(target-1),
                chain_over_direct=target-1,once_current_native_rebuild_flops=sum(native(lengths[u]) for u in users))
        results[target]=dict(target=target,fit_users=100,validation_users=16,map_geometry=dict(pairs=100,kv_width=64,
            input_width=128,hidden=identity['hidden'],output_width=64,parameters=next(iter(maps.values()))['parameters']),
            native_capture_flops=int(per_version_native*target),native_capture_versions=list(range(1,target+1)),
            native_cache_prefills=116*target,unique_completed_map_fits=len(maps),maps=maps,endpoint_recipe_references=endpoint,
            attempt_map_training_validation_estimate=sum(m['training_plus_backward_and_validation_estimate'] for m in maps.values()),
            actual_chain_sample_materialization_flops=model.forward_flops(sampled_rows)*(target-1),
            materialization=material,forward_flops_at_N1024=model.forward_flops(1024))
        for p in (sp,folder/'configuration.json',request_path):sources.append(dict(path=str(p.resolve()),sha256=sha256(p)))
    notes=['FLOPs are forward arithmetic from the frozen CostModel and RowResidualMap.forward_flops; this is not measured runtime or a complete serving ledger.',
        'Training counts use actual256 optimizer steps for every unique fitted map, even when the selected checkpoint is step224. Shared first-stage inputs use one exact shared fit.',
        'Backward uses an explicit approximate2× forward charge (training total3× forward); optimizer updates, fit normalization statistics, transfers and KV-MSE diagnostics are excluded.',
        'Native capture is the actual116-user snapshot reconstruction under each required model version. It includes original-source and teacher prefills; all versions see the same retained raw prefix.',
        'Direct and Chain have unequal total budgets: Direct uses one map fit and one materialization; Chain uses two/three. Per-map capacity,256 steps, fit/val UIDs and sampled rows are matched.',
        'The completed research attempt contains unused anchored, short and intermediate-direct fits from before the user narrowed scope. They are counted in the attempt total but are not endpoint quality methods.',
        'Endpoint native-capture references show required versions as a standalone recipe; captures are shared within the actual attempt and must not be double-counted as independently executed work.',
        'Materialization is once per fixed snapshot. Native candidate reads, source-cache preparation, repeated requests and all native appends are outside that incremental map column. No full-release serving-cost ratio is claimed.']
    paths=[population_path,Path(__file__),Path(__file__).with_name('state_lineage_probe.py'),ROOT/'src/hstu_kvcache/methods/baselines/protocol/cost.py']
    sources += [dict(path=str(p.resolve()),sha256=sha256(p)) for p in paths]
    payload=dict(format='state_lineage_cost_reference_v1',targets=results,notes=notes,sources=sources)
    output=RUN/'state/cost_reference.json';write_json(output,payload)
    lines=['# Snapshot lineage: arithmetic reference','','Native capture and fitted-map work are separated. Values below are teraFLOPs (10^12), not complete serving costs.','',
        '| Final target | Actual native capture | All completed unique fits | Direct map train+val estimate | Chain map train+val estimate | Direct materialization / observed users | Chain materialization / observed users |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for target,value in results.items():
        a=value['endpoint_recipe_references'];m=value['materialization']['observed_feedback']
        lines.append(f"| V{target} | {value['native_capture_flops']/1e12:.4f} | {value['unique_completed_map_fits']} | {a['direct']['map_training_validation_estimate']/1e12:.4f} | {a['chain']['map_training_validation_estimate']/1e12:.4f} | {m['direct_once_map_flops']/1e12:.4f} / {m['users']} | {m['chain_all_maps_flops']/1e12:.4f} / {m['users']} |")
    lines+=['','Every map has 1,241,600 parameters (100 independent layer/head pairs; 128→64→64 residual MLP). Its analytic N1024 forward is 2,627,993,600 FLOPs.','',*['- '+note for note in notes]]
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(status='complete',path=str(output),targets=results)))


if __name__=='__main__':main()
