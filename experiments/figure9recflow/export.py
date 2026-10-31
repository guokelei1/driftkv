"""Export every completed edge/policy without outcome-based filtering."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

from .analysis import aggregate_edges
from .common import METRIC_NAMES, write_json
from .inputs import RecFlowInputs


def export(runs, output):
    inputs=RecFlowInputs()
    found={}
    for run in runs:
        run=Path(run)
        plan=json.loads((run/'plan.json').read_text())
        completed=json.loads((run/'summary.json').read_text())
        if completed['edge_count']!=len(plan['edges']):
            raise ValueError('Run is not completely aggregated')
        for edge in completed['edges']:
            name=edge['edge_id']
            if name in found: raise ValueError(f'Duplicate edge: {name}')
            found[name]=(edge,run,plan)
    names=[e.id for e in inputs.edges]
    if set(found)!=set(names): raise ValueError('Every actual accepted edge must be present')
    edge_summaries=[found[name][0] for name in names]
    combined=aggregate_edges(edge_summaries,expected_edges=names)
    combined['scope']='Bounded preparation benchmark; not full-population scientific results'
    combined['source_runs']=[str(Path(r).resolve()) for r in runs]
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'all_edges.json',combined)
    rows=[]
    for name in names:
        edge,run,plan=found[name]
        costs={p['tag']:p for p in edge['costs']}
        for result in edge['results']:
            row=dict(edge=name,evaluation_day=edge['evaluation_day'],tag=result['tag'],
                     method=result.get('method',''),population_users=plan['users'],
                     scope='bounded_benchmark',cost_percent=costs[result['tag']]['percent_of_once_rebuild'])
            for metric in METRIC_NAMES:
                value=result['metrics'][metric]
                for key in ('value','gain_over_reuse','recovery_percent','recovery_ci_status'):
                    row[f'{metric}_{key}']=value[key]
            rows.append(row)
    with (output/'all_203_points.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    write_json(output/'completion.json',dict(status='complete',edges=len(names),points=len(rows),
        formal_started=False,outcome_filtering=False,automatic_successors=False,
        source_runs=combined['source_runs']))
    return dict(edges=len(names),points=len(rows),output=str(output))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(export(args.runs,args.output)))
