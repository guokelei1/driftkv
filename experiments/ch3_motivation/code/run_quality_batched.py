"""Run the frozen quality sweep with verified batched append-input transfers."""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
import json
from pathlib import Path
import sys
from experiments.ch3_motivation.code import (
    batched_append_inputs as optimized, sweep_evaluate_probes as original,
)
from experiments.ch3_motivation.code.common import write_json
from hstu_kvcache.common.artifacts import sha256


def run(args):
    proof=json.loads(args.readiness.read_text())
    if (proof['status']!='passed_actual_fg_and_nonzero_fgh' or proof['execution_binding']!=optimized.binding()
        or json.loads((args.readiness.parent/'exit.json').read_text())['exit_code']!=0):
        raise ValueError('actual FG/H batched-input numerical readiness has not passed')
    config=dict(format='batched_append_quality_execution_v1',physical_gpu=args.gpu,
        plan=str(args.plan.resolve()),plan_sha256=sha256(args.plan),
        readiness=str(args.readiness.resolve()),readiness_sha256=sha256(args.readiness),
        execution_binding=optimized.binding(),wrapper_sha256=sha256(Path(__file__)))
    target=args.output/'execution_optimization_configuration.json'
    if target.exists():
        if json.loads(target.read_text())!=config:raise ValueError('quality execution helper/plan/readiness changed on resume')
    elif (args.output/'configuration.json').exists():
        raise ValueError('existing unbound sweep cannot silently adopt another execution helper')
    write_json(target,config)
    stats=dict(users=0,input_rows_prefetched=0,input_tensor_bytes_total=0,max_transient_user_input_bytes=0,host_to_device_array_transfers=0)
    method=optimized.BatchedAppendInputsTrajectory.report
    def audited_report(self):
        result=method(self)
        value=result['append_input_execution']
        stats['users']+=1
        stats['input_rows_prefetched']+=value['input_rows_prefetched']
        stats['input_tensor_bytes_total']+=value['input_tensor_bytes']
        stats['max_transient_user_input_bytes']=max(stats['max_transient_user_input_bytes'],value['input_tensor_bytes'])
        stats['host_to_device_array_transfers']+=value['host_to_device_array_transfers']
        return result
    try:
        optimized.BatchedAppendInputsTrajectory.report=audited_report
        with optimized.batched_append_inputs():original.main(args)
    finally:
        optimized.BatchedAppendInputsTrajectory.report=method
    if optimized.binding()!=config['execution_binding']:raise ValueError('execution source changed during quality')
    write_json(args.output/'execution_optimization.json',dict(enabled=True,configuration_sha256=sha256(target),
        original_configuration_sha256=sha256(args.output/'configuration.json'),original_summary_sha256=sha256(args.output/'summary.json'),
        execution_binding=config['execution_binding'],actual_input_transfer_counts=stats,
        counter_scope='this invocation only; resumed blocks retain the same committed execution configuration',
        memory_scope='extra transient quality-input buffers, including unconsumed window tail; add peak input bytes to other auxiliary memory, not to persistent method state',
        arithmetic_scope='model operations/causal event ledger/FLOPs unchanged; shared execution time is diagnostic'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,choices=[1],default=1)
    p.add_argument('--readiness',type=Path,required=True)
    run(p.parse_args())
