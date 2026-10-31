"""Run an unchanged E6 driver with the verified projected-SVD execution.

Original input/score identities remain intact. The additional execution
configuration is committed before any score block and must match on resume.
"""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code.common import write_json
from experiments.ch3_motivation.code.sketch_svd_execution import projected_svd, binding
from hstu_kvcache.common.artifacts import sha256


def run(args):
    readiness = json.loads(args.readiness.read_text())
    if (readiness['status'] != 'passed_actual_both_protocols_all8'
            or readiness['execution_binding'] != binding()):
        raise ValueError('both actual protocol proofs must match this SVD kernel')
    for record in readiness['records']:
        if sha256(Path(record['comparison'])) != record['comparison_sha256']:
            raise ValueError('actual all-rank numerical comparison changed')
    values = args.driver_arguments
    if values and values[0] == '--':
        values = values[1:]
    for flag, value in [('--method', 'randomized'), ('--oversample', '16'),
                        ('--power-iterations', '2'), ('--candidates', '230')]:
        if flag not in values or values[values.index(flag) + 1] != value:
            raise ValueError('formal decomposition differs from verified settings')
    output = Path(values[values.index('--output') + 1]).resolve()
    driver = ROOT / 'experiments/ch3_motivation/code' / (
        'run_lowrank_release.py' if args.protocol == 'release' else 'run_lowrank_timestamp_guarded.py')
    configuration = dict(format='verified_projected_svd_execution_v1', enabled=True,
        diagnostic_kernel_signature=binding(), readiness=str(args.readiness.resolve()),
        readiness_sha256=sha256(args.readiness), protocol=args.protocol,
        original_driver=str(driver), original_driver_sha256=sha256(driver),
        wrapper_source_sha256=sha256(Path(__file__)), driver_arguments=values,
        scope='same matrices/sketches/QR/ranks/guards/exact fallbacks; original score configuration plus this execution binding')
    path = output / 'execution_optimization_configuration.json'
    if path.exists() and json.loads(path.read_text()) != configuration:
        raise ValueError('SVD execution changed across committed score blocks')
    write_json(path, configuration)
    sys.argv = [str(driver), *values]
    with projected_svd() as counts:
        runpy.run_path(str(driver), run_name='__main__')
    if binding() != readiness['execution_binding']:
        raise ValueError('SVD execution source changed during inference')
    result_path = output / ('diagnostic.json' if '--diagnostic-users' in values else 'summary.json')
    result = json.loads(result_path.read_text())
    if result['status'] != 'complete':
        raise ValueError('original E6 driver did not finish')
    write_json(output / 'execution_optimization.json', dict(format=configuration['format'], enabled=True,
        execution_configuration_sha256=sha256(path), original_configuration_sha256=sha256(output / 'configuration.json'),
        original_result_sha256=sha256(result_path), diagnostic_kernel_signature=binding(),
        last_invocation_calls_by_shape=dict(counts),
        count_scope='this invocation only; resumed blocks keep the same precommitted execution configuration'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', choices=('release', 'timestamp'), required=True)
    parser.add_argument('--readiness', type=Path, required=True)
    parser.add_argument('driver_arguments', nargs=argparse.REMAINDER)
    run(parser.parse_args())
