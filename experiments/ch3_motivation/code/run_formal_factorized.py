"""Run frozen formal jobs with the GPU-verified equivalent first affine."""

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

import torch
from experiments.ch3_motivation.code import (
    factorized_probe_fitting as optimized, sweep_probe_calibration_formal as sweep,
)
from experiments.ch3_motivation.code.common import write_json
from hstu_kvcache.common.artifacts import sha256


def run(args):
    plan = json.loads(args.plan.read_text())
    spec = plan["training_execution_optimization"]
    if spec["binding"] != optimized.binding():
        raise ValueError("formal optimization source binding changed")
    oracle = Path(spec["readiness_summary"])
    if sha256(oracle) != spec["readiness_summary_sha256"]:
        raise ValueError("actual GPU readiness evidence changed")
    report = json.loads(oracle.read_text())
    if (report["status"] != "passed_actual_large_factorized_fitting"
            or not report["gradients"]["passed"] or not report["final_state"]["passed"]
            or report["execution_binding"] != optimized.binding()
            or json.loads((oracle.parent / "exit.json").read_text())["exit_code"] != 0):
        raise ValueError("GPU readiness did not pass for this implementation")
    original_audit = sweep.quality.audit_weight

    def audit(path, *values, **kwargs):
        result = original_audit(path, *values, **kwargs)
        metadata = torch.load(path, map_location="cpu", weights_only=False)["metadata"]
        execution = metadata.get("execution_optimizations", {})
        if (execution.get("binding") != spec["binding"]
                or execution.get("deltas_already_applied_to_core_arithmetic") is not True
                or execution.get("original_training_objective") is not True):
            raise ValueError(f"missing or mismatched actual optimized fitting provenance: {path}")
        return result

    original_fit = sweep.resident_probe_fitting.fit_probe_resident
    try:
        sweep.quality.audit_weight = audit
        sweep.resident_probe_fitting.fit_probe_resident = optimized.fit_probe_factorized
        return sweep.main(args)
    finally:
        sweep.quality.audit_weight = original_audit
        sweep.resident_probe_fitting.fit_probe_resident = original_fit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=("main", "curve"), required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.resident_inputs = True
    ledger = args.output / args.family / "execution" / f"gpu_{args.gpu}"
    try:
        run(args)
        write_json(ledger / "exit.json", dict(exit_code=0))
    except BaseException as error:
        write_json(ledger / "exit.json", dict(exit_code=1, error=str(error)))
        raise
