#!/usr/bin/env python3
"""Same-query affine read residual on top of the nonlinear history mapping."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from read_correction_v4.common import OUTPUT
from read_correction_v4.cost_nonlinear import correction_forward
from read_correction_v4.history_conditioned.fit_query_compensation import fit_query_compensation
from read_correction_v4.probe import run
from read_correction_v4.probe_nonlinear import fit as fit_nonlinear, nonlinear_configuration


def query_configuration():
    return {**nonlinear_configuration(), "revision": "v4_nonlinear_query_probe",
        "query_compensation": "sequential actual-query affine residual after frozen token mapping",
        "query_ridge": .01}


def fit(current, rows, train, validation, cfg, device, scale):
    base, layers, base_cost = fit_nonlinear(current, rows, train, validation, cfg, device, scale)
    modules, fitted = fit_query_compensation(current, rows, train, validation, base,
        cfg, device, scale, batch_size=8)
    ledger = {"nonlinear_and_cache": base_cost, "query_compensation": fitted["cost"],
        "calibration_flops": base_cost["calibration_flops"] + fitted["cost"]["calibration_flops"],
        "convention": "one shared cache capture, affine and nonlinear fit, then one sequential query residual fit"}
    return modules, {**layers, "query_compensation": fitted["layers"]}, ledger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("medium", "large", "max"), required=True)
    parser.add_argument("--edge", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT / "nonlinear_query_probe")
    run(parser.parse_args(), fitter=fit, config=query_configuration(),
        artifact_kind="token_read_nonlinear_v4", correction_cost=correction_forward)
