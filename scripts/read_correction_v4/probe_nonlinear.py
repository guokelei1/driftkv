#!/usr/bin/env python3
"""Fixed four-edge nonlinear history-read development probe."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from read_correction_v4.common import OUTPUT, configuration
from read_correction_v4.cost_nonlinear import correction_forward
from read_correction_v4.history_conditioned.fit_nonlinear import fit_nonlinear
from read_correction_v4.probe import fit_modules, run


def nonlinear_configuration():
    return {**configuration(), "revision": "v4_nonlinear_probe",
        "candidate_mode": "uniform_known", "nonlinear_epochs": 6,
        "nonlinear_learning_rate": .001, "weight_decay": .0001,
        "nonlinear_width_rule": "model dimension / 2",
        "fit_objective": "normalized token KV error + same-actual-query history-read error",
        "selection": "minimum independent-user validation objective, including zero residual epoch0"}


def fit(current, rows, train, validation, cfg, device, scale):
    base, base_layers, base_cost = fit_modules(current, rows, train, validation, cfg, device, scale)
    modules, fitted = fit_nonlinear(current, rows, train, validation, base,
        config=cfg, device=device, scale=scale, batch_size=8)
    ledger = {"affine_and_cache": base_cost, "nonlinear": fitted["cost"],
        "calibration_flops": base_cost["calibration_flops"] + fitted["cost"]["calibration_flops"],
        "convention": "affine fitting and cache capture once; nonlinear incremental preparation, fitting and validation once"}
    return modules, {"affine": base_layers, "nonlinear": fitted["layers"]}, ledger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("medium", "large", "max"), required=True)
    parser.add_argument("--edge", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT / "nonlinear_probe")
    run(parser.parse_args(), fitter=fit, config=nonlinear_configuration(),
        artifact_kind="token_read_nonlinear_v4", correction_cost=correction_forward)
