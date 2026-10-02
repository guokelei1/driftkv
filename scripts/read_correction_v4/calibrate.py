#!/usr/bin/env python3
"""Fit the fixed v4 candidate, or reuse its matching retained calibration."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from read_correction_v4.common import PANEL_ROOT, edge_name, sha256
from read_correction_v4.probe import run
from read_correction_v4.probe_nonlinear import fit, nonlinear_configuration
from read_correction_v4.cost_nonlinear import correction_forward


def candidate(kind):
    if kind == "nonlinear":
        return fit, nonlinear_configuration()
    from read_correction_v4.probe_query import fit as fit_query, query_configuration
    return fit_query, query_configuration()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("medium", "large", "max"), required=True)
    parser.add_argument("--edge", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--kind", choices=("nonlinear", "nonlinear_query"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    fitter, cfg = candidate(args.kind)
    output = args.output_root / args.scale / edge_name(args.edge)
    record = output / "calibration.json"
    if record.exists():
        saved = json.loads(record.read_text())
        if (saved["status"] != "complete" or saved["settings"] != cfg
            or saved["panel_binding_sha256"] != sha256(PANEL_ROOT / args.scale / edge_name(args.edge) / "binding.json")
            or saved["weights_sha256"] != sha256(output / "calibration.pt")):
            raise RuntimeError("existing calibration settings, panel or weights differ")
        print(json.dumps({"status": "calibration_already_complete", "path": str(record)}), flush=True)
        return
    args.calibration_only = True
    run(args, fitter=fitter, config=cfg, artifact_kind="token_read_nonlinear_v4",
        correction_cost=correction_forward)


if __name__ == "__main__":
    main()
