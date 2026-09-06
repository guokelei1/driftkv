#!/usr/bin/env python3
"""Control the number of coupled-query refits using the stratified baseline."""

import argparse
import json
from types import SimpleNamespace

from design import run
from design.data import ROOT


def main(cli):
    config = json.loads((ROOT / "results/design/v9_stratified_stable64_01/configuration.json").read_text())
    args = SimpleNamespace(**config)
    args.run_id, args.steps, args.detached = cli.run_id, cli.passes, cli.detached
    args.source_confidence = args.second_moments = args.source_kernel = False
    args.prospective_resources = dict(expected_seconds=[50, 180], expected_gpu_mib=12000,
        conservative_executor_estimate_seconds=2694.7185,
        estimate_basis="same64/16 stratified fit44s plus approximately9s for three additional query refits; retain conservative executor guard",
        passing_canary="v9_stratified_calibration_canary_01",
        scope="controlled internal convergence comparison; unchanged inference")
    run.main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--passes", type=int, choices=(3, 6), default=6)
    parser.add_argument("--detached", action="store_true")
    main(parser.parse_args())
