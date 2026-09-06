#!/usr/bin/env python3
"""Compare a source-state kernel with the frozen stratified linear design."""

import argparse
import json
from types import SimpleNamespace

from design import run
from design.data import ROOT


def main(cli):
    config = json.loads((ROOT / "results/design/v9_stratified_stable64_01/configuration.json").read_text())
    args = SimpleNamespace(**config)
    args.run_id, args.fit_users, args.lifetime_users = cli.run_id, cli.users, cli.users//4
    args.targets = cli.targets
    args.source_confidence = args.second_moments = False
    args.source_kernel = True
    args.prospective_resources = dict(expected_seconds=[25, 120] if cli.users == 16 else [60, 240],
        expected_gpu_mib=16000, estimate_basis="stratified16/64 calibration24/44s plus centered-kernel solve and support lookup")
    run.main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--users", type=int, choices=(16, 64), default=64)
    parser.add_argument("--targets", type=int, choices=(1, 2, 3, 4, 5), default=5)
    main(parser.parse_args())
