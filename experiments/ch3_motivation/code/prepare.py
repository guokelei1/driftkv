#!/usr/bin/env python3
"""Prepare fresh fixed Chapter 3 inputs without loading any checkpoint."""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
from pathlib import Path
import sys

from experiments.ch3_motivation.code.common import ROOT, prepare_population, prepare_edge

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"experiments/ch3_motivation/inputs")
    parser.add_argument("--edges",type=int,nargs="+",default=[1,2,3,4,5])
    parser.add_argument("--threads",type=int,default=12)
    args=parser.parse_args()
    population=prepare_population(args.output,threads=args.threads)
    print({"status":"population_frozen","counts":population["counts"]},flush=True)
    for edge in args.edges:
        value=prepare_edge(args.output/f"edge_{edge}",population,edge,threads=args.threads)
        print({"status":"edge_inputs_ready","edge":edge,"users":value["packed_users"],
            "requests":value["feedback_events"],"seconds":value["seconds"]},flush=True)
