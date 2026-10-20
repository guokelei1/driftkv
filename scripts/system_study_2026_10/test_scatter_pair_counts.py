"""Closed-form instrumentation must preserve the original dense mask ledger."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src")]
from system_study_2026_10.batched_session import append_pair_counts


def test_all_production_lengths_widths_and_code_prefixes():
    maximum=1024
    # All historical lengths and all code-prefix lengths, both supported
    # causal diagonals, and every append-band width used by this executor.
    for diagonal in (-1,0):
        for n in range(maximum+1):
            for width in (1,2,4,8,16,32):
                positions=np.arange(n,n+width)
                keys=np.arange(n+width)
                keep=(keys[None]<=positions[:,None]+diagonal)&(keys[None]>positions[:,None]-maximum)
                native=int(keep.sum())
                prefix=np.concatenate(([0],keep.sum(axis=0).cumsum()))
                for coded in range(n+1):
                    assert append_pair_counts(n,width,coded,maximum,diagonal)==(native,int(prefix[coded]))


def test_retained_real_append_geometry():
    path=ROOT/"results/system_study_2026_10/service/batched_session32_graph_full/attempt1/relay_calls.json"
    calls=json.loads(path.read_text())
    for row in calls:
        if row["operation"]!="append":
            continue
        n,width=row["history_length"],row["append_width"]
        positions=np.arange(n,n+width)
        keys=np.arange(n+width)
        keep=(keys[None]<=positions[:,None]-1)&(keys[None]>positions[:,None]-1024)
        for coded in row["session_coded_rows"]:
            assert append_pair_counts(n,width,coded,1024,-1)==(int(keep.sum()),int(keep[:,:coded].sum()))
