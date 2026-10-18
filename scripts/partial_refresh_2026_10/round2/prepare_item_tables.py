"""Freeze shared-item admission from fit histories strictly before each release."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src")]
from design_two.common import RELEASES
from design_two.prepare import load_history,load_panel,record
from unified_reuse_2026_09.common import write_json


def prepare(args):
    panel,groups,_=load_panel(args.panel)
    fit=groups["fit"][:256]
    history=load_history(args.panel,requested_uids=fit)
    arrays,versions={},{}
    for version,at in RELEASES:
        counts=Counter()
        for uid in fit:
            times,items,*_=history.rows[uid]
            stop=int(np.searchsorted(times,at,side="left"))
            counts.update(items[max(0,stop-1024):stop].tolist())
        ranked=np.array(sorted(counts,key=lambda x:(-counts[x],x)),dtype=np.int64)
        versions[version]=dict(timestamp=at,events=sum(counts.values()),unique_items=len(ranked),tables={})
        for k in args.sizes:
            ids=ranked[:k]
            arrays[f"{version}_{k}"]=ids
            versions[version]["tables"][str(k)]=dict(items=len(ids),shared_fp32_projection_bytes=len(ids)*1152,
                preparation_item_projection_flops=len(ids)*110880)
    args.output.mkdir(parents=True,exist_ok=False)
    np.savez(args.output/"item_ids.npz",**arrays)
    write_json(args.output/"summary.json",dict(status="frozen",role="fit-only prerelease shared-item admission",
        sizes=args.sizes,fit_uids=fit,by_release=versions,source_panel=record(args.panel/"binding.json"),
        item_ids=record(args.output/"item_ids.npz"),source_script=record(Path(__file__)),
        rule="counts on each fit user's latest1024 events strictly before release; descending count then item ID; no query/label input"))
    print(json.dumps(versions))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel",type=Path,default=ROOT/"results/design_two_2026_10/benchmark_panel")
    p.add_argument("--sizes",type=int,nargs="+",default=[4096,16384,32768,65536])
    p.add_argument("--output",type=Path,required=True)
    prepare(p.parse_args())
