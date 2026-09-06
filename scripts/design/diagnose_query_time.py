#!/usr/bin/env python3
"""Does a candidate-shared response remain shared across causal query times?

All probes use the identical prefix and candidate bank within its pre-release
idle interval. These are unlabeled diagnostic queries, not service outcomes.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))

from design.data import DAY, fixed_split, frozen_model, histories  # noqa: E402
from design.run import cache_at, new_state, new_translator, tensor, write_json  # noqa: E402
from insight.candidate_shared_causal import signed_head_intervention  # noqa: E402
from insight_two.common import load_frozen_inputs, metrics_row, score_metrics  # noqa: E402

from hstu_kvcache.adaptation.reader import score  # noqa: E402


@torch.no_grad()
def main(cli):
    out=ROOT/"results/design"/cli.run_id
    out.mkdir(parents=True,exist_ok=False)
    trained=ROOT/"results/design"/cli.reference
    args=SimpleNamespace(**json.loads((trained/"configuration.json").read_text()))
    uids=fixed_split()["development"][:128]
    write_json(out/"configuration.json",dict(target=5,users=uids,reference=str(trained.relative_to(ROOT)),
        reference_configuration_sha256=hashlib.sha256((trained/"configuration.json").read_bytes()).hexdigest(),
        translator_sha256=hashlib.sha256((trained/"translator_v5.pt").read_bytes()).hexdigest(),
        query_offsets=["cutover_gap","1 second","min(60,cutover_gap)","min(3600,cutover_gap)"],
        prefix="same strict-prior prefix; no events between last event and cutover",
        scope="unlabeled retrospective causal-time diagnostic; fixed/dynamic teacher interventions not executable actions",
        estimated_seconds=[25,80],confirmation_read=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    (out/"source.py").write_bytes(Path(__file__).read_bytes())
    start=time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device("cuda:0")
    h=histories(uids,288)
    parent,current=frozen_model(4,device),frozen_model(5,device)
    translator=new_translator(args,5,device)
    saved=torch.load(trained/"translator_v5.pt",map_location=device,weights_only=False)
    translator.load_state_dict(saved["state_dict"])
    translator.supported_producers=set(saved["supported_producers"])
    _,panels,_=load_frozen_inputs()
    rows,raw=[],[]
    for index,uid in enumerate(uids):
        cache,times=cache_at(parent,h,uid,287*DAY)
        exact,_=cache_at(current,h,uid,287*DAY)
        gap=287*DAY-int(times[-1])
        assert gap>=1
        state=new_state(cache,times,4,args)
        state.release(5,translator)
        candidates=tensor(panels[4,index,:16][None],device)
        at_release=signed_head_intervention(current,exact,cache,candidates,tensor([gap],device,floating=True),mode="shared_only")
        fixed=torch.stack(at_release.shared_components,1).flatten(2).to(device)
        for name,value in (("cutover",gap),("one_second",1),("one_minute",min(60,gap)),("one_hour",min(3600,gap))):
            delta=tensor([value],device,floating=True)
            reference=score(current,exact,candidates,delta)[0]
            reuse=score(current,cache,candidates,delta)[0]
            learned=state.score(current,candidates,delta)
            dynamic=signed_head_intervention(current,exact,cache,candidates,delta,mode="shared_only")
            held=score(current,cache,candidates,delta,response_delta=fixed)[0]
            if name=="cutover":
                torch.testing.assert_close(held,dynamic.scores,atol=2e-4,rtol=2e-5)
            wanted=torch.stack(dynamic.shared_components,1).flatten(2).to(device)
            mismatch=(fixed-wanted).square().sum((0,2))/wanted.square().sum((0,2)).clamp_min(1e-20)
            outputs=dict(learned=learned,fixed_cutover_oracle=held,dynamic_query_oracle=dynamic.scores)
            for method,logits in outputs.items():
                rows.append(dict(uid=uid,query=name,delta_seconds=value,method=method,
                    **metrics_row(score_metrics(reference,reuse,logits))))
            raw.append(dict(uid=uid,query=name,delta_seconds=value,last_timestamp=int(times[-1]),
                exact=reference.cpu().tolist(),reuse=reuse.cpu().tolist(),
                outputs={k:v.cpu().tolist() for k,v in outputs.items()},
                fixed_response_relative_error_by_layer=mismatch.cpu().tolist()))
    write_json(out/"raw.json",raw)
    summary=[]
    for query in ("cutover","one_second","one_minute","one_hour"):
        for method in ("learned","fixed_cutover_oracle","dynamic_query_oracle"):
            group=[r for r in rows if r["query"]==query and r["method"]==method]
            summary.append(dict(query=query,method=method,users=len(group),
                mean_user_recovery=float(np.mean([r["probability_gap_recovery"] for r in group])),
                mean_probability_gap=float(np.mean([r["observed_probability_gap"] for r in group]))))
    write_json(out/"summary.json",dict(status="diagnostic_complete",rows=summary,
        elapsed_seconds=time.perf_counter()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/(1<<20)))
    print((out/"summary.json").read_text())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--reference",required=True)
    main(parser.parse_args())
