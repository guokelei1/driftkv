#!/usr/bin/env python3
"""Compare transient-query and real-write response targets at causal prefixes."""

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
from design.run import (  # noqa: E402
    cache_at,
    new_state,
    new_translator,
    response_targets,
    tensor,
    write_json,
)
from insight_two.common import load_frozen_inputs  # noqa: E402

from hstu_kvcache.adaptation.reader import read_embedded, score  # noqa: E402
from hstu_kvcache.models.state_transition import retain_latest_cache  # noqa: E402


@torch.no_grad()
def main():
    out=ROOT/"results/design/v2_write_response_probe_02"
    out.mkdir(parents=True,exist_ok=False)
    trained=ROOT/"results/design/v2_ridge_cal512_probe_01"
    args=SimpleNamespace(**json.loads((trained/"configuration.json").read_text()))
    uids=fixed_split()["development"][:128]
    cfg=dict(target=1,development_uids=uids,confirmation_read=False,labels_read=False,
        translator_reference=str(trained.relative_to(ROOT)),
        translator_sha256=hashlib.sha256((trained/"translator_v1.pt").read_bytes()).hexdigest(),
        selection="first true event in each user's final pre-day231 timestamp group; entire group excluded from prefix",
        scope="query-type response diagnostic; no fitting or method effectiveness claim",
        estimated_seconds=[25,50],estimate_basis="15s data + 6s models + 256 builds + two traced reads per user",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(out/"configuration.json",cfg)
    (out/"source.py").write_bytes(Path(__file__).read_bytes())
    start=time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device("cuda:0")
    history=histories(uids,232)
    _,panels,_=load_frozen_inputs()
    previous,current=frozen_model(0,device),frozen_model(1,device)
    mapper=new_translator(args,1,device)
    saved=torch.load(trained/"translator_v1.pt",map_location=device,weights_only=False)
    mapper.load_state_dict(saved["state_dict"])
    mapper.supported_producers=set(saved["supported_producers"])
    records=[]
    for index,uid in enumerate(uids):
        ts,items,actions=history.rows[uid]
        timestamp=int(ts[np.searchsorted(ts,231*DAY,side="left")-1])
        event_index=np.searchsorted(ts,timestamp,side="left")
        source,times=cache_at(previous,history,uid,timestamp)
        teacher,_=cache_at(current,history,uid,timestamp)
        if source.seq_len>=1024:
            source,teacher=retain_latest_cache(source,1023),retain_latest_cache(teacher,1023)
            times=times[-1023:]
        state=new_state(source,times,0,args)
        state.release(1,mapper)
        delta=tensor([min(7*DAY,timestamp-int(times[-1]))],device,floating=True)
        candidates=tensor(panels[0,index][None],device)
        _,read=score(current,source,candidates,delta,state.source,state.translated,trace=True)
        x=current.embed_inputs(tensor([[int(items[event_index])]],device),
            tensor([[int(actions[event_index])]],device),delta[:,None])
        write=read_embedded(current,source,x,state.source,state.translated,trace=True)
        native_write=read_embedded(current,source,x)
        exact_write=read_embedded(current,teacher,x)
        old=state.writer.old_mask(1)
        read_targets=response_targets(current,read,source,teacher,old)
        write_targets=response_targets(current,write,source,teacher,old)
        record=dict(uid=uid,timestamp=timestamp,action=int(actions[event_index]),prefix=source.seq_len,
            read_mse=[],write_mse=[],read_to_write_mse=[],write_kv_mse=[],native_write_kv_mse=[])
        for layer in range(6):
            rt,wt=read_targets[layer],write_targets[layer]
            record["read_mse"].append(float((read.corrections[layer]-rt).square().sum()/rt.square().sum().clamp_min(1e-20)))
            record["write_mse"].append(float((write.corrections[layer]-wt).square().sum()/wt.square().sum().clamp_min(1e-20)))
            record["read_to_write_mse"].append(float((rt.mean(2,keepdim=True)-wt).square().sum()/wt.square().sum().clamp_min(1e-20)))
            wanted=torch.stack((exact_write.new_kv.k[layer],exact_write.new_kv.v[layer]))
            got=torch.stack((write.new_kv.k[layer],write.new_kv.v[layer]))
            record["write_kv_mse"].append(float((got-wanted).square().sum()/wanted.square().sum().clamp_min(1e-20)))
            native=torch.stack((native_write.new_kv.k[layer],native_write.new_kv.v[layer]))
            record["native_write_kv_mse"].append(float((native-wanted).square().sum()/wanted.square().sum().clamp_min(1e-20)))
        records.append(record)
    write_json(out/"raw.json",records)
    write_json(out/"summary.json",dict(status="diagnostic_complete",users=len(records),
        mean_by_layer={key:np.mean([r[key] for r in records],axis=0).tolist()
            for key in ("read_mse","write_mse","read_to_write_mse","write_kv_mse","native_write_kv_mse")},
        elapsed_seconds=time.perf_counter()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/(1<<20),
        configuration_sha256=hashlib.sha256((out/"configuration.json").read_bytes()).hexdigest()))


if __name__=="__main__":
    main()
