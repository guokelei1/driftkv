#!/usr/bin/env python3
"""Real six-layer comparison of batched sliding-mask replay and event replay."""

import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))

from design.data import DAY, diagnostic_admissions, frozen_model, histories  # noqa: E402
from design.run import (  # noqa: E402
    append_event,
    cache_at,
    event_range,
    new_state,
    new_translator,
    tensor,
    write_json,
)
from insight_two.common import load_frozen_inputs  # noqa: E402

from hstu_kvcache.models.state_transition import (  # noqa: E402
    append_with_rolling_band,
    append_with_rolling_cap,
)


@torch.no_grad()
def main():
    out=ROOT/"results/design/v3_rolling_band_canary_01"
    out.mkdir(parents=True,exist_ok=False)
    fitted=ROOT/"results/design/v3_query_only128_01"
    args=SimpleNamespace(**json.loads((fitted/"configuration.json").read_text()))
    cfg=dict(users=[1930,543930],forward_target=1,writer_target_probe=2,admissions=diagnostic_admissions(1),chunk_size=128,
        input_reference=str(fitted.relative_to(ROOT)),confirmation_read=False,
        scope="native rolling-band equivalence and cost canary; no quality result",
        estimate_seconds=[25,55],estimate_basis="15s IO + 2353 real events in native and functional reference + batched replay",
        tolerance=dict(kv_atol=2e-4,kv_rtol=2e-5,score_atol=2e-5,score_rtol=2e-5))
    files=sorted((ROOT/"src/hstu_kvcache/models").glob("*.py"))+sorted((ROOT/"src/hstu_kvcache/adaptation").glob("*.py"))+[Path(__file__)]
    cfg["source_sha256"]={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with tarfile.open(out/"source.tar.gz","w:gz") as archive:
        for p in files:
            archive.add(p,arcname=str(p.relative_to(ROOT)))
    write_json(out/"configuration.json",cfg)
    started=time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device("cuda:0")
    history=histories(cfg["users"],245)
    _,panels,_=load_frozen_inputs()
    parent,current=frozen_model(0,device),frozen_model(1,device)
    mapper=new_translator(args,1,device)
    saved=torch.load(fitted/"translator_v1.pt",map_location=device,weights_only=False)
    mapper.load_state_dict(saved["state_dict"])
    records=[]
    for index,uid in enumerate(cfg["users"]):
        cache,times=cache_at(parent,history,uid,231*DAY)
        events=event_range(history,uid,231*DAY,245*DAY)
        sequential,batched=new_state(cache,times,0,args),new_state(cache,times,0,args)
        sequential.release(1,mapper)
        batched.release(1,mapper)
        last=int(times[-1])
        items=tensor([[e[1] for e in events]],device)
        actions=tensor([[e[2] for e in events]],device)
        stamps=[e[0] for e in events]
        deltas=tensor([[min(7*DAY,b-a) for a,b in zip([last]+stamps[:-1],stamps,strict=True)]],device,floating=True)
        # The first sixteen writes cannot see later tokens in the same chunk.
        short=append_with_rolling_band(current,cache,items[:,:16],actions[:,:16],deltas[:,:16],1024)
        longer=append_with_rolling_band(current,cache,items[:,:128],actions[:,:128],deltas[:,:128],1024)
        torch.testing.assert_close(short.k[:,:,-16:],longer.k[:,:,1024-128:1024-128+16],atol=2e-4,rtol=2e-5)
        torch.cuda.synchronize()
        clock=time.perf_counter()
        native=append_with_rolling_cap(current,cache,items,actions,deltas,1024)
        torch.cuda.synchronize()
        native_seconds=time.perf_counter()-clock
        clock=time.perf_counter()
        for event in events:
            append_event(current,sequential,event,last)
            last=event[0]
        torch.cuda.synchronize()
        sequential_seconds=time.perf_counter()-clock
        clock=time.perf_counter()
        for offset in range(0,len(events),128):
            stop=min(len(events),offset+128)
            batched.append_native_chunk(current,items[:,offset:stop],actions[:,offset:stop],deltas[:,offset:stop],stamps[offset:stop])
        torch.cuda.synchronize()
        batch_seconds=time.perf_counter()-clock
        torch.testing.assert_close(batched.cache.k,native.k,atol=2e-4,rtol=2e-5)
        torch.testing.assert_close(batched.cache.v,native.v,atol=2e-4,rtol=2e-5)
        assert list(batched.writer.events)==list(sequential.writer.events)
        torch.testing.assert_close(batched.writer.pack(2).payload,sequential.writer.pack(2).payload,atol=2e-4,rtol=2e-5)
        candidates=tensor(panels[0,index][None],device)
        delta=tensor([245*DAY-last],device,floating=True)
        got=batched.score(current,candidates,delta)
        wanted=sequential.score(current,candidates,delta)
        torch.testing.assert_close(got,wanted,atol=2e-5,rtol=2e-5)
        records.append(dict(uid=uid,events=len(events),native_seconds=native_seconds,sequential_method_seconds=sequential_seconds,
            batch_method_seconds=batch_seconds,score_max_abs=float((got-wanted).abs().max()),
            kv_max_abs=max(float((batched.cache.k-native.k).abs().max()),float((batched.cache.v-native.v).abs().max()))))
    result=dict(passed=True,records=records,elapsed_seconds=time.perf_counter()-started,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/(1<<20),
        configuration_sha256=hashlib.sha256((out/"configuration.json").read_bytes()).hexdigest())
    write_json(out/"canary.pass.json",result)
    write_json(out/"summary.json",dict(status="canary_complete",**result))
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
