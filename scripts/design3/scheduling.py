"""Small fixed ready-work experiment, not a production-arrival tail-latency claim."""
import json,time,statistics
from types import SimpleNamespace
import torch
from design.data import ROOT,histories,prefix,frozen_model,quality_requests
from hstu_kvcache.design2.rebuild import current_kv
from hstu_kvcache.design3.renewal import rebuild_steps,finish
from hstu_kvcache.design3.execution import Slot
from hstu_kvcache.adaptation.reader import score


@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.cuda.set_device('cuda:3');torch.backends.cuda.matmul.allow_tf32=False
    model=frozen_model(5,'cuda:3');uid=560030
    users=json.loads((ROOT/'results/design3/initial_01/uids.json').read_text())['original']
    h=histories(users,301)
    items,actions,deltas,_=prefix(h,uid,287*86400,'cuda:3')
    data=torch.load(ROOT/'results/design3/initial_01/inputs/m5_00.pt',map_location='cuda:3',weights_only=True)
    p=data['prepared'];ready=[]
    requests=quality_requests(users,287*86400,301*86400)
    for other_uid in users:
        if other_uid==uid or not requests.get(other_uid):continue
        first=min(int(r['query_timestamp']) for r in requests[other_uid]);rows=[r for r in requests[other_uid] if int(r['query_timestamp'])==first]
        i,a,delta,ts=prefix(h,other_uid,first,'cuda:3');cache=current_kv(model,i,a,delta)
        panel=(torch.tensor([[r['item_idx'] for r in rows]],device='cuda:3'),torch.tensor([first-int(ts[-1])],dtype=torch.float32,device='cuda:3'))
        slot=Slot(lambda ca,pa,pr:score(model,ca,*pa)[0],cache,panel,p)
        ready.append((other_uid,cache,panel,slot))
        if len(ready)==4:break
    assert len(ready)==4
    reference=current_kv(model,items,actions,deltas)
    candidate=finish(rebuild_steps(model,items,actions,deltas))
    torch.testing.assert_close(candidate.k,reference.k,rtol=0,atol=0);torch.testing.assert_close(candidate.v,reference.v,rtol=0,atol=0)
    records=[]
    for repeat in range(9):
        for policy in (['serial','cooperative'] if repeat%2==0 else ['cooperative','serial']):
            torch.cuda.synchronize();start=time.perf_counter();latencies=[];stages=[]
            def ordinary():
                _,cache,panel,slot=ready[len(latencies)]
                slot.run(cache,panel,p,True);torch.cuda.synchronize();latencies.append((time.perf_counter()-start)*1000)
            if policy=='serial':
                result=finish(rebuild_steps(model,items,actions,deltas));torch.cuda.synchronize();build_ms=(time.perf_counter()-start)*1000
                for _ in range(4):ordinary()
            else:
                # A renewal is admitted first; four other-UID reads become ready at first yield.
                # Submit at most one ordinary read after each bounded target work piece.
                def between(stage):
                    stages.append(stage)
                    if len(latencies)<4:ordinary()
                result=finish(rebuild_steps(model,items,actions,deltas),between);torch.cuda.synchronize();build_ms=(time.perf_counter()-start)*1000
            torch.cuda.synchronize();total=(time.perf_counter()-start)*1000
            torch.testing.assert_close(result.k,reference.k,rtol=0,atol=0);torch.testing.assert_close(result.v,reference.v,rtol=0,atol=0)
            records.append(dict(repeat=repeat,policy=policy,total_ms=total,target_completion_ms=build_ms,ordinary_completion_ms=latencies,yields=len(stages)))
    out=ROOT/'results/design3/initial_01/scheduling_distinct.json'
    out.write_text(json.dumps(dict(scope='one actual1024-event M5 rebuild and four distinct other-UID actual first-request native reads; reads given simultaneous readiness as a controlled execution workload, not their historical arrival schedule; same thread and GPU stream; sequential baseline uses same generator; synchronize completed reads; no production arrivals or SLO claim',uid=uid,n=items.shape[1],ordinary=[dict(uid=u,n=ca.seq_len,q=pa[0].numel()) for u,ca,pa,sl in ready],kv_bitwise=True,records=records),indent=2))
    print(out)


if __name__=='__main__':main()
