"""Shared frozen-teacher rows and length batches; old decoder-only CLI retired."""
from collections import defaultdict
import torch
from hstu_kvcache.adaptation.reader import score
from read_correction_2026_09.cost import CostModel, eager_read


@torch.no_grad()
def prepare_rows(model,adapter,captured,uids,device,costs):
    """Resident frozen old caches/codes and one teacher logit per query."""
    rows={}
    cost_model=CostModel(model.cfg.hidden_size,len(model.blocks),model.cfg.num_heads,"torch")
    lookup_bytes=0
    for uid in uids:
        row=captured[uid]
        cache,teacher=row["parent"].to(device),row["teacher"].to(device)
        item=row["item_features"].to(device)
        ids=row["candidates"].to(device)[None]
        delta=torch.tensor([row["query_delta"]],device=device)
        wanted=score(model,teacher,ids,delta)[0].detach()
        state=adapter.encode_cache(cache,item)
        rows[uid]=dict(cache=cache,state=state,candidates=ids,query_delta=delta,
                       teacher_logits=wanted)
        costs["teacher_final_query_capture_flops"]+=eager_read(cost_model,teacher.seq_len,queries=ids.shape[1])
        costs["frozen_code_preparation_flops"]+=adapter.estimate_flops(tokens=cache.seq_len)["token_transform"]
        lookup_bytes+=item.numel()*item.element_size()
    return rows,lookup_bytes


def batches(uids,rows,batch_size,*,generator=None):
    order=list(uids)
    if generator is not None:
        order=[uids[i] for i in torch.randperm(len(uids),generator=generator).tolist()]
    lengths=defaultdict(list)
    for uid in order:
        lengths[rows[uid]["cache"].seq_len].append(uid)
    for members in lengths.values():
        for start in range(0,len(members),batch_size):
            yield members[start:start+batch_size]
