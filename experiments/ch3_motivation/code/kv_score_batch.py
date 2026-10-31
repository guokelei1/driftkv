"""Evaluate independent same-length cache controls in one native batch."""
import torch
from hstu_kvcache.adaptation.reader import score
from hstu_kvcache.models import HSTUKVCache


@torch.no_grad()
def score_cache_group(model,caches,candidates,query_delta):
    """Return [methods,queries]; candidate/time input is identical per control."""
    if not caches or candidates.shape[0]!=1:
        raise ValueError('a nonempty single-user control group is required')
    n=caches[0].seq_len
    if any(c.seq_len!=n or c.k.shape[1]!=1 for c in caches):
        raise ValueError('controls must have aligned single-user cache lengths')
    packed=HSTUKVCache(torch.cat([c.k for c in caches],dim=1),
                       torch.cat([c.v for c in caches],dim=1),n)
    batch=len(caches)
    delta=query_delta.expand(batch,*query_delta.shape[1:])
    values,_=score(model,packed,candidates.expand(batch,-1),delta)
    return values
