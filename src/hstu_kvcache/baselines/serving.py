"""Small serving-path helpers shared by the four request-local baselines."""

import torch

from hstu_kvcache.baselines.layer_recompute.core import _evaluation


@torch.no_grad()
def score_one_query(current, cache, candidate_ids, query_time_deltas):
    """Score one transient candidate per row without copying the retained K/V.

    ``candidate_ids`` is [B,1]; the return value is [B,1]. The existing native
    append-only reader returns only the one query K/V row, which is discarded.
    Neither the input cache nor the serving trajectory is mutated.
    """
    if candidate_ids.ndim != 2 or candidate_ids.shape[1] != 1:
        raise ValueError("this evaluation helper scores one candidate per batch row")
    if candidate_ids.shape[0] != cache.k.shape[1]:
        raise ValueError("query batch and cache batch differ")
    with _evaluation(current):
        embedded = current.embed_query_tokens(candidate_ids, query_time_deltas)
        hidden, _ = current.forward_with_cache_embedded_new_kv(cache, embedded)
        return current.cc_score_head(hidden[:, 0]).reshape(candidate_ids.shape)
