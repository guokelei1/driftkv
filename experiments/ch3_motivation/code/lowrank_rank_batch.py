"""Equivalent parallel replay of all truncation ranks on one frozen snapshot.

The transient score batch indexes ranks. Every rank still computes its own
complete upper-layer query; only immutable cache/features and SVD factors are
broadcast. This never shares a truncated correction across changing times.
"""
import torch


def reconstruct_rank_batch(decomposition, ranks, *, candidates=slice(None), rows=slice(None)):
    """Return [ranks,heads,candidates,rows,KVdim] using shared compact factors."""
    obj = decomposition
    maximum = obj.max_rank
    bounded = [min(rank, maximum) for rank in ranks]
    if any(rank < 0 for rank in bounded):
        raise ValueError("truncation rank must be nonnegative")
    original = obj.original[:, candidates][:, :, rows]
    left = obj.left[:, candidates]
    shape = left.shape
    index = torch.arange(maximum, device=left.device)
    mask = index[None] < torch.as_tensor(bounded, device=left.device)[:, None]
    scaled = (left[None]*obj.singular_values[None, :, None, None]
              *mask[:, None, None, None])
    low = scaled.reshape(len(ranks), shape[0], shape[1]*shape[2], maximum) @ obj.right[None, :, :, rows]
    low = low.reshape(len(ranks), shape[0], shape[1], shape[2], original.shape[2]).transpose(-1, -2)
    low = low*obj.block_scales[None, :, None, None]
    main = obj.row_effect[:, :, rows]+obj.candidate_effect[:, candidates]-obj.grand_mean
    output = original[None].expand(len(ranks), -1, -1, -1, -1).clone()
    output[..., obj.selected_slice] = (main[None, ..., obj.selected_slice]+low).to(output.dtype)
    return output
