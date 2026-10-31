"""Diagnostic alternative for the wide, small randomized-SVD projection.

This is not enabled in E6. A real-model rank-score oracle is required before
use: small singular values can lose accuracy in normal equations.
"""
import torch


@torch.no_grad()
def wide_svd(matrix):
    """Reduced SVD of [..., sketch_width, history], via a float64 Gram matrix."""
    if matrix.ndim < 2 or matrix.shape[-2] > matrix.shape[-1]:
        raise ValueError('requires a wide projected matrix')
    if matrix.dtype not in (torch.float32, torch.float64):
        raise ValueError('diagnostic supports FP32/FP64 projections')
    work = matrix.double()
    gram = work @ work.transpose(-1, -2)
    values, vectors = torch.linalg.eigh(gram)
    singular = values.flip(-1).clamp_min(0).sqrt()
    left = vectors.flip(-1)
    # Zero/roundoff singular directions need no reconstruction contribution.
    floor = singular[..., :1] * torch.finfo(torch.float64).eps * matrix.shape[-1]
    valid = singular > floor
    right = (left.transpose(-1, -2) @ work) / singular.clamp_min(torch.finfo(torch.float64).tiny)[..., :, None]
    right = torch.where(valid[..., :, None], right, torch.zeros_like(right))
    singular = torch.where(valid, singular, torch.zeros_like(singular))
    return left.to(matrix.dtype), singular.to(matrix.dtype), right.to(matrix.dtype)
