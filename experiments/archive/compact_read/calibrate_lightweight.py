"""Shared normalization and local-read arithmetic; retired Value-only CLI removed."""
import torch
from torch.nn import functional as F


def feature_statistics(features, lengths):
    """Moments over real training rows only; validation and padding excluded."""
    mask = torch.arange(features.shape[1], device=features.device)[None] < lengths[:, None]
    count = mask.sum()
    masked = features * mask[:, :, None, None]
    center = masked.sum((0, 1)) / count
    variance = (masked.square().sum((0, 1)) / count - center.square()).clamp_min(0)
    return center, variance.sqrt().clamp_min(1e-5)


def cached_prediction(layer, normalized_features, normalized_weights):
    """Residual rate in training units; features and attention are constants."""
    encoded = F.silu(normalized_features.transpose(1, 2) @ layer.input_weight[None]
                     + layer.input_bias[None, :, None])
    return (normalized_weights @ encoded) @ layer.decoder[None]


def local_flops(*, batch, length, heads, dim, rank, queries, backward):
    """Executed dense local arithmetic; no gradients to cached inputs.

    multiply-add=2; SiLU forward=1, backward=5. MSE forward=3/output,
    backward=3/output. Adam including clipping is a declared 18/parameter
    arithmetic estimate (not measured GPU instructions).
    """
    b, n, h, d, r, q = batch, length, heads, dim, rank, queries
    encoder_mm = 2*b*h*n*(3*d)*r
    aggregate_mm = 2*b*h*q*n*r
    decoder_mm = 2*b*h*q*r*d
    encoded_elements, output_elements = b*h*n*r, b*h*q*d
    forward = encoder_mm + 2*encoded_elements + aggregate_mm + decoder_mm
    loss = 3*output_elements
    result = {"local_forward_flops": forward, "local_loss_flops": loss}
    if backward:
        parameters = h*(3*d*r+r+r*d)
        result.update(local_backward_flops=encoder_mm + aggregate_mm + 2*decoder_mm
                      + 6*encoded_elements + 3*output_elements,
                      optimizer_clip_flops=18*parameters + 10)
    return result
