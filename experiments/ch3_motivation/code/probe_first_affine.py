"""Differentiable factorization of the ordinary concatenated h MLP.

Only the first affine is split: W[q,x]+b = Wq q + (Wx x+b).
All subsequent nonlinearities/layers and gradients remain in the same MLP;
there is no low-rank model or feature cache across optimizer steps.
"""
from contextlib import contextmanager
from types import MethodType

import torch


def interaction(module, q, rows):
    dim = q.shape[-1]
    weight = module.weights[0]
    query = torch.einsum("bhqi,hio->bhqo", q, weight[:, :dim])
    row = torch.einsum("bhni,hio->bhno", rows, weight[:, dim:]) + module.biases[0][None, :, None]
    value = query[:, :, :, None] + row[:, :, None]
    for index in range(1, len(module.weights)):
        value = torch.nn.functional.silu(value)
        value = torch.einsum("bh...i,hio->bh...o", value, module.weights[index])
        bias = module.biases[index]
        value = value + bias.reshape(1, bias.shape[0], *([1] * (value.ndim - 3)), -1)
    return value


def first_affine_flops(cfg, rows, queries, *, batch=1):
    n, q, b = int(rows), int(queries), int(batch)
    if n == 0 or q == 0:
        return dict(original=0, factorized=0, delta=0)
    if n > cfg.row_chunk or q > cfg.query_chunk:
        raise ValueError("training ledger is bounded to the actual single row/query chunks")
    lh, d, r, w = cfg.num_layers * cfg.num_heads, cfg.head_dim, cfg.row_dim, cfg.hidden_width
    original = b * lh * q * n * (2 * (d + r) * w + w)
    factorized = b * lh * (n * (2 * r * w + w) + q * 2 * d * w + q * n * w)
    return dict(original=int(original), factorized=int(factorized), delta=int(factorized - original))


def counted_first_affine(cfg, counts, phase):
    """Sum actual bounded sample shapes from the original fitter's counters."""
    rows = counts.get(phase + "_probe_rows", 0)
    queries = counts.get(phase + "_native_queries", 0)
    pairs = counts.get(phase + "_candidate_row_pairs", 0)
    lh, d, r, w = cfg.num_layers * cfg.num_heads, cfg.head_dim, cfg.row_dim, cfg.hidden_width
    original = lh * pairs * (2 * (d + r) * w + w)
    factorized = lh * (rows * (2 * r * w + w) + queries * 2 * d * w + pairs * w)
    return dict(original=int(original), factorized=int(factorized), delta=int(factorized - original))


@contextmanager
def factorized_reads(probe):
    """Original RMS query coordinates, with only h's first affine rewritten."""
    if probe.h is None:
        yield
        return
    if "interaction_correction" in probe.__dict__:
        raise ValueError("unexpected existing interaction override")

    def read(self, layer, q, rows):
        normalized = q / self.query_scale[layer][None, :, None] if self.cfg.normalize else q
        value = interaction(self.h[layer], normalized, rows)
        return value * self._output_scale(layer, value.ndim)

    probe.interaction_correction = MethodType(read, probe)
    try:
        yield
    finally:
        delattr(probe, "interaction_correction")


@contextmanager
def fitting_ledger(probe, *, enabled):
    if not enabled or probe.h is None:
        yield
        return
    if "forward_flops" in probe.__dict__:
        raise ValueError("unexpected existing probe arithmetic override")
    original = probe.forward_flops

    def forward(self, history_length, queries, *, batch=1):
        result = original(history_length, queries, batch=batch)
        delta = first_affine_flops(self.cfg, history_length, queries, batch=batch)["delta"]
        for name in ("h_read_extra_flops", "read_extra_flops", "total_correction_forward_flops"):
            result[name] += delta
        return result

    probe.forward_flops = MethodType(forward, probe)
    try:
        yield
    finally:
        delattr(probe, "forward_flops")
