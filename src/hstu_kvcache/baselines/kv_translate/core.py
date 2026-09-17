"""Closed-form, per-release translation of aligned HSTU K/V caches.

Inputs contain only valid persistent tokens, with every row aligned between
source and target. Data selection, user splits and release admission belong to
the caller. No model execution or cache reconstruction takes place here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...models.kv_cache import HSTUKVCache


@torch.no_grad()
def fit_affine_ridge(
    x: torch.Tensor, y: torch.Tensor, ridge: float = 0.01
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit ``y = x @ weight + bias`` with unnormalized centered Gram.

    Bias is unregularized. Solve in float64 to preserve the ridge term for
    correlated model features, then return the input x dtype. A zero penalty
    uses least squares, including rank-deficient CPU probe inputs.
    """
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or not x.shape[0]:
        raise ValueError("x and y must be aligned nonempty matrices")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    x64, y64 = x.to(torch.float64), y.to(torch.float64)
    mean_x, mean_y = x64.mean(0), y64.mean(0)
    xc, yc = x64 - mean_x, y64 - mean_y
    if ridge == 0:
        weight = torch.linalg.lstsq(xc, yc).solution
    else:
        gram = xc.T @ xc
        gram.diagonal().add_(ridge)
        weight = torch.linalg.solve(gram, xc.T @ yc)
    bias = mean_y - mean_x @ weight
    return weight.to(x.dtype), bias.to(x.dtype)


def _paired_rows(
    source: HSTUKVCache, target: HSTUKVCache
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    shape = source.k.shape
    if (
        len(shape) != 4
        or source.v.shape != shape
        or target.k.shape != shape
        or target.v.shape != shape
        or source.seq_len != shape[2]
        or target.seq_len != shape[2]
        or not shape[1] * shape[2]
    ):
        raise ValueError("paired caches must have the same nonempty [L,B,N,D] shape")
    # Flatten the same (batch, token) ordering on both sides. Padding is not
    # inferred from numeric values: the caller must supply only valid rows.
    return (
        (source.k.flatten(1, 2).to(torch.float64), target.k.flatten(1, 2).to(torch.float64)),
        (source.v.flatten(1, 2).to(torch.float64), target.v.flatten(1, 2).to(torch.float64)),
    )


@torch.no_grad()
def select_source_layers(
    source: HSTUKVCache,
    target: HSTUKVCache,
    num_heads: int,
    k: int,
    *,
    validation_source: HSTUKVCache | None = None,
    validation_target: HSTUKVCache | None = None,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Rank same-head single-source OLS probes; average heads and K/V.

    Returns ``(layers[L,k], scores[target_layer,source_layer], evaluation)``.
    Each head uses 1-SSE/SST over its coordinates; constant target heads carry
    no selection information and contribute zero. Ties prefer lower layers.
    ``provided_validation`` records a separate supplied pair, not proof of a
    held-out UID split. Without that pair, scores are explicitly ``in_sample``.
    """
    pairs = _paired_rows(source, target)
    layers, _, width = pairs[0][0].shape
    if num_heads < 1 or width % num_heads or not 1 <= k <= layers:
        raise ValueError("num_heads must divide cache width and 1 <= k <= layers")
    if (validation_source is None) != (validation_target is None):
        raise ValueError("provide both validation caches or neither")
    if validation_source is None:
        evaluation_pairs, evaluation = pairs, "in_sample"
    else:
        evaluation_pairs = _paired_rows(validation_source, validation_target)
        if evaluation_pairs[0][0].shape[::2] != (layers, width):
            raise ValueError("validation cache layer count and width must match fitting")
        evaluation = "provided_validation"
    head_width = width // num_heads
    scores = pairs[0][0].new_zeros(layers, layers)
    for (x, y), (eval_x, eval_y) in zip(pairs, evaluation_pairs, strict=True):
        for target_layer in range(layers):
            for source_layer in range(layers):
                for head in range(num_heads):
                    columns = slice(head * head_width, (head + 1) * head_width)
                    weight, bias = fit_affine_ridge(
                        x[source_layer, :, columns], y[target_layer, :, columns], ridge=0
                    )
                    truth = eval_y[target_layer, :, columns]
                    prediction = eval_x[source_layer, :, columns] @ weight + bias
                    total = (truth - truth.mean(0)).square().sum()
                    residual = (truth - prediction).square().sum()
                    r2 = torch.where(
                        total > 0,
                        1 - residual / total.clamp_min(torch.finfo(total.dtype).tiny),
                        torch.zeros_like(total),
                    )
                    scores[target_layer, source_layer] += r2 / (2 * num_heads)
    selected = scores.argsort(dim=1, descending=True, stable=True)[:, :k]
    return selected, scores, evaluation


def _features(values: torch.Tensor, layers: torch.Tensor) -> torch.Tensor:
    # [L,B,N,D] -> [B,N,kD], retaining complete heads for every chosen layer.
    selected = values.index_select(0, layers.to(values.device))
    return selected.permute(1, 2, 0, 3).flatten(2)


@dataclass
class KVTranslator:
    source_layers: torch.Tensor
    k_weight: torch.Tensor
    k_bias: torch.Tensor
    v_weight: torch.Tensor
    v_bias: torch.Tensor
    selection_scores: torch.Tensor | None = None
    selection_evaluation: str = "explicit_parameters"

    @torch.no_grad()
    def apply(self, cache: HSTUKVCache) -> HSTUKVCache:
        """Map all rows from the same input snapshot, without mutating it.

        Parameters follow the input device/dtype; no RoPE or model operations
        are inserted. The returned cache retains the incoming sequence length.
        """
        outputs = []
        for values, weights, biases in (
            (cache.k, self.k_weight, self.k_bias),
            (cache.v, self.v_weight, self.v_bias),
        ):
            weights, biases = weights.to(values), biases.to(values)
            outputs.append(torch.stack([
                _features(values, selected) @ weights[layer] + biases[layer]
                for layer, selected in enumerate(self.source_layers)
            ]))
        return HSTUKVCache(outputs[0], outputs[1], cache.seq_len)


@torch.no_grad()
def fit(
    source: HSTUKVCache,
    target: HSTUKVCache,
    num_heads: int,
    k: int = 2,
    ridge: float = 0.01,
    *,
    selection_source: HSTUKVCache | None = None,
    selection_target: HSTUKVCache | None = None,
) -> KVTranslator:
    """Select source layers and fit independent K/V affine ridge maps.

    Selection probes fit on source/target and are scored on the optional
    selection pair. Final ridge always fits source/target alone. All target
    heads share a Gram but have independent output parameters.
    """
    selected, scores, evaluation = select_source_layers(
        source, target, num_heads, k,
        validation_source=selection_source, validation_target=selection_target,
    )
    parameters = []
    for source_values, target_values in ((source.k, target.k), (source.v, target.v)):
        weights, biases = [], []
        for layer, source_layers in enumerate(selected):
            x = _features(source_values, source_layers).flatten(0, 1)
            y = target_values[layer].flatten(0, 1)
            weight, bias = fit_affine_ridge(x, y, ridge)
            weights.append(weight)
            biases.append(bias)
        parameters.extend((torch.stack(weights), torch.stack(biases)))
    return KVTranslator(selected, *parameters, scores, evaluation)
