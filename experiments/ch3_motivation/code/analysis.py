"""E4 mismatch summaries and E6 hindsight compression for Chapter 3.

E4 sums K/V squared differences over all heads before comparing rows/layers.
Its relative variant divides each (layer, row) difference by that rebuilt
row's K/V squared norm.  These are numerical mismatch statistics, not weights
of an event's contribution to AUC.

E6 compresses an already evaluated, frozen ``h`` tensor.  A driver must first
trace the full f+g+h scoring path, decompose h at each history/layer/head, and
replay complete scoring with these reconstructed tensors.  The tensors are
held fixed while replay queries propagate through the model.  Full-rank replay
must recover the traced read/logits.  Lower-rank factors are hindsight factors
for that snapshot/candidate set, not a deployable row encoder.  A new snapshot
requires a new trace; only real-feedback candidates contribute to AUC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
import torch


def _share(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _quartiles(values: list[float | None]) -> dict[str, float | int | None]:
    finite = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=np.float64)
    if not len(finite):
        return dict(count=0, q25=None, median=None, q75=None)
    q25, median, q75 = np.quantile(finite, [.25, .5, .75]).tolist()
    return dict(count=len(finite), q25=q25, median=median, q75=q75)


def _mismatch_rows_and_values(old_cache, new_cache, *, uids=None, lengths=None,
                              top_fraction=.1, relative_floor=1e-12):
    if old_cache.k.shape != new_cache.k.shape or old_cache.v.shape != new_cache.v.shape:
        raise ValueError("old/rebuilt caches must be aligned")
    if old_cache.k.ndim != 4 or old_cache.k.shape != old_cache.v.shape:
        raise ValueError("K/V must both have shape [layers,batch,rows,width]")
    if not 0 < top_fraction <= 1 or relative_floor <= 0:
        raise ValueError("invalid concentration fraction or relative-norm floor")
    layers, batch, width, _ = old_cache.k.shape
    if old_cache.seq_len != width or new_cache.seq_len != width:
        raise ValueError("seq_len must match the aligned tensor rows")
    uids = list(range(batch)) if uids is None else [int(uid) for uid in uids]
    lengths = [width] * batch if lengths is None else [int(n) for n in lengths]
    if len(uids) != batch or len(lengths) != batch or any(n < 0 or n > width for n in lengths):
        raise ValueError("UIDs/lengths must describe the cache batch")

    # Only one layer's full-width difference is materialized at a time.  The
    # retained CPU arrays have one double per layer/row, independent of heads.
    difference = torch.empty((layers, batch, width), dtype=torch.float64)
    norm = torch.empty_like(difference)
    for layer in range(layers):
        delta = torch.zeros((batch, width), device=old_cache.k.device, dtype=torch.float64)
        rebuilt_norm = torch.zeros_like(delta)
        for old, new in ((old_cache.k, new_cache.k), (old_cache.v, new_cache.v)):
            current = new[layer].detach().to(dtype=torch.float64)
            delta += (current - old[layer].detach().to(dtype=torch.float64)).square().sum(-1)
            rebuilt_norm += current.square().sum(-1)
        difference[layer].copy_(delta.cpu())
        norm[layer].copy_(rebuilt_norm.cpu())
    relative = difference / norm.clamp_min(relative_floor)
    records, row_values = [], []
    for user, (uid, n) in enumerate(zip(uids, lengths, strict=True)):
        record: dict[str, Any] = dict(uid=uid, history_rows=n, long_history=n >= 256,
                                      top_fraction=float(top_fraction),
                                      selected_rows=int(math.ceil(top_fraction * n)),
                                      relative_floor=float(relative_floor),
                                      relative_floor_cells=int((norm[:, user, :n] < relative_floor).sum()))
        values = {}
        for name, array in (("absolute", difference), ("relative", relative)):
            selected = array[:, user, :n]
            per_row = selected.sum(0)
            per_layer = selected.sum(1)
            total = float(per_layer.sum())
            top = float(per_row.topk(record["selected_rows"]).values.sum()) if n else 0.
            shares = (per_layer / total).tolist() if total > 0 else [None] * layers
            evenness = [x * layers if x is not None else None for x in shares]
            record.update({f"{name}_total_deviation": total,
                           f"{name}_top_rows_deviation": top,
                           f"{name}_top10_row_share": _share(top, total),
                           f"{name}_layer_deviation": per_layer.tolist(),
                           f"{name}_layer_shares": shares,
                           f"{name}_layer_evenness": evenness,
                           f"{name}_max_layer_evenness": max(evenness) if total > 0 else None})
            values[name] = per_row
        records.append(record)
        row_values.append(values)
    return records, row_values


@torch.no_grad()
def mismatch_rows(old_cache, new_cache, *, uids=None, lengths=None,
                  top_fraction=.1, relative_floor=1e-12) -> list[dict[str, Any]]:
    """Return exportable per-user E4 statistics; lengths exclude padded rows.

    ``absolute_top10_row_share`` uses ceil(0.1 * actual_history_rows) entries.
    With a custom fraction the field is retained for stable exports, and the
    actual fraction/selected count are explicitly included.  A zero-mismatch
    history has undefined concentration (None), not a fabricated uniform share.
    """
    return _mismatch_rows_and_values(old_cache, new_cache, uids=uids, lengths=lengths,
                                    top_fraction=top_fraction, relative_floor=relative_floor)[0]


def summarize_mismatch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-user ratios and pooled layer/within-user-selected mass.

    The pooled-within-user top share weights each user's top-10% numerator by
    its mismatch mass.  It is distinct from choosing the largest 10% of rows
    after merging users.  ``MismatchAccumulator`` additionally reports that
    latter global concentration, using retained scalar row totals.
    """
    def subset(selected):
        result: dict[str, Any] = dict(users=len(selected), history_rows=sum(r["history_rows"] for r in selected))
        if not selected:
            return result
        layers = len(selected[0]["absolute_layer_deviation"])
        for name in ("absolute", "relative"):
            total = sum(r[f"{name}_total_deviation"] for r in selected)
            top = sum(r[f"{name}_top_rows_deviation"] for r in selected)
            layer_mass = np.asarray([r[f"{name}_layer_deviation"] for r in selected], dtype=np.float64).sum(0)
            layer_shares = (layer_mass / total).tolist() if total > 0 else [None] * layers
            result[name] = dict(
                top10_row_share=_quartiles([r[f"{name}_top10_row_share"] for r in selected]),
                max_layer_evenness=_quartiles([r[f"{name}_max_layer_evenness"] for r in selected]),
                layer_evenness=[_quartiles([r[f"{name}_layer_evenness"][l] for r in selected]) for l in range(layers)],
                pooled_total_deviation=total,
                pooled_within_user_top10_share=_share(top, total),
                pooled_layer_shares=layer_shares,
                pooled_layer_evenness=[x * layers if x is not None else None for x in layer_shares])
        return result
    return dict(all=subset(rows), history_at_least_256=subset([r for r in rows if r["long_history"]]),
                definitions=dict(row_deviation="sum of K/V squared differences over layers and heads",
                                 relative_deviation="per-layer/row squared difference divided by rebuilt K/V squared norm, with declared floor",
                                 zero_mismatch="undefined concentration; excluded from ratio quantiles"))


@dataclass
class MismatchAccumulator:
    """Stream cache batches, retaining only export rows and scalar row masses."""
    rows: list[dict[str, Any]] = field(default_factory=list)
    _row_values: list[dict[str, torch.Tensor]] = field(default_factory=list)

    @torch.no_grad()
    def update(self, old_cache, new_cache, **kwargs) -> list[dict[str, Any]]:
        records, values = _mismatch_rows_and_values(old_cache, new_cache, **kwargs)
        self.rows.extend(records)
        self._row_values.extend(values)
        return records

    def summary(self) -> dict[str, Any]:
        result = summarize_mismatch(self.rows)
        for group, indices in (("all", list(range(len(self.rows)))),
                               ("history_at_least_256", [i for i, r in enumerate(self.rows) if r["long_history"]])):
            if not indices:
                continue
            fractions = {self.rows[i]["top_fraction"] for i in indices}
            if len(fractions) != 1:
                raise ValueError("pooled concentration requires a common top fraction")
            fraction = fractions.pop()
            for name in ("absolute", "relative"):
                merged = torch.cat([self._row_values[i][name] for i in indices])
                n = int(math.ceil(fraction * merged.numel()))
                numerator = float(merged.topk(n).values.sum()) if n else 0.
                result[group][name]["pooled_global_top10_share"] = _share(numerator, float(merged.sum()))
        return result


@dataclass
class InteractionDecomposition:
    """One frozen h tensor and reusable factors of its double-centered part."""
    original: torch.Tensor
    row_effect: torch.Tensor
    candidate_effect: torch.Tensor
    grand_mean: torch.Tensor
    left: torch.Tensor
    singular_values: torch.Tensor
    right: torch.Tensor
    selected_slice: slice
    block_scales: torch.Tensor
    metadata: dict[str, Any]

    @property
    def max_rank(self) -> int:
        return int(self.singular_values.numel())

    def reconstruct(self, rank: int | None, *, candidates=slice(None), rows=slice(None)) -> torch.Tensor:
        """Return h_r, optionally only a candidate/row chunk.

        None denotes the original full-rank tensor and returns exact input
        values.  Integer ranks reconstruct the selected block from stored
        factors, retaining both main effects and any unselected K/V block.
        """
        original = self.original[candidates][:, rows]
        if rank is None:
            return original
        if rank < 0 or rank > self.max_rank:
            raise ValueError(f"rank must lie in [0,{self.max_rank}] or be None")
        count = self.original.shape[0]
        main = self.row_effect[:, rows] + self.candidate_effect[candidates] - self.grand_mean
        # Factor rows are candidate x feature; history positions are columns.
        selected_dimensions = self.left.shape[0] // count
        left = self.left.reshape(count, selected_dimensions, self.max_rank)[candidates, :, :rank]
        low = torch.matmul(left * self.singular_values[:rank], self.right[:rank, rows]).transpose(-1, -2)
        low = low * self.block_scales
        output = original.clone()
        output[..., self.selected_slice] = (main[..., self.selected_slice] + low).to(output.dtype)
        return output

    def relative_error(self, rank: int) -> float:
        """Squared Frobenius error of the full h tensor (diagnostic only)."""
        numerator = (self.reconstruct(rank).double() - self.original.double()).square().sum()
        denominator = self.original.double().square().sum()
        return float(numerator / denominator) if float(denominator) else 0.


@torch.no_grad()
def decompose_interaction(h: torch.Tensor, *, normalize_kv=False, only="both",
                          method="exact", max_rank=16, oversample=8,
                          power_iterations=1, seed=17) -> InteractionDecomposition:
    """Decompose h[C,N,2d], preserving row/candidate main effects.

    Matrix columns are history rows; matrix rows are candidate x K/V feature.
    K/V normalization uses each *centered interaction block's* RMS, restored
    after reconstruction.  A zero block receives scale one.  ``only='key'``
    or ``'value'`` leaves the other block exactly unchanged.

    Exact SVD is the reference.  Randomized SVD is a seeded top-r approximation
    for the large E6 matrices; the driver must compare it to exact SVD on a
    fixed diagnostic subset before adopting it.  This function records the
    captured interaction energy and algorithm rather than labeling approximate
    factors exact.  Every requested integer rank reuses this one decomposition.
    """
    if h.ndim != 3 or not h.is_floating_point() or h.shape[-1] % 2:
        raise ValueError("h must have shape [candidates,history_rows,2*head_width]")
    if not h.shape[0] or not h.shape[1] or not h.shape[2]:
        raise ValueError("h requires nonempty candidates, rows, and K/V features")
    if only not in ("both", "key", "value") or method not in ("exact", "randomized"):
        raise ValueError("unsupported block or decomposition method")
    if max_rank < 0 or oversample < 0 or power_iterations < 0:
        raise ValueError("rank, oversampling, and iteration counts must be nonnegative")
    work = h.detach().to(dtype=torch.float64 if h.dtype == torch.float64 else torch.float32)
    row_effect = work.mean(0, keepdim=True)
    candidate_effect = work.mean(1, keepdim=True)
    grand_mean = work.mean((0, 1), keepdim=True)
    width = h.shape[-1] // 2
    selected_slice = {"both": slice(None), "key": slice(0, width), "value": slice(width, 2 * width)}[only]
    centered = (work - row_effect - candidate_effect + grand_mean)[..., selected_slice]
    scales = []
    for part in ((centered[..., :width], centered[..., width:]) if only == "both" else (centered,)):
        rms = part.square().mean().sqrt() if normalize_kv else part.new_tensor(1.)
        scales.append(torch.where(rms > 0, rms, rms.new_tensor(1.)))
    block_scales = torch.repeat_interleave(torch.stack(scales), width)
    normalized = centered / block_scales
    matrix = normalized.permute(0, 2, 1).reshape(-1, h.shape[1])
    rank = min(int(max_rank), min(matrix.shape))
    if rank == 0:
        left = matrix.new_empty((matrix.shape[0], 0))
        singular_values = matrix.new_empty((0,))
        right = matrix.new_empty((0, matrix.shape[1]))
    elif method == "exact":
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        left, singular_values, right = u[:, :rank], s[:rank], vh[:rank]
    else:
        sketch_width = min(rank + oversample, min(matrix.shape))
        generator = torch.Generator(device=matrix.device).manual_seed(seed)
        omega = torch.randn(matrix.shape[1], sketch_width, device=matrix.device,
                            dtype=matrix.dtype, generator=generator)
        basis = torch.linalg.qr(matrix @ omega, mode="reduced").Q
        for _ in range(power_iterations):
            transposed = torch.linalg.qr(matrix.T @ basis, mode="reduced").Q
            basis = torch.linalg.qr(matrix @ transposed, mode="reduced").Q
        small_u, s, vh = torch.linalg.svd(basis.T @ matrix, full_matrices=False)
        left, singular_values, right = (basis @ small_u[:, :rank]), s[:rank], vh[:rank]
    energy = float(matrix.square().sum())
    metadata = dict(method=method, maximum_rank=rank, requested_maximum_rank=int(max_rank),
                    candidates=h.shape[0], history_rows=h.shape[1], kv_dimensions=h.shape[2],
                    matrix_shape=list(matrix.shape), normalize_kv=bool(normalize_kv), only=only,
                    block_scales=[float(x) for x in scales],
                    normalized_interaction_squared_norm=energy,
                    normalized_retained_energy_at_max_rank=(float(singular_values.square().sum()) / energy if energy else 1.),
                    oversample=int(oversample) if method == "randomized" else None,
                    power_iterations=int(power_iterations) if method == "randomized" else None,
                    seed=int(seed) if method == "randomized" else None,
                    main_effects="row mean + candidate mean - grand mean, preserved at every rank",
                    query_protocol="h traced at frozen full-h queries; compression tensors fixed during complete-path replay; repeat at each history snapshot",
                    interpretation="hindsight compression, not an out-of-sample deployable row encoder")
    return InteractionDecomposition(h.detach(), row_effect, candidate_effect, grand_mean,
                                    left, singular_values, right, selected_slice, block_scales, metadata)


def decompose_layer_heads(h: torch.Tensor, **kwargs) -> list[InteractionDecomposition]:
    """Decompose one layer's h[heads,candidates,history_rows,2*head_width]."""
    if h.ndim != 4:
        raise ValueError("layer h must have shape [heads,candidates,rows,2*width]")
    return [decompose_interaction(head, **kwargs) for head in h]
