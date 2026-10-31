"""RecFlow arithmetic ledger; this is not the legacy Yambda ELU ledger.

FMA=2; exp/log/sin/cos/reciprocal/sqrt each count as one. SiLU is
negate/exp/add/reciprocal/multiply=5. LayerNorm uses the declared two-pass
mean/variance equation (7D+2 per row), RMSNorm uses 4D+2. These are analytical
equation FLOPs, not a claim about exact fused CUDA instructions. Integer ID
mapping, table lookup, indexing, comparisons, RNG and copies are excluded.
Dense attention charges masked pairs. Native Triton charges visited 32x64
tiles, including padding; its mask is a select, unlike eager multiplication.

Every curve is charged as a standalone policy. Sharing teacher capture,
embeddings or a prefix across policies accelerates the experiment but does
not make one policy's deployment or calibration free.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2


def parameter_count(width, channels=320, dim=32):
    return channels * (2 * dim * width + width + dim)


def deployment_flops(rows, width, channels=320, dim=32):
    return channels * rows * (4 * dim * width + 6 * width + 6 * dim)


def step_flops(batch, width, channels=320, dim=32):
    """Two forward and three backward GEMMs; frozen inputs need no dX.

    AdamW foreach=False/fused=False: 14 parameter operations and 12 scalar
    operations for each of four parameter tensors, matching the Figure9b
    convention. Actual partial minibatches are counted by fitting_flops.
    """
    return dict(
        forward_gemm=channels * batch * 4 * dim * width,
        forward_pointwise=channels * batch * (6 * width + dim),
        backward_gemm=channels * batch * 6 * dim * width,
        backward_pointwise=channels * (batch * 9 * width + (batch - 1) * (width + dim)),
        loss_forward_backward=channels * batch * 5 * dim,
        optimizer=14 * parameter_count(width, channels, dim) + 48,
    )


def fitting_flops(observations, width, epochs=8, batch_rows=8192,
                  teacher_flops=0, channels=320, dim=32):
    if min(observations, width, epochs, batch_rows, channels, dim) < 1 or teacher_flops < 0:
        raise ValueError("Positive dimensions/epochs and nonnegative teacher FLOPs required")
    full, remainder = divmod(observations, batch_rows)
    batches = [(batch_rows, full)] + ([(remainder, 1)] if remainder else [])
    parts = {key: 0 for key in step_flops(1, width, channels, dim)}
    steps = 0
    for size, count in batches:
        steps += epochs * count
        for key, value in step_flops(size, width, channels, dim).items():
            parts[key] += epochs * count * value
    # Delta formation plus source/delta mean, center, variance, scale.
    # Reductions use the established Figure9b N-term convention.
    preparation = dict(
        teacher_source_preparation=int(teacher_flops),
        normalization=11 * channels * observations * dim + 2 * channels * dim,
        initialization=channels * dim * width + 2,
        fit_diagnostics=2 * channels * min(2048, observations) *
                        (4 * dim * width + 6 * width + 4 * dim),
    )
    parts = {**preparation, **parts}
    return dict(parts=parts, total_flops=sum(parts.values()), observations=observations,
                epochs=epochs, steps=steps, training_rows=epochs * observations,
                parameters=parameter_count(width, channels, dim),
                arithmetic="Analytical equation FLOPs; FMA=2; actual minibatches; no dX/backbone gradients; RNG/indexing/copies excluded")


@dataclass(frozen=True)
class RecFlowCostModel:
    hidden_size: int = 320
    num_layers: int = 16
    num_heads: int = 10
    item_embedding_dim: int = 320
    temporal_frequencies: int = 16
    relative_position_bias: bool = True
    attention_backend: str = "torch"
    skip_last_output: bool = True
    max_seq_len: int = 1025

    def __post_init__(self):
        if min(self.hidden_size, self.num_layers, self.num_heads, self.item_embedding_dim) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("Hidden width must divide into heads")
        if self.attention_backend not in ("torch", "triton"):
            raise ValueError("Specify actual torch or triton execution")

    @classmethod
    def from_model(cls, ranker, *, attention_backend="torch", skip_last_output=True):
        model = ranker.backbone
        cfg = model.cfg
        if (cfg.block_variant != "hstu_reference" or cfg.activation != "silu" or
                cfg.gating != "silu_gate" or cfg.causal_diagonal != "inclusive" or
                cfg.attn_dropout != 0 or cfg.head_dim != cfg.hidden_size // cfg.num_heads):
            raise ValueError("Cost ledger requires the actual RecFlow HSTU-reference architecture")
        if ranker.features.projection.in_features != ranker.features.item.embedding_dim + 48:
            raise ValueError("Unexpected RecFlow metadata encoder shape")
        return cls(cfg.hidden_size, cfg.num_layers, cfg.num_heads,
                   ranker.features.item.embedding_dim, model.temporal_enc.num_freqs,
                   cfg.relative_position_bias, attention_backend, skip_last_output, cfg.max_seq_len)

    @property
    def head_dim(self):
        return self.hidden_size // self.num_heads

    def layer_norm(self, rows):
        return rows * (7 * self.hidden_size + 2)

    def rms_norm(self, rows):
        return rows * (4 * self.hidden_size + 2)

    def features(self, rows):
        """ItemFeatures: numeric transform/linear, metadata projection, LN, validity mask."""
        d = self.hidden_size
        # Duration /1000, log1p, /10 =3; 1->8 affine =16+8.
        return rows * (27 + 2 * (self.item_embedding_dim + 48) * d + d) + self.layer_norm(rows)

    def embedding(self, rows, *, query=False, calls=1, include_features=True):
        d, f = self.hidden_size, self.temporal_frequencies
        # Frequency setup: f multiplications/divisions/exp; phases/sin/cos=3f.
        return ((self.features(rows) if include_features else 0) +
                rows * (4 * f * d + 2 * d * d + 3 * f + (3 if query else 2) * d) +
                calls * 3 * f)

    def attention_pairs(self, rows, keys, query_start=0, *, window_size=None):
        if min(rows, keys) < 0:
            raise ValueError("Negative attention shape")
        if rows == 0 or keys == 0:
            return 0
        if self.attention_backend == "torch":
            return rows * keys
        pairs = 0
        for start in range(0, rows, 32):
            first = max(0, (query_start + start - window_size + 1) // 64 * 64) if window_size else 0
            last = min(keys, query_start + start + 32)
            pairs += 32 * 64 * ceil(max(0, last - first) / 64)
        return pairs

    def kv_only(self, rows):
        d = self.hidden_size
        return self.layer_norm(rows) + rows * (4 * d * d + 10 * d)

    def attention(self, pairs, *, dense=None):
        # QK + AV, scale, optional bias, SiLU, division, eager mask multiply.
        if dense is None:
            dense = self.attention_backend == "torch"
        pointwise = 1 + int(self.relative_position_bias) + 5 + 1 + int(dense)
        return pairs * (4 * self.hidden_size + pointwise * self.num_heads)

    def block(self, rows, pairs, *, dense=None, residual_scales=False):
        d = self.hidden_size
        # Q,K,V,gate,out projections, 3 projection SiLUs + gate SiLU,
        # gate multiply, residual addition and two affine LayerNorms.
        return (10 * rows * d * d + 2 * self.layer_norm(rows) +
                rows * d * (22 + 2 * int(residual_scales)) + self.attention(pairs, dense=dense))

    def full_cache(self, n, batch=1, *, include_valid_masks=False):
        rows, pairs = batch * n, batch * self.attention_pairs(n, n)
        full_layers = self.num_layers - int(self.skip_last_output)
        cost = self.embedding(rows) + full_layers * self.block(rows, pairs, residual_scales=True)
        cost += self.kv_only(rows) if self.skip_last_output else 0
        # cache_ops.encode_cache uses unpadded inputs and omits valid masks.
        # Optional argument describes original RequestRanker.encode_history.
        if include_valid_masks:
            cost += rows * self.hidden_size * (1 + full_layers)
        return cost

    def selected(self, n, selected_tokens, batch=1):
        """Arbitrary selected positions, dense masked attention; whole history native endpoint."""
        m = min(n, selected_tokens)
        if m < 0:
            raise ValueError("Negative selected row count")
        if not m:
            return 0
        if m == n:
            return self.full_cache(n, batch)
        rows = batch * m
        full_layers = self.num_layers - int(self.skip_last_output)
        return (self.embedding(rows) + full_layers * self.block(rows, batch * m * n, dense=True) +
                (self.kv_only(rows) if self.skip_last_output else 0))

    def layer(self, n, start, end, batch=1, *, interval_end_kv_only=True):
        if not 0 <= start <= end < self.num_layers:
            raise ValueError("Layer interval must be zero-based inclusive")
        rows, pairs = batch * n, batch * self.attention_pairs(n, n)
        skip = interval_end_kv_only or (self.skip_last_output and end == self.num_layers - 1)
        return ((self.embedding(rows) if start == 0 else 0) +
                (end - start + 1 - int(skip)) * self.block(rows, pairs, residual_scales=True) +
                (self.kv_only(rows) if skip else 0))

    def deviation_selector(self, n, batch=1):
        rows, d = batch * n, self.hidden_size
        # cache_ops.deviation_scores: squared K/V deltas and means, no tie floor.
        return self.embedding(rows) + self.kv_only(rows) + rows * (6 * d + 1)

    def tail(self, n, selected_tokens, batch=1):
        m = min(n, selected_tokens)
        if m < 0:
            raise ValueError("Negative selected row count")
        return self.append(n - m, m, batch) if m else 0

    def append(self, prefix_n, new_tokens=1, batch=1, *, window_size=None):
        if not new_tokens:
            return 0
        rows = batch * new_tokens
        pairs = batch * self.attention_pairs(new_tokens, prefix_n + new_tokens,
                                             prefix_n, window_size=window_size)
        full_layers = self.num_layers - int(self.skip_last_output)
        return (self.embedding(rows) + full_layers * self.block(rows, pairs, residual_scales=True) +
                (self.kv_only(rows) if self.skip_last_output else 0))

    def rolling_append(self, prefix_n, new_tokens, batch=1, *, window_size=1024):
        """cache_ops.append_behaviors: embed once, evict before EACH scalar write."""
        if new_tokens < 0 or not 0 <= prefix_n <= window_size:
            raise ValueError("Invalid rolling shape")
        rows = batch * new_tokens
        total = self.embedding(rows)
        for offset in range(new_tokens):
            prefix = min(prefix_n + offset, window_size - 1)
            pairs = batch * self.attention_pairs(1, prefix + 1, prefix)
            full_layers = self.num_layers - int(self.skip_last_output)
            total += full_layers * self.block(batch, pairs, residual_scales=True)
            total += self.kv_only(batch) if self.skip_last_output else 0
        return total

    def read(self, n, queries=1, batch=1, *, chunk_size=128):
        """Shared-cache independent candidates, actual self read, final RMSNorm/head."""
        if not queries:
            return 0
        rows, d, h = batch * queries, self.hidden_size, self.num_heads
        # Prefix attention is dense regardless of the history encoding backend.
        prefix = self.attention(rows * n, dense=True)
        # Self dot uses D multiplies+(D-1) adds/head; SiLU5, scale1,
        # optional bias1, divisor1, weight*V + prefix add (2D).
        self_read = rows * (4 * d + h * (6 + int(self.relative_position_bias)))
        return (self.embedding(rows, query=True, calls=ceil(queries / chunk_size)) + self.num_layers *
                (self.block(rows, 0) + prefix + self_read) +
                self.rms_norm(rows) + rows * (2 * d + 1))

    def operation_cost(self, method, n, *, selected_tokens=0, interval=None, width=None,
                       batch=1, interval_end_kv_only=True):
        selection, repair, comparisons, boundary = 0, 0, 0., 0
        if method == "reuse":
            pass
        elif method == "rebuild":
            repair = self.full_cache(n, batch)
        elif method in ("alignment", "kv_map"):
            if width is None:
                raise ValueError("Map width required")
            repair = deployment_flops(batch * n, width, 2 * self.num_layers * self.num_heads, self.head_dim)
        elif method == "layer":
            if interval is None:
                raise ValueError("Layer interval required")
            repair = self.layer(n, *interval, batch=batch, interval_end_kv_only=interval_end_kv_only)
            boundary = batch * n * self.hidden_size * 4 if interval[0] else 0
        elif method in ("tail", "deviation"):
            repair = self.tail(n, selected_tokens, batch) if method == "tail" else self.selected(n, selected_tokens, batch)
            if method == "deviation" and 0 < selected_tokens < n:
                selection = self.deviation_selector(n, batch)
                comparisons = batch * (n * log2(max(n, 1)) + selected_tokens * log2(max(selected_tokens, 1)))
        else:
            raise ValueError(f"Unknown repair method: {method}")
        return dict(recompute_flops=repair, selection_flops=selection, total_flops=repair + selection,
                    selection_sort_comparisons_estimate=comparisons, old_boundary_bytes=boundary)
