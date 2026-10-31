"""Executed-shape arithmetic for the four selective-recomputation baselines.

The units are analytical forward FLOPs: multiply-add=2; an elementary
activation, reciprocal/sqrt or trigonometric operation counts as one. These
are not measured GPU instructions or elapsed time. Dense attention pays for
masked pairs; native Triton attention pays for every visited 32x64 tile,
including padding. The legacy model's final block output is charged whenever
the executor computes it. Memory traffic, indexing, comparisons and sorting
are not floating-point operations; sorting work is reported separately.

All three frozen architectures are legacy ELU+1, RMSNorm, SiLU-gated HSTU
with head width32, projection width=hidden width, no attention bias. This
module intentionally does not describe other model variants.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, log2
from typing import Mapping


@dataclass(frozen=True)
class CostModel:
    hidden_size: int
    num_layers: int
    num_heads: int
    attention_backend: str = "triton"

    @classmethod
    def for_scale(cls, scale: str, attention_backend: str = "triton") -> "CostModel":
        dimensions = {"medium": (192, 6, 6), "large": (320, 10, 10), "max": (320, 16, 10)}
        return cls(*dimensions[scale], attention_backend=attention_backend)

    def __post_init__(self):
        if self.attention_backend not in ("torch", "triton"):
            raise ValueError("record the resolved attention backend: torch or triton")
        if self.hidden_size != 32 * self.num_heads:
            raise ValueError("this frozen experiment uses head width32")

    def attention_pairs(self, rows: int, keys: int, query_start: int = 0,
                        *, window_size: int | None = None) -> int:
        """QK / AV matrix entries per head, matching native causal execution.

        Triton implementation: models/triton_attention.py, BM32/BN64/BD32.
        Native one-token append evicts before execution. Grouped band append
        keeps the prefix and masks out each query's individually evicted rows.
        Dense eager matmuls visit all row/key pairs, including window-masked ones.
        """
        if rows == 0 or keys == 0:
            return 0
        if self.attention_backend == "torch":
            return rows * keys
        pairs = 0
        for start in range(0, rows, 32):
            first_key = (max(0, (query_start + start - window_size + 1) // 64 * 64)
                         if window_size is not None else 0)
            last_key = min(keys, query_start + start + 32)
            pairs += 32 * 64 * ceil(max(0, last_key - first_key) / 64)
        return pairs

    def rms_norm(self, rows: int) -> int:
        return rows * (4 * self.hidden_size + 2)

    def embedding(self, rows: int, *, query: bool = False, calls: int = 1) -> int:
        """Fourier time features, time projection, input sums and input projection.

        Frequency setup (16 multiplication/division/exp each) is per call.
        Table lookup and concatenation incur memory traffic, not FLOPs.
        """
        d = self.hidden_size
        return (rows * (2 * (32 * d + d * d) + 48 + (3 if query else 2) * d)
                + 48 * calls) if rows else 0

    def block(self, rows: int, pairs: int, *, residual_scales: bool = True) -> int:
        """Five complete projections, RMSNorm, attention, gate and residual.

        `pairs` is an executed matrix shape, not the useful causal triangle.
        Scale, ELU, +1 and mask count as four pointwise attention operations.
        """
        d, h = self.hidden_size, self.num_heads
        return (10 * rows * d * d + 4 * d * pairs + 4 * h * pairs
                + self.rms_norm(rows) + (5 if residual_scales else 3) * rows * d)

    def full_cache(self, n: int, batch: int = 1) -> int:
        """Current.compute_kv execution, including last block and final norm."""
        if n == 0:
            return 0
        return (self.embedding(batch * n)
                + self.num_layers * self.block(batch * n, batch * self.attention_pairs(n, n))
                + self.rms_norm(batch * n))

    def append(self, prefix_n: int, new_tokens: int = 1, batch: int = 1,
               *, new_kv_only: bool = True) -> int:
        if not new_tokens:
            return 0
        pairs = batch * self.attention_pairs(new_tokens, prefix_n + new_tokens, prefix_n)
        rows = batch * new_tokens
        return (self.embedding(rows)
                + self.num_layers * self.block(rows, pairs, residual_scales=not new_kv_only)
                + self.rms_norm(rows))

    def band_append(self, prefix_n: int, new_tokens: int, batch: int = 1,
                    *, window_size: int = 1024) -> int:
        """Existing bounded-window grouped append, including complete blocks.

        The actual helper keeps the old prefix, appends the whole suffix with
        a per-query sliding mask, then retains the newest context. It computes
        no final model norm. Tiled/dense masked pairs are charged as executed.
        """
        if not new_tokens:
            return 0
        rows = batch * new_tokens
        pairs = batch * self.attention_pairs(new_tokens, prefix_n + new_tokens,
                                             prefix_n, window_size=window_size)
        return self.embedding(rows) + self.num_layers * self.block(rows, pairs)

    def native_read(self, n: int, queries: int = 1, batch: int = 1) -> int:
        """Full ordinary current-model query reader, including scalar head."""
        rows = batch * queries
        pairs = rows * self.attention_pairs(1, n + 1, n)
        return (self.embedding(rows, query=True)
                + self.num_layers * self.block(rows, pairs)
                + self.rms_norm(rows) + rows * (2 * self.hidden_size + 1))

    def layer_rebuild(self, n: int, interval: tuple[int, int], batch: int = 1) -> int:
        start, end = interval
        if not 0 <= start <= end < self.num_layers:
            raise ValueError("replayed interval outside frozen architecture")
        if not n:
            return 0
        rows, pairs = batch * n, batch * self.attention_pairs(n, n)
        return ((self.embedding(rows) if start == 0 else 0)
                + (end - start + 1) * self.block(rows, pairs))

    def tail_rebuild(self, n: int, selected_tokens: int, batch: int = 1) -> int:
        m = min(n, selected_tokens)
        # Existing hybrid_tail_refresh calls full forward_with_cache.
        return self.append(n - m, m, batch, new_kv_only=False) if m else 0

    def sparse_rebuild(self, n: int, selected_tokens: int, batch: int = 1) -> int:
        m = min(n, selected_tokens)
        if not m:
            return 0
        # Arbitrary causal rows use dense QK/AV, including masked entries.
        return (self.embedding(batch * m)
                + self.num_layers * self.block(batch * m, batch * m * n,
                                               residual_scales=False))

    def deviation_selector(self, n: int, batch: int = 1) -> int:
        if not n:
            return 0
        d, rows = self.hidden_size, batch * n
        # Two projections, two squared K/V differences and their combined mean.
        # The fixed numerical tie floor also computes two current K/V squared
        # means, adds them and scales by (32 * dtype_eps)^2: another4D+2/row.
        # Comparison and masked fill are logical/memory work, not FLOPs.
        return (self.embedding(rows) + self.rms_norm(rows) + 4 * rows * d * d
                + rows * ((6 * d + 1) + (4 * d + 2)))

    def query_selector(self, n: int, queries: int = 1, batch: int = 1) -> int:
        if not n:
            return 0
        d, h, rows = self.hidden_size, self.num_heads, batch * queries
        pairs = rows * n
        # q projection + QK, scale/ELU/+1, abs and sum/mean over head/query axes.
        return (self.embedding(rows, query=True) + self.rms_norm(rows)
                + 2 * rows * d * d + 2 * d * pairs + 5 * h * pairs)

    def operation_cost(self, method: str, n: int, *, selected_tokens: int = 0,
                       interval: tuple[int, int] | None = None,
                       queries: int = 1, batch: int = 1) -> dict:
        """Standalone repair of one request's common rolling-Reuse snapshot.

        Every partial deviation/query budget pays for its selector independently.
        When all tokens are selected, the kernel directly invokes compute_kv
        and need not select positions. This also applies to short histories.
        A zero budget is actual Reuse and runs neither selector nor replay.
        Layer calibration is added once by the caller, not once per request.
        """
        selection = recompute = 0
        comparisons = 0.0
        if method == "layer":
            if interval is not None:
                recompute = self.layer_rebuild(n, interval, batch)
        elif method == "tail":
            recompute = self.tail_rebuild(n, selected_tokens, batch)
        elif method in ("deviation", "query"):
            if n > 0 and selected_tokens >= n:
                recompute = self.full_cache(n, batch)
            elif min(n, selected_tokens) > 0:
                selection = (self.deviation_selector(n, batch) if method == "deviation"
                             else self.query_selector(n, queries, batch))
                recompute = self.sparse_rebuild(n, selected_tokens, batch)
                m = min(n, selected_tokens)
                comparisons = batch * (n * log2(max(n, 1)) + m * log2(max(m, 1)))
        else:
            raise ValueError(f"unknown baseline: {method}")
        return {"recompute_flops": recompute, "selection_flops": selection,
                "total_flops": recompute + selection,
                "selection_sort_comparisons_estimate": comparisons}

    def workload_denominator(self, full_history_hist: Mapping[int, int],
                             append_prefix_hist: Mapping[int, int], *,
                             append_new_kv_only: bool = True,
                             torch_full_history_hist: Mapping[int, int] | None = None,
                             torch_append_prefix_hist: Mapping[int, int] | None = None,
                             band_append_hist: Mapping[str, int] | None = None,
                             torch_band_append_hist: Mapping[str, int] | None = None) -> dict:
        """Actual request count histogram and one-token append count histogram.

        Both count individual users, including empty histories where relevant.
        Ordinary same-shape scoring reads and initial Parent cache construction
        cancel. Full recomputes each request and does not maintain rolling K/V.
        Temporal-frequency call setup is modeled per request/event here; the
        48-operation setup difference from batching is documented, negligible.
        Optional Torch histograms count subsets of the corresponding total
        histograms. They replace this model's native backend cost for initially
        short cohorts, including after those users reach the retained length.
        Grouped calls have their own ``"prefix_n:new_tokens" -> users`` histogram;
        these calls must not also appear in the scalar append histogram. The
        optional Torch band histogram is a subset of the complete band histogram.
        """
        full_history_hist = {int(n): int(count) for n, count in full_history_hist.items()}
        append_prefix_hist = {int(n): int(count) for n, count in append_prefix_hist.items()}
        torch_full = {int(n): int(count) for n, count in (torch_full_history_hist or {}).items()}
        torch_append = {int(n): int(count) for n, count in (torch_append_prefix_hist or {}).items()}
        bands = {tuple(map(int, key.split(":"))): int(count) for key, count in (band_append_hist or {}).items()}
        torch_bands = {tuple(map(int, key.split(":"))): int(count) for key, count in (torch_band_append_hist or {}).items()}
        if any(len(shape) != 2 or shape[0] < 0 or shape[1] < 1 or count < 0
               for shape, count in bands.items()):
            raise ValueError("band histogram keys must be prefix_n:new_tokens with positive suffix length")
        for subset, total in ((torch_full, full_history_hist), (torch_append, append_prefix_hist),
                              (torch_bands, bands)):
            if any(count < 0 or count > total.get(n, 0) for n, count in subset.items()):
                raise ValueError("Torch workload counts must be subsets of total workload counts")
        full = sum(int(count) * self.full_cache(int(n)) for n, count in full_history_hist.items())
        reuse = sum(int(count) * self.append(int(n), new_kv_only=append_new_kv_only)
                    for n, count in append_prefix_hist.items())
        eager = replace(self, attention_backend="torch")
        full += sum(count * (eager.full_cache(n) - self.full_cache(n)) for n, count in torch_full.items())
        reuse += sum(count * (eager.append(n, new_kv_only=append_new_kv_only)
                              - self.append(n, new_kv_only=append_new_kv_only))
                     for n, count in torch_append.items())
        reuse += sum(count * self.band_append(*shape) for shape, count in bands.items())
        reuse += sum(count * (eager.band_append(*shape) - self.band_append(*shape))
                     for shape, count in torch_bands.items())
        return {"full_history_flops": full, "reuse_append_flops": reuse,
                "full_minus_reuse_flops": full - reuse}

    def layer_profile(self, histories_and_queries: list[tuple[int, int]]) -> dict:
        """Complete contiguous-interval profiler, paid once per standalone curve.

        Includes Parent cache acquisition for calibration users, Current teacher
        cache/read, all candidate replays/reads, and logit MSE. No evaluation
        labels select intervals. Boundary capture adds memory, not FLOPs.
        """
        intervals = [(start, end) for start in range(self.num_layers)
                     for end in range(start, self.num_layers)]
        parts = {"profile_parent_cache": 0, "profile_teacher_cache": 0,
                 "profile_teacher_reads": 0, "profile_interval_rebuilds": 0,
                 "profile_interval_reads": 0, "profile_logit_mse": 0}
        for n, queries in histories_and_queries:
            parts["profile_parent_cache"] += self.full_cache(n)
            parts["profile_teacher_cache"] += self.full_cache(n)
            parts["profile_teacher_reads"] += self.native_read(n, queries)
            parts["profile_interval_rebuilds"] += sum(self.layer_rebuild(n, interval) for interval in intervals)
            parts["profile_interval_reads"] += len(intervals) * self.native_read(n, queries)
            parts["profile_logit_mse"] += len(intervals) * 3 * queries
        return {"components": parts, "calibration_flops": sum(parts.values()),
                "profile_intervals": len(intervals), "profile_users": len(histories_and_queries)}


def normalized_point(*, full_auc: float, reuse_auc: float, baseline_auc: float,
                     extra_flops: float, full_minus_reuse_flops: float) -> dict:
    """Preserve signed and >100% values; undefined ratios remain explicit."""
    gap = full_auc - reuse_auc
    return {"auc_gap": gap,
            "recovery_percent": 100 * (baseline_auc - reuse_auc) / gap if gap != 0 else None,
            "relative_flops_percent": (100 * extra_flops / full_minus_reuse_flops
                                       if full_minus_reuse_flops != 0 else None)}
