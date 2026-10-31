"""Batched E6 SVD with the scalar decomposition as its numerical oracle.

One batch contains equal C/N/D heads. Main effects and K/V RMS are still
computed independently per head; batching only changes linear algebra calls.
The returned objects retain compact top-r factors rather than full-SVD views.
"""
from __future__ import annotations

import torch
import time

from experiments.ch3_motivation.code.analysis import InteractionDecomposition
from experiments.ch3_motivation.code.energy_certificate import sketch_energy_certificate


class BatchedReconstruction:
    """Pack compact factors once, then replay all heads in one matmul/chunk."""
    def __init__(self, original, decompositions):
        self.original = original
        self.max_rank = decompositions[0].max_rank
        self.selected_slice = decompositions[0].selected_slice
        self.row_effect = torch.stack([obj.row_effect for obj in decompositions])
        self.candidate_effect = torch.stack([obj.candidate_effect for obj in decompositions])
        self.grand_mean = torch.stack([obj.grand_mean for obj in decompositions])
        self.block_scales = torch.stack([obj.block_scales for obj in decompositions])
        dimensions = decompositions[0].left.shape[0]//original.shape[1]
        self.left = torch.stack([obj.left for obj in decompositions]).reshape(
            len(decompositions), original.shape[1], dimensions, self.max_rank)
        self.singular_values = torch.stack([obj.singular_values for obj in decompositions])
        self.right = torch.stack([obj.right for obj in decompositions])

    def reconstruct(self, rank, *, candidates=slice(None), rows=slice(None)):
        original = self.original[:, candidates][:, :, rows]
        if rank is None:
            return original
        if not 0 <= rank <= self.max_rank:
            raise ValueError("rank exceeds this shared decomposition")
        main = self.row_effect[:, :, rows] + self.candidate_effect[:, candidates] - self.grand_mean
        left = self.left[:, candidates, :, :rank]*self.singular_values[:, None, None, :rank]
        shape = left.shape
        low = (left.reshape(shape[0], shape[1]*shape[2], rank)@self.right[:, :rank, rows])
        low = low.reshape(shape[0], shape[1], shape[2], -1).transpose(-1, -2)
        low = low*self.block_scales[:, None, None]
        result = original.clone()
        result[..., self.selected_slice] = (main[..., self.selected_slice]+low).to(result.dtype)
        return result


@torch.no_grad()
def decompose_interaction_batch(h, *, normalize_kv=False, only="both", method="exact",
        max_rank=16, oversample=8, power_iterations=1, seed=17,
        energy_guard=False, energy_threshold=.98, energy_tolerance=1e-5):
    """Return one decomposition per h[heads,candidates,history_rows,2D].

The shared random sketch is exactly the sketch used by scalar calls with the
same seed; it imposes no shared factors across heads. SVD signs/bases in tied
singular subspaces need not match, while reconstructed matrices do.
"""
    if h.ndim != 4 or not h.is_floating_point() or h.shape[-1] % 2 or min(h.shape) < 1:
        raise ValueError("h must be nonempty [heads,candidates,history_rows,2D]")
    if only not in ("both", "key", "value") or method not in ("exact", "randomized"):
        raise ValueError("unsupported block or decomposition method")
    if min(max_rank, oversample, power_iterations) < 0:
        raise ValueError("rank, oversampling, iterations must be nonnegative")
    work = h.detach().to(dtype=torch.float64 if h.dtype == torch.float64 else torch.float32)
    heads, candidates, rows, dimensions = h.shape
    width = dimensions // 2
    row_effect = work.mean(1, keepdim=True)
    candidate_effect = work.mean(2, keepdim=True)
    grand = work.mean((1, 2), keepdim=True)
    selected = {"both": slice(None), "key": slice(0, width), "value": slice(width, dimensions)}[only]
    centered = (work-row_effect-candidate_effect+grand)[..., selected]
    blocks = (centered[..., :width], centered[..., width:]) if only == "both" else (centered,)
    scales = torch.stack([part.square().mean((1, 2, 3)).sqrt() if normalize_kv
        else part.new_ones(heads) for part in blocks], -1)
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    block_scales = torch.repeat_interleave(scales, width, dim=-1)
    normalized = centered/block_scales[:, None, None]
    matrix = normalized.permute(0, 1, 3, 2).reshape(heads, -1, rows)
    rank = min(int(max_rank), min(matrix.shape[1:]))
    certificate_rows = None
    guard_seconds = fallback_seconds = 0.
    fallback_indices = set()
    if not rank:
        left = matrix.new_empty((heads, matrix.shape[1], 0))
        singular = matrix.new_empty((heads, 0))
        right = matrix.new_empty((heads, 0, rows))
    elif method == "exact":
        u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        left, singular, right = u[..., :rank], singular[..., :rank], vh[:, :rank]
    else:
        sketch_width = min(rank+oversample, min(matrix.shape[1:]))
        seeds = list(seed) if isinstance(seed, (tuple, list)) else None
        if seeds is not None and len(seeds) != heads:
            raise ValueError("one sketch seed per head is required")
        def sketch(value):
            generator = torch.Generator(device=matrix.device).manual_seed(int(value))
            return torch.randn(rows, sketch_width, dtype=matrix.dtype,
                device=matrix.device, generator=generator)
        omega = torch.stack([sketch(value) for value in seeds]) if seeds is not None else sketch(seed)
        basis = torch.linalg.qr(matrix@omega, mode="reduced").Q
        for _ in range(power_iterations):
            transposed = torch.linalg.qr(matrix.transpose(-1, -2)@basis, mode="reduced").Q
            basis = torch.linalg.qr(matrix@transposed, mode="reduced").Q
        small_u, singular, vh = torch.linalg.svd(basis.transpose(-1, -2)@matrix, full_matrices=False)
        all_sketch_singular = singular.clone()
        left, singular, right = basis@small_u[..., :rank], singular[..., :rank], vh[:, :rank]
    energy = matrix.square().sum((1, 2))
    if energy_guard and method == "randomized" and rank:
        def sync():
            if matrix.device.type == "cuda":
                torch.cuda.synchronize(matrix.device)
        sync()
        guard_started = time.perf_counter()
        # Frobenius defect bounds the spectral defect. The additional term is
        # an explicit numerical allowance, not an interval-arithmetic proof.
        gram = basis.transpose(-1, -2)@basis
        defect = torch.linalg.matrix_norm(gram-torch.eye(gram.shape[-1],
            dtype=gram.dtype, device=gram.device), ord="fro", dim=(-2, -1))
        defect = defect+10*torch.finfo(matrix.dtype).eps*gram.shape[-1]
        certificate = sketch_energy_certificate(energy, all_sketch_singular,
            ranks=(2, 6, 16), orthogonality_defect=defect,
            threshold=energy_threshold, relative_energy_tolerance=energy_tolerance)
        # One compact transfer for the batch, rather than a sync per matrix.
        values = torch.cat((energy[:, None].double(), certificate["range_energy"][:, None],
            certificate["residual_range_energy_upper_bound"][:, None], defect[:, None].double(),
            certificate["captured_energy"], certificate["captured_energy_ratio_lower_bound"],
            certificate["needs_refinement"][:, None].double(), all_sketch_singular.double()), -1).cpu().tolist()
        certificate_rows = []
        for head, values_head in enumerate(values):
            certificate_rows.append(dict(total_energy=values_head[0], range_energy=values_head[1],
                residual_range_energy_upper_bound=values_head[2], orthogonality_defect_upper_bound=values_head[3],
                ranks=[2, 6, 16], captured_energy=values_head[4:7],
                captured_energy_ratio_lower_bound=values_head[7:10],
                exact_fallback=bool(values_head[10]), all_sketch_singular_values=values_head[11:],
                threshold=energy_threshold, relative_energy_tolerance=energy_tolerance,
                scope="numerical energy screen with explicit allowances; not a formal floating-point proof"))
            if values_head[10]:
                fallback_indices.add(head)
        guard_seconds = time.perf_counter()-guard_started
        if fallback_indices:
            # Exact fallback uses the VERY SAME centered/normalized matrix.
            # It does not repeat scalar centering and is limited to failed heads.
            fallback_started = time.perf_counter()
            left, singular, right = left.clone(), singular.clone(), right.clone()
            for head in sorted(fallback_indices):
                u, s, vh = torch.linalg.svd(matrix[head], full_matrices=False)
                left[head], singular[head], right[head] = u[:, :rank], s[:rank], vh[:rank]
            sync()
            fallback_seconds = time.perf_counter()-fallback_started
    # Transfer one compact summary/batch. Per-head float(tensor) would insert
    # many host/device synchronizations and undo much of the batching gain.
    retained = torch.where(energy > 0, singular.square().sum(-1)/energy,
                           torch.ones_like(energy))
    summaries = torch.cat((energy[:, None], retained[:, None], scales), -1).cpu().tolist()
    result = []
    for head in range(heads):
        metadata = dict(method=method, maximum_rank=rank, requested_maximum_rank=int(max_rank),
            candidates=candidates, history_rows=rows, kv_dimensions=dimensions,
            matrix_shape=list(matrix.shape[1:]), normalize_kv=bool(normalize_kv), only=only,
            block_scales=summaries[head][2:], normalized_interaction_squared_norm=summaries[head][0],
            normalized_retained_energy_at_max_rank=summaries[head][1],
            oversample=oversample if method == "randomized" else None,
            power_iterations=power_iterations if method == "randomized" else None,
            seed=(seed[head] if isinstance(seed, (tuple, list)) else seed) if method == "randomized" else None,
            execution_batch_heads=heads,
            energy_guard=certificate_rows[head] if certificate_rows is not None else None,
            energy_guard_batch_seconds=guard_seconds, exact_fallback_batch_seconds=fallback_seconds,
            exact_fallback_batch_matrices=len(fallback_indices), execution_batch_index=head,
            main_effects="row mean + candidate mean - grand mean, preserved at every rank",
            query_protocol="h traced at frozen full-h queries; fixed during complete-path rank replay",
            interpretation="hindsight compression, not an out-of-sample deployable row encoder")
        result.append(InteractionDecomposition(h[head].detach(), row_effect[head], candidate_effect[head],
            grand[head], left[head].clone(), singular[head].clone(), right[head].clone(),
            selected, block_scales[head], metadata))
    return result
