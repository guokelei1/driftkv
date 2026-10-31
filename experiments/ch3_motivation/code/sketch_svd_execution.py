"""Isolated diagnostic interception of the small randomized-SVD projection."""
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import torch
from experiments.ch3_motivation.code.sketch_svd_gram import wide_svd
from hstu_kvcache.common.artifacts import sha256


def binding():
    paths = [Path(__file__), Path(__file__).with_name('sketch_svd_gram.py')]
    return dict(format='projected_svd_gram_fp64_diagnostic_v1',
        source_sha256={str(p.resolve()): sha256(p) for p in paths},
        scope='small wide projected matrix only; original sketch/QR/ranks and all-matrix guard; exact fallback unchanged',
        prerequisite='actual-model all-rank score comparison; not enabled in formal E6')


@contextmanager
def projected_svd():
    original = torch.linalg.svd
    counts = Counter()

    def svd(matrix, full_matrices=True, **kwargs):
        if (matrix.device.type == 'cuda' and matrix.dtype == torch.float32 and not full_matrices
                and matrix.ndim in (2, 3) and matrix.shape[-2] <= 64
                and matrix.shape[-1] >= 4 * matrix.shape[-2] and not kwargs):
            counts[str(tuple(matrix.shape))] += 1
            return wide_svd(matrix)
        return original(matrix, full_matrices=full_matrices, **kwargs)

    try:
        torch.linalg.svd = svd
        yield counts
    finally:
        torch.linalg.svd = original
