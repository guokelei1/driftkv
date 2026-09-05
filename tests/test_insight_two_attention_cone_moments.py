from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from insight_two.attention_cone_moments import (
    attention_cone_mask_statistics,
    build_positive_affine_moments,
    decompose_elu_plus_one_response,
    read_positive_affine_moments,
    subtract_affine_moments,
)


def test_fixed_positive_cone_affine_moments_are_exact_with_qk_scale() -> None:
    generator = torch.Generator().manual_seed(311)
    # All candidates stay inside one mixed-sign cone: only coordinate zero is
    # active in q, while history keys alternate across that half-space.
    q = torch.zeros(2, 3, 5, 4)
    q[..., 0] = torch.rand(2, 3, 5, generator=generator) + 0.2
    k = torch.randn(2, 3, 7, 4, generator=generator)
    key_signs = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    k[..., 0] = key_signs * (k[..., 0].abs() + 0.2)
    v = torch.randn(2, 3, 7, 6, generator=generator)
    scale = 0.37
    decomposition = decompose_elu_plus_one_response(q, k, v, scale=scale)
    assert decomposition.positive_mask.any()
    assert (~decomposition.positive_mask).any()
    assert torch.equal(
        decomposition.positive_mask,
        decomposition.positive_mask[:, :, :1].expand_as(decomposition.positive_mask),
    )

    fixed_mask = decomposition.positive_mask[:, :, 0]
    moments = build_positive_affine_moments(k, v, fixed_mask)
    affine = read_positive_affine_moments(q, moments, scale=scale)

    assert torch.allclose(affine, decomposition.positive_response, atol=2e-6, rtol=2e-6)


def test_attention_cone_statistics_detect_shared_and_candidate_specific_masks() -> None:
    shared = torch.tensor(
        [[[[True, False, True, False], [True, False, True, False], [True, False, True, False]]]]
    )
    shared_stats = attention_cone_mask_statistics(shared)
    assert torch.allclose(shared_stats.positive_fraction, torch.tensor([[0.5]]))
    assert torch.allclose(shared_stats.shared_positive_fraction, torch.tensor([[0.5]]))
    assert torch.allclose(shared_stats.shared_negative_fraction, torch.tensor([[0.5]]))
    assert torch.equal(shared_stats.unanimous_fraction, torch.ones(1, 1))
    assert torch.equal(shared_stats.majority_agreement, torch.ones(1, 1))
    assert torch.equal(shared_stats.pairwise_sign_agreement, torch.ones(1, 1))
    assert torch.equal(shared_stats.pairwise_positive_jaccard, torch.ones(1, 1))

    varying = torch.tensor(
        [[[[True, False, True, False], [True, True, False, False]]]]
    )
    varying_stats = attention_cone_mask_statistics(varying)
    assert torch.allclose(varying_stats.unanimous_fraction, torch.tensor([[0.5]]))
    assert torch.allclose(varying_stats.majority_agreement, torch.tensor([[0.75]]))
    assert torch.allclose(varying_stats.pairwise_sign_agreement, torch.tensor([[0.5]]))
    assert torch.allclose(
        varying_stats.pairwise_positive_jaccard, torch.tensor([[1.0 / 3.0]])
    )


def test_identical_current_parent_have_zero_signed_affine_moments() -> None:
    generator = torch.Generator().manual_seed(313)
    k = torch.randn(2, 2, 9, 3, generator=generator)
    v = torch.randn(2, 2, 9, 5, generator=generator)
    mask = torch.randn(2, 2, 9, generator=generator) >= 0
    current = build_positive_affine_moments(k, v, mask)
    parent = build_positive_affine_moments(k.clone(), v.clone(), mask.clone())

    signed = subtract_affine_moments(current, parent)

    assert torch.count_nonzero(signed.base) == 0
    assert torch.count_nonzero(signed.linear) == 0
    assert signed.current_positive_mask.shape == (2, 2, 9)
    assert torch.equal(signed.current_positive_mask, signed.parent_positive_mask)


def test_negative_branch_decomposition_matches_native_elu_plus_one() -> None:
    q = torch.tensor([[[[1.0, 0.0], [-1.0, 0.0]]]])
    k = torch.tensor([[[[2.0, 0.0], [-2.0, 0.0], [0.0, 1.0]]]])
    v = torch.tensor([[[[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]]]])
    scale = 0.5

    observed = decompose_elu_plus_one_response(q, k, v, scale=scale)
    native_weights = torch.nn.functional.elu(observed.logits) + 1.0
    native_response = torch.matmul(native_weights, v)

    assert torch.allclose(observed.full_response, native_response)
    assert torch.allclose(
        observed.full_response,
        observed.positive_response + observed.negative_response,
    )
    assert torch.all(observed.negative_activation_fraction > 0)
    assert torch.all(observed.negative_activation_fraction < 1)
