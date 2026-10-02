"""Batch-centered residual loss equals a scaled pairwise-difference objective."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from design_one.calibrate_nonlinear import joint_logit_loss


def test_centered_residual_pairwise_value_gradient_and_shift_invariance():
    generator = torch.Generator().manual_seed(172020)
    logits = torch.randn(3, 4, generator=generator, dtype=torch.double, requires_grad=True)
    teacher = torch.randn(3, 4, generator=generator, dtype=torch.double)
    unit = torch.tensor(2.3, dtype=torch.double)
    actual = joint_logit_loss(logits, teacher, unit, centered=True)
    residual = (logits-teacher).flatten()
    explicit = (residual[:, None]-residual[None, :]).square().mean()/(2*unit.square())
    torch.testing.assert_close(actual, explicit, atol=1e-12, rtol=1e-12)
    actual_grad, = torch.autograd.grad(actual, logits, retain_graph=True)
    pairwise_grad, = torch.autograd.grad(explicit, logits)
    torch.testing.assert_close(actual_grad, pairwise_grad, atol=1e-12, rtol=1e-12)
    shifted = (logits.detach()+19.).requires_grad_(True)
    shifted_loss = joint_logit_loss(shifted, teacher, unit, centered=True)
    shifted_grad, = torch.autograd.grad(shifted_loss, shifted)
    torch.testing.assert_close(shifted_loss, actual, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(shifted_grad, actual_grad, atol=1e-12, rtol=1e-12)
    assert abs(float(actual_grad.sum())) < 1e-12
    per_user = (((logits-teacher)-(logits-teacher).mean(1, keepdim=True))/unit).square().mean()
    assert not torch.isclose(actual, per_user)
    assert torch.equal(joint_logit_loss(logits, teacher, unit), ((logits-teacher)/unit).square().mean())
