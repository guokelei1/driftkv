from pathlib import Path
import sys

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[3]/"scripts"),
               str(Path(__file__).resolve().parents[3]/"src")]
from hstu_kvcache.read_correction import QueryCorrection
from read_correction_2026_09.v2.calibrate import mixed_candidates
from read_correction_2026_09.v2.query_only.fit import (
    OutputUnits, distillation_loss, loss_statistics,
)


def test_recent_candidates_are_unique_legal_and_reproducible():
    history = np.asarray([2,0,3,99,2,4,3])
    first = mixed_candidates(17,history,known=50,count=6)
    assert first[:3].tolist() == [3,4,2]
    assert len(set(first)) == 6 and min(first)>0 and max(first)<50
    assert np.array_equal(first,mixed_candidates(17,history,known=50,count=6))


def test_centered_distillation_is_pairwise_difference_error():
    teacher=torch.tensor([[.1,.7,.4],[-.3,.2,.9]])
    student=torch.tensor([[.3,.8,.2],[-.2,.1,.6]],requires_grad=True)
    loss,centered,_=distillation_loss(student,teacher,loss_statistics(teacher))
    errors=student-teacher
    pairwise=(errors[:,:,None]-errors[:,None,:]).square().mean()
    torch.testing.assert_close(2*centered,pairwise)
    loss.backward()
    assert torch.isfinite(student.grad).all()


def test_output_units_foldback_preserves_physical_correction():
    torch.manual_seed(17)
    module=QueryCorrection(2,3)
    with torch.no_grad():
        module.weight.normal_();module.bias.normal_()
    q=torch.randn(4,2,5,3)
    counts=torch.tensor([3,4,7,9])
    expected=module(q,None,None,counts).detach()
    wrapper=OutputUnits(module,torch.tensor([1e-4,1e4]))
    torch.testing.assert_close(wrapper(q,None,None,counts),expected,rtol=3e-6,atol=1e-6)
    torch.testing.assert_close(wrapper.folded()(q,None,None,counts),expected,rtol=3e-6,atol=1e-6)
