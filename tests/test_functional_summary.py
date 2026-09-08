"""K-V pairing information and subtractive maintenance, independent references."""

import torch

from hstu_kvcache.adaptation.functional_summary import FunctionalSummary


def test_pairing_counterexample_and_incremental_eviction():
    probes = torch.tensor([[[[0.], [1.]]]])
    k = torch.tensor([[[1.], [-1.]]])
    v = k.clone()
    producers = torch.tensor([0, 0])
    plus, minus = FunctionalSummary(probes, 1), FunctionalSummary(probes, 1)
    plus.update(k, v, producers)
    minus.update(k, -v, producers)
    expected = 2 - torch.exp(torch.tensor(-1.))
    torch.testing.assert_close(plus.responses[0, 0, 0, :, 0], torch.stack((expected*0, expected)))
    torch.testing.assert_close(minus.responses, -plus.responses)
    # Remove the actual first event, append a new producer, compare full scan.
    plus.update(k[:, :1], v[:, :1], producers[:1], sign=-1)
    new_k, new_v = torch.tensor([[[.5]]]), torch.tensor([[[3.]]])
    plus.update(new_k, new_v, torch.tensor([2]))
    reference = FunctionalSummary(probes, 1)
    reference.update(torch.cat((k[:, 1:], new_k), 1), torch.cat((v[:, 1:], new_v), 1), torch.tensor([0, 2]))
    torch.testing.assert_close(plus.responses, reference.responses)
    assert plus.counts.tolist() == reference.counts.tolist() == [1, 0, 1, 0, 0, 0]
    plus.update(k[:, 1:], v[:, 1:], producers[:1], sign=-1)
    assert not torch.count_nonzero(plus.responses[0])
