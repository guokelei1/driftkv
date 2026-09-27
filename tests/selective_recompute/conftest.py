import pytest
import torch

from hstu_kvcache.models import HSTU, HSTUConfig


@pytest.fixture
def model_factory():
    def make(seed=17, *, reference=False, diagonal="inclusive", layers=3):
        torch.manual_seed(seed)
        return HSTU(HSTUConfig(
            num_items=40, num_behaviors=3, hidden_size=12, num_heads=3,
            num_layers=layers, max_seq_len=32, input_dropout=0.2,
            relative_position_bias=True, causal_diagonal=diagonal,
            block_variant="hstu_reference" if reference else "legacy",
            activation="silu" if reference else "elu_plus1",
        )).eval()
    return make


@pytest.fixture
def raw_events():
    return (
        torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]),
        torch.tensor([[1, 2, 1, 2, 1, 2], [2, 1, 2, 1, 2, 1]]),
        torch.tensor([[0., 2., 3., 1., 4., 2.], [0., 1., 5., 2., 1., 3.]]),
    )
