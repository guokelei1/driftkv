import torch

from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.streaming.kuairand_kv_invariant import (
    _set_trainable_tail,
    kv_invariant_parameter_names,
)


def test_native_tail_parameters_preserve_prefix_kv():
    torch.manual_seed(23)
    model = HSTU(
        HSTUConfig(
            num_items=20,
            num_prediction_items=15,
            num_behaviors=3,
            hidden_size=8,
            num_layers=2,
            num_heads=2,
            head_dim=4,
            max_seq_len=6,
            input_dropout=0.0,
        )
    )
    model.eval()
    item_ids = torch.tensor([[1, 2, 3, 4, 5]])
    behaviors = torch.tensor([[1, 2, 1, 2, 1]])
    deltas = torch.tensor([[0.0, 1.0, 2.0, 1.0, 3.0]])
    before_hidden, before_cache = model(
        item_ids,
        behaviors,
        deltas,
        return_kv=True,
    )
    trainable = _set_trainable_tail(model)

    with torch.no_grad():
        for parameter in trainable:
            parameter.add_(0.1 * torch.randn_like(parameter))
    after_hidden, after_cache = model(
        item_ids,
        behaviors,
        deltas,
        return_kv=True,
    )

    assert kv_invariant_parameter_names(model) == [
        "blocks.1.attn.q_proj.weight",
        "blocks.1.attn.out_proj.weight",
        "blocks.1.gate_proj.weight",
        "final_norm.weight",
    ]
    assert torch.equal(before_cache.k, after_cache.k)
    assert torch.equal(before_cache.v, after_cache.v)
    assert not torch.equal(before_hidden, after_hidden)
    assert all(
        parameter.requires_grad
        == (name in set(kv_invariant_parameter_names(model)))
        for name, parameter in model.named_parameters()
    )
