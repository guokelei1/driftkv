"""Small numerical checks for the isolated nonlinear token-map candidate."""
import copy
from pathlib import Path
import sys

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "src"),
               str(Path(__file__).resolve().parents[2] / "scripts")]
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.adaptation.reader import history_read
from hstu_kvcache.read_correction_v4.token_read import TokenReadCorrection, score_token_corrected
from hstu_kvcache.read_correction_v4.nonlinear import NonlinearTokenReadCorrection
from read_correction_v4.history_conditioned.fit_nonlinear import fit_nonlinear, normalized_losses, _finish_context
from read_correction_2026_09.v2.history_conditioned.fit import project_context


def model(seed):
    torch.manual_seed(seed)
    return HSTU(HSTUConfig(num_items=32, num_behaviors=3, hidden_size=32, num_heads=1,
        num_layers=2, max_seq_len=8, input_dropout=0., temporal_num_freqs=16)).eval().requires_grad_(False)


def test_zero_residual_preserves_nonzero_base_producer_gate_and_cache():
    torch.manual_seed(17)
    base = TokenReadCorrection(2, 3, query_affine=True).double()
    with torch.no_grad():
        base.map_weight.normal_(std=.02)
        base.map_bias.normal_(std=.01)
        base.query_correction.weight.normal_(std=.01)
    module = NonlinearTokenReadCorrection(**base.get_config()).double().initialize_base(base)
    module.set_residual_scale(torch.logspace(-5, 2, 12).double())
    k, v = torch.randn(2, 7, 6).double(), torch.randn(2, 7, 6).double()
    saved_k, saved_v = k.clone(), v.clone()
    count, old = torch.tensor([7, 5]), torch.tensor([3, 0])
    actual, expected = module.map_tokens(k, v, count, old), base.map_tokens(k, v, count, old)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.equal(k, saved_k) and torch.equal(v, saved_v)
    assert not module.map_weight.requires_grad and not module.query_correction.weight.requires_grad


def test_loss_matches_physical_teacher_read_and_has_trainable_residual():
    torch.set_num_threads(1)
    torch.manual_seed(9)
    attention = model(7).blocks[0].attn
    module = NonlinearTokenReadCorrection(1, 32, nonlinear_width=8)
    teacher = copy.deepcopy(module)
    with torch.no_grad():
        teacher.output.weight.normal_(std=.04)
        teacher.output.bias.normal_(std=.01)
    x = torch.randn(3, 5, 64)
    base = torch.randn(3, 5, 64)
    q, counts = torch.randn(3, 1, 4, 32), torch.tensor([5, 3, 4])
    valid = torch.arange(5)[None] < counts[:, None]
    delta = teacher.residual_delta(x).detach()
    k, v = (base + delta * valid[..., None]).split(32, -1)
    target = history_read(attention, q, k, v, count=valid).detach()
    unit = torch.tensor([2.])
    zero, _, _ = normalized_losses(teacher, x, base, q, delta, target, counts, attention, unit)
    torch.testing.assert_close(zero, torch.tensor(0.))
    before = normalized_losses(module, x, base, q, delta, target, counts, attention, unit)[0]
    optimizer = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad], lr=.003)
    for step in range(8):
        optimizer.zero_grad(set_to_none=True)
        loss = normalized_losses(module, x, base, q, delta, target, counts, attention, unit)[0]
        loss.backward()
        if step == 1:
            assert module.encoder.weight.grad.abs().sum() > 0
        optimizer.step()
    after = normalized_losses(module, x, base, q, delta, target, counts, attention, unit)[0]
    assert after < before
    assert module.map_weight.grad is None


def test_zero_epoch_fit_matches_full_native_override_and_keeps_units_heldout():
    torch.set_num_threads(1)
    parent, current = model(17), model(29)
    rows = {}
    with torch.no_grad():
        for uid in range(6):
            length = 3 if uid % 2 else 5
            items = torch.arange(1 + uid, 1 + uid + length)[None]
            data = items, torch.ones_like(items), torch.zeros_like(items).float()
            rows[uid] = {"parent": parent.compute_kv(*data), "teacher": current.compute_kv(*data),
                         "candidates": torch.tensor([20, 24, 28]), "query_delta": 2.}
    bases = [TokenReadCorrection(1, 32) for _ in current.blocks]
    with torch.no_grad():
        for base in bases:
            base.map_weight.normal_(std=.001)
    config = {"attention_backend": "torch", "nonlinear_epochs": 0, "seed": 17}
    modules, stats = fit_nonlinear(current, rows, [0, 1, 2, 3], [4, 5], bases,
                                    config=config, device="cpu", scale="medium", batch_size=2)
    assert stats["cost"]["nonlinear_train_flops_estimate"] == 0
    assert all(layer["selected_epoch"] == 0 for layer in stats["layers"])
    cache = rows[0]["parent"]
    candidates, dt, counts = rows[0]["candidates"][None], torch.tensor([2.]), torch.tensor([cache.seq_len])
    expected = score_token_corrected(current, cache, candidates, dt, bases, counts)[0]
    actual, trace = score_token_corrected(current, cache, candidates, dt, modules, counts, trace=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    hidden = current.embed_query_tokens(candidates, dt)
    for layer, block in enumerate(current.blocks):
        context = project_context(block, hidden, cache.k[layer], cache.v[layer])
        torch.testing.assert_close(context["query"], trace.queries[layer], rtol=0, atol=0)
        hidden = _finish_context(block, context, cache.k[layer], cache.v[layer], counts, modules[layer])
        torch.testing.assert_close(hidden, trace.layer_outputs[layer], rtol=0, atol=0)
    altered = copy.deepcopy(rows)
    for uid in (4, 5):
        altered[uid]["teacher"].k.mul_(10.)
        altered[uid]["teacher"].v.mul_(10.)
    _, changed = fit_nonlinear(current, altered, [0, 1, 2, 3], [4, 5], bases,
                                config=config, device="cpu", scale="medium", batch_size=2)
    for original, changed_layer in zip(stats["layers"], changed["layers"]):
        assert original["output_scale_per_coordinate"] == changed_layer["output_scale_per_coordinate"]
        assert original["read_scale_per_head"] == changed_layer["read_scale_per_head"]
