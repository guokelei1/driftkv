import torch

from hstu_kvcache.drift import dtheta_as_dict, ground_truth_drift, naive_per_user_jvp
from hstu_kvcache.models import HSTU, HSTUConfig
from hstu_kvcache.streaming import model_params_vec


def _setup(device="cpu"):
    torch.manual_seed(0)
    cfg = HSTUConfig(
        num_items=50, num_behaviors=8, hidden_size=32, num_layers=2,
        num_heads=2, head_dim=16, max_seq_len=32,
    )
    model = HSTU(cfg).to(device)
    batch = {
        "item_ids": torch.randint(1, 51, (2, 8), device=device),
        "behaviors": torch.randint(0, 9, (2, 8), device=device),
        "time_deltas": torch.rand(2, 8, device=device) * 100,
    }
    return model, batch


def test_jvp_returns_drift_vector():
    model, batch = _setup()
    theta = model_params_vec(model)
    dtheta = torch.randn_like(theta) * 0.01
    est = naive_per_user_jvp(model, batch, dtheta_as_dict(model, dtheta), torch.device("cpu"), warmup=1, repeats=2)
    assert est.drift_vec.numel() == est.kv_numel
    assert est.drift_vec.norm().item() > 0


def test_ground_truth_drift_zero_at_zero_dtheta():
    model, batch = _setup()
    theta = model_params_vec(model)
    zero_dtheta = torch.zeros_like(theta)
    kv0, kv1, metrics = ground_truth_drift(model, batch, zero_dtheta, torch.device("cpu"))
    assert metrics["drift_l2"] < 1e-4


def test_ground_truth_drift_nonzero():
    model, batch = _setup()
    theta = model_params_vec(model)
    dtheta = torch.randn_like(theta) * 0.1
    _, _, metrics = ground_truth_drift(model, batch, dtheta, torch.device("cpu"))
    assert metrics["drift_l2"] > 0.1


def test_linearization_tracks_direction():
    """At small dtheta, ||J.dtheta|| should be within an order of magnitude of ||gt drift||."""
    model, batch = _setup()
    theta = model_params_vec(model)
    dtheta = torch.randn_like(theta) * 0.005
    est = naive_per_user_jvp(model, batch, dtheta_as_dict(model, dtheta), torch.device("cpu"), warmup=1, repeats=2)
    _, _, gt = ground_truth_drift(model, batch, dtheta, torch.device("cpu"))
    ratio = est.drift_vec.norm().item() / (gt["drift_l2"] + 1e-8)
    assert 0.1 < ratio < 10.0
