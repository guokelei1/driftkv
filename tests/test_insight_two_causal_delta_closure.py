from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hstu_kvcache.models import HSTU, HSTUConfig, HSTUKVCache
from insight_two.address_response_memory import AddressLandmarkSelection
from insight_two.causal_delta_closure import (
    build_causal_delta_closure,
    chronological_carrier_partition,
)
from insight_two.signed_response_memory import intervene_oracle_signed_response_memory


def _model(seed: int, *, length: int = 16) -> HSTU:
    torch.manual_seed(seed)
    return HSTU(
        HSTUConfig(
            num_items=128,
            num_behaviors=3,
            hidden_size=16,
            num_layers=3,
            num_heads=2,
            head_dim=8,
            max_seq_len=length,
            temporal_num_freqs=2,
            input_dropout=0.0,
            activation="elu_plus1",
            block_variant="legacy",
            relative_position_bias=False,
        )
    ).eval()


def _history(length: int = 16) -> tuple[torch.Tensor, ...]:
    items = (torch.arange(length).reshape(1, length) + 1).long()
    actions = (torch.arange(length).reshape(1, length) % 2 + 1).long()
    deltas = torch.arange(length).reshape(1, length).float()
    return items, actions, deltas


def _selection(positions: torch.Tensor, length: int) -> AddressLandmarkSelection:
    positions = positions.long()
    # Assign each source row to its nearest selected chronological position.
    distances = (torch.arange(length)[:, None] - positions[None, :]).abs()
    assignments = distances.argmin(dim=1)
    assignments[positions] = torch.arange(positions.numel())
    masses = torch.bincount(assignments, minlength=positions.numel())
    return AddressLandmarkSelection(
        source_length=length,
        selected_positions=positions,
        cluster_masses=masses,
        assignments=assignments,
    )


def test_partition_reorders_positions_masses_and_assignments_together() -> None:
    selection = AddressLandmarkSelection(
        source_length=6,
        selected_positions=torch.tensor([4, 1, 5]),
        cluster_masses=torch.tensor([2, 3, 1]),
        assignments=torch.tensor([1, 1, 1, 0, 0, 2]),
    )
    partition = chronological_carrier_partition(selection)
    assert partition.positions.tolist() == [1, 4, 5]
    assert partition.masses.tolist() == [3, 2, 1]
    assert partition.assignments.tolist() == [0, 0, 0, 1, 1, 2]


def test_full_support_recursive_closure_reconstructs_exact_current() -> None:
    length = 16
    parent = _model(811, length=length)
    current = _model(821, length=length)
    items, actions, deltas = _history(length)
    parent_cache = parent.compute_kv(items, actions, deltas)
    exact_cache = current.compute_kv(items, actions, deltas)
    identity = _selection(torch.arange(length), length)

    recursive = build_causal_delta_closure(
        current, parent_cache, items, actions, deltas, identity
    )
    assert torch.allclose(
        recursive.current_carriers.k, exact_cache.k, atol=2e-5, rtol=2e-5
    )
    assert torch.allclose(
        recursive.current_carriers.v, exact_cache.v, atol=2e-5, rtol=2e-5
    )

    candidates = torch.tensor([[61, 62, 63, 64]])
    query_delta = torch.tensor([21.0])
    exact_scores, _ = current.observe_cc_reuse(
        exact_cache, candidates, query_delta
    )
    corrected = intervene_oracle_signed_response_memory(
        current,
        parent_cache,
        recursive.memory,
        candidates,
        query_delta,
    )
    assert torch.allclose(corrected.scores, exact_scores, atol=5e-5, rtol=5e-5)


def test_recursive_term_is_the_dependency_closure_not_just_the_selector() -> None:
    length = 16
    parent = _model(823, length=length)
    current = _model(827, length=length)
    items, actions, deltas = _history(length)
    parent_cache = parent.compute_kv(items, actions, deltas)
    exact_cache = current.compute_kv(items, actions, deltas)
    identity = _selection(torch.arange(length), length)
    recursive = build_causal_delta_closure(
        current, parent_cache, items, actions, deltas, identity,
        recursive_delta=True,
    )
    independent = build_causal_delta_closure(
        current, parent_cache, items, actions, deltas, identity,
        recursive_delta=False,
    )
    assert torch.allclose(recursive.current_carriers.k, exact_cache.k, atol=2e-5, rtol=2e-5)
    assert not torch.allclose(independent.current_carriers.k[2], exact_cache.k[2])


def test_closure_uses_no_exact_current_cache_and_is_suffix_causal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    length = 16
    parent = _model(829, length=length)
    current = _model(839, length=length)
    items, actions, deltas = _history(length)
    parent_cache = parent.compute_kv(items, actions, deltas)
    selection = _selection(torch.tensor([1, 5, 9, 13]), length)

    def forbidden(*args, **kwargs):
        raise AssertionError("causal closure must not request Current Exact cache")

    monkeypatch.setattr(current, "compute_kv", forbidden)
    original = build_causal_delta_closure(
        current, parent_cache, items, actions, deltas, selection
    )

    changed = HSTUKVCache(
        k=parent_cache.k.clone(),
        v=parent_cache.v.clone(),
        seq_len=parent_cache.seq_len,
    )
    changed.k[:, :, 9:].add_(37.0)
    changed.v[:, :, 9:].sub_(19.0)
    causal = build_causal_delta_closure(
        current, changed, items, actions, deltas, selection
    )
    # Carriers at positions 1 and 5 cannot observe a Parent suffix starting at 9.
    assert torch.equal(
        original.current_carriers.k[:, :, :2], causal.current_carriers.k[:, :, :2]
    )
    assert torch.equal(
        original.current_carriers.v[:, :, :2], causal.current_carriers.v[:, :, :2]
    )


def test_invalid_future_center_mass_is_not_read_by_an_earlier_carrier() -> None:
    length = 16
    parent = _model(853, length=length)
    current = _model(857, length=length)
    items, actions, deltas = _history(length)
    parent_cache = parent.compute_kv(items, actions, deltas)
    # The center at 13 owns several early rows, but it is unavailable to the
    # carrier at 5.  The audit fraction therefore remains below one.
    selection = AddressLandmarkSelection(
        source_length=length,
        selected_positions=torch.tensor([1, 5, 13]),
        cluster_masses=torch.tensor([2, 3, 11]),
        assignments=torch.tensor([0, 0, 1, 1, 2, 1] + [2] * 10),
    )
    built = build_causal_delta_closure(
        current, parent_cache, items, actions, deltas, selection
    )
    assert built.represented_prefix_fractions[1] < 1.0
    assert built.represented_prefix_fractions[2] <= 1.0
