import torch

from hstu_kvcache.migration.recursive_d1 import (
    RecursiveBatchState,
    action_plan_document,
    fit_rollout_aware_direct_program,
    mix_exact_cache,
    rollout_stability_certificate,
    select_depth_balanced_tokens,
    storage_cache,
    token_balanced_renewal,
    update_lineage,
)
from hstu_kvcache.migration.stage45_oldkv import DirectOldKVProgram
from hstu_kvcache.migration.xp_d1_quality import apply_direct_oldkv
from hstu_kvcache.models import HSTUKVCache


def _cache(joined: torch.Tensor) -> HSTUKVCache:
    width = joined.shape[-1] // 2
    return HSTUKVCache(
        k=joined[..., :width].contiguous(),
        v=joined[..., width:].contiguous(),
        seq_len=joined.shape[2],
    )


def test_token_balanced_renewal_rotates_with_integer_cap() -> None:
    records = [(index, 79, "length0") for index in range(4096)]
    observed = set()
    for edge in range(10):
        exact, ledger = token_balanced_renewal(
            records,
            colors=10,
            edge_ordinal=edge,
            salt="test-renewal",
        )
        assert ledger["within_integer_token_cap"] is True
        assert ledger["scheduled_exact_valid_tokens"] <= ledger[
            "integer_token_cap"
        ]
        assert ledger["actual_valid_token_fraction"] <= (
            0.1 + 1 / 4096
        )
        assert not observed.intersection(exact)
        observed.update(exact)
    assert observed == set(range(4096))


def test_depth_balanced_sampling_covers_each_available_depth() -> None:
    record_ids = (torch.tensor([1, 2, 3, 4]),)
    lengths = (torch.tensor([4, 4, 4, 4]),)
    depths = (torch.tensor([0, 1, 2, 3]),)
    selected, ledger = select_depth_balanced_tokens(
        record_ids,
        lengths,
        depths,
        maximum_global_tokens=8,
        seed=17,
    )
    assert len(selected) == 8
    assert ledger["selected_tokens_by_depth"] == {
        "0": 2,
        "1": 2,
        "2": 2,
        "3": 2,
    }


def test_ract_direct_fit_reduces_paired_objective() -> None:
    generator = torch.Generator().manual_seed(23)
    source_joined = torch.randn(1, 6, 5, 4, generator=generator)
    error_direction = torch.tensor([0.2, -0.1, 0.05, -0.15])
    deployed_joined = source_joined + error_direction
    target_joined = source_joined @ torch.tensor(
        [
            [1.05, 0.02, 0.00, 0.01],
            [0.00, 0.97, 0.03, 0.00],
            [0.01, 0.00, 1.02, 0.02],
            [0.00, 0.01, 0.00, 0.99],
        ]
    )
    base = DirectOldKVProgram(
        source_version="theta1",
        target_version="theta2",
        weights=torch.eye(4).unsqueeze(0).half().contiguous(),
        biases=torch.zeros(1, 4, dtype=torch.float16),
    )
    exact_source = _cache(source_joined.half())
    deployed_source = _cache(deployed_joined.half())
    exact_target = _cache(target_joined.half())
    lengths = (torch.full((6,), 5, dtype=torch.int64),)
    indices = torch.arange(30, dtype=torch.int64)

    fitted_float, fitted_half, metrics = fit_rollout_aware_direct_program(
        base,
        (exact_source,),
        (deployed_source,),
        (exact_target,),
        lengths,
        indices,
        mode="ract_kv",
        rank=2,
        ridge=1e-3,
        seed=31,
        device=torch.device("cpu"),
    )

    base_deployed = apply_direct_oldkv(base, deployed_source)
    fitted_deployed = apply_direct_oldkv(fitted_float, deployed_source)
    base_error = (
        torch.cat((base_deployed.k, base_deployed.v), dim=-1).float()
        - target_joined
    ).square().mean()
    fitted_error = (
        torch.cat((fitted_deployed.k, fitted_deployed.v), dim=-1).float()
        - target_joined
    ).square().mean()
    assert fitted_error < base_error
    assert fitted_half.weights.dtype == torch.float16
    assert metrics["fit_mode"] == "ract_kv"
    assert metrics["labels_used"] is False

    certificate = rollout_stability_certificate(
        fitted_float,
        fitted_half,
        (exact_source,),
        (deployed_source,),
        (exact_target,),
        lengths,
        (torch.ones(6, dtype=torch.int64),),
        target_ratio=1.0,
        hard_ratio=2.0,
        device=torch.device("cpu"),
    )
    assert certificate["maximum_incoming_depth"] == 1
    assert certificate["labels_used"] is False
    assert certificate["rows"]


def test_mixed_cache_lineage_and_action_plan_are_bound() -> None:
    compiled = _cache(torch.ones(1, 3, 2, 4, dtype=torch.float16))
    exact = _cache(torch.zeros(1, 3, 2, 4, dtype=torch.float16))
    mixed = mix_exact_cache(
        compiled,
        exact,
        torch.tensor([True, False, False]),
    )
    torch.testing.assert_close(mixed.k[:, 0], exact.k[:, 0])
    torch.testing.assert_close(mixed.k[:, 1], compiled.k[:, 1])
    state = RecursiveBatchState(
        cache=storage_cache(compiled),
        record_ids=torch.tensor([10, 11, -1]),
        depths=torch.tensor([2, 1, 0]),
        last_exact_versions=torch.tensor([1, 1, -1]),
    )
    depths, last_exact, rows = update_lineage(
        state,
        target_version=3,
        exact_record_ids={10},
        action_for_nonexact="compiled",
    )
    assert depths.tolist() == [0, 2, 0]
    assert last_exact.tolist() == [3, 1, -1]
    plan = action_plan_document(
        method="ract_kv_exact10",
        source_version=2,
        target_version=3,
        prefix_tokens=87,
        program_sha256="program",
        renewal={"actual_valid_token_fraction": 0.1},
        fallback_all_exact=False,
        rows=rows,
        input_lineage_sha256="input",
        output_cache_state_sha256="output-cache",
    )
    assert plan["record_count"] == 2
    assert plan["single_current_serving_model"] is True
    assert plan["output_lineage_sha256"]
