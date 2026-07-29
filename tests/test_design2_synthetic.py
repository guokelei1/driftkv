import hashlib
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from hstu_kvcache.migration import (
    D2ActionCounts,
    D2ActionPlan,
    D2ActionProvenance,
    D2ActionRecord,
    D2EmbeddingLookupCounter,
    D2WavePlan,
    build_d2_phase_ledger,
    build_retained_history_batch,
    canonical_sha256,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _record(
    record_id: int,
    reason: str,
) -> D2ActionRecord:
    if reason == "migrate":
        values = {
            "requested_action": "compiled",
            "old_tokens": 3,
            "retained_start": 1,
            "retained_tokens": 2,
            "delta_start": 2,
            "delta_tokens": 1,
            "target_prefix_tokens": 3,
            "latest_tokens": 1,
            "final_tokens": 4,
            "last_exact_version": "theta0",
            "migration_depth": 1,
            "previous_cache_expected": True,
            "previous_cache_present": True,
            "old_history_sha256": _sha(f"old-{record_id}"),
        }
    elif reason == "scheduled_exact":
        values = {
            "requested_action": "exact",
            "old_tokens": 2,
            "retained_start": 0,
            "retained_tokens": 2,
            "delta_start": 2,
            "delta_tokens": 1,
            "target_prefix_tokens": 3,
            "latest_tokens": 1,
            "final_tokens": 4,
            "last_exact_version": "theta0",
            "migration_depth": 1,
            "previous_cache_expected": True,
            "previous_cache_present": True,
            "old_history_sha256": _sha(f"old-{record_id}"),
        }
    else:
        values = {
            "requested_action": "exact",
            "old_tokens": 0,
            "retained_start": 0,
            "retained_tokens": 0,
            "delta_start": 0,
            "delta_tokens": 2,
            "target_prefix_tokens": 2,
            "latest_tokens": 1,
            "final_tokens": 3,
            "last_exact_version": None,
            "migration_depth": 0,
            "previous_cache_expected": False,
            "previous_cache_present": False,
            "old_history_sha256": None,
        }
    return D2ActionRecord(
        record_id=record_id,
        prepared_user_id=record_id + 1,
        requested_reason=reason,
        target_history_sha256=_sha(f"target-{record_id}"),
        retained_identity_sha256=_sha(f"retained-{record_id}"),
        delta_identity_sha256=_sha(f"delta-{record_id}"),
        target_prefix_identity_sha256=_sha(f"prefix-{record_id}"),
        **values,
    )


@pytest.fixture
def synthetic_plan() -> D2ActionPlan:
    records = (
        _record(0, "migrate"),
        _record(1, "scheduled_exact"),
        _record(2, "natural_exact"),
    )
    partition = canonical_sha256(
        {
            "migrate_ids": [0],
            "scheduled_exact_ids": [1],
            "natural_exact_ids": [2],
        }
    )
    return D2ActionPlan(
        source_version="theta1",
        target_version="theta2",
        producer="synthetic_test",
        policy="fixed",
        provenance=D2ActionProvenance(
            artifact="synthetic.json",
            artifact_sha256=_sha("artifact"),
            artifact_protocol=(
                "cohortkv_single_config_stage4_9_same_device_confirmation_v2"
            ),
            step_index=1,
            action_partition_sha256=partition,
            lineage_sha256=_sha("lineage"),
            prepared_data="synthetic.npz",
            prepared_data_sha256=_sha("prepared"),
            manifest_content_sha256=_sha("manifest"),
            target_window_content_sha256=_sha("window"),
        ),
        records=records,
        counts=D2ActionCounts(
            compiled=1,
            scheduled_exact=1,
            natural_exact=1,
            records=3,
        ),
    )


def test_synthetic_plan_roundtrip_wave_binding_and_ledger(
    synthetic_plan: D2ActionPlan,
    tmp_path,
) -> None:
    path = tmp_path / "plan.json"
    synthetic_plan.write(path)
    loaded = D2ActionPlan.load(path)
    assert loaded == synthetic_plan
    wave = D2WavePlan.single_rank(loaded, "synthetic-wave")
    wave.validate_against_action_plan(loaded)
    requests = wave.to_stage5_requests()
    assert [value.requested_action for value in requests] == [
        "migrate",
        "exact",
        "exact",
    ]
    assert requests[-1].retained_tokens == 0
    ledger = build_d2_phase_ledger(loaded, embedding_dim=4)
    assert ledger.boundaries["retained_prefix"][
        "mixed_lookup_tokens"
    ] == 2
    assert ledger.boundaries["integrated_post_append"][
        "mixed_lookup_tokens"
    ] == 9
    with pytest.raises(ValueError):
        replace(
            wave,
            action_plan_sha256="0" * 64,
        ).validate_against_action_plan(loaded)


def test_synthetic_plan_rejects_unsafe_cache_and_depth_contracts(
    synthetic_plan: D2ActionPlan,
) -> None:
    with pytest.raises(ValueError):
        replace(
            synthetic_plan.records[1],
            previous_cache_present=False,
        )
    bad = replace(
        synthetic_plan.records[0],
        migration_depth=0,
    )
    with pytest.raises(ValueError):
        replace(
            synthetic_plan,
            records=(bad, *synthetic_plan.records[1:]),
        )


def test_lookup_counter_requires_phase_and_records_padding() -> None:
    embedding = nn.Embedding(8, 4, padding_idx=0)
    with D2EmbeddingLookupCounter(embedding) as counter:
        with pytest.raises(RuntimeError):
            embedding(torch.tensor([1]))
        with counter.phase("compiled_retained", 0):
            pass
        with counter.phase("exact_prefix", 2):
            embedding(torch.tensor([[1, 0, 2]]))
        values = {
            value.phase: value for value in counter.observations()
        }
    assert values["compiled_retained"].lookup_calls == 0
    assert values["exact_prefix"].lookup_calls == 1
    assert values["exact_prefix"].logical_lookup_tokens == 2
    assert values["exact_prefix"].padded_lookup_elements == 3
    assert values["exact_prefix"].nonpadding_lookup_elements == 2


def test_retained_history_batch_is_library_owned() -> None:
    class History:
        def __init__(self) -> None:
            self.item_ids = np.array([1, 2, 3], dtype=np.int64)
            self.behaviors = np.array([4, 5, 6], dtype=np.int64)
            self.time_deltas = np.array(
                [0.0, 1.0, 2.0],
                dtype=np.float32,
            )

        def __len__(self) -> int:
            return len(self.item_ids)

    history = History()
    batch = build_retained_history_batch(
        record_ids=(7,),
        migration_anchor_version="theta1",
        histories=(history,),
        retained_tokens=(2,),
        device="cpu",
    )
    assert batch.record_ids == (7,)
    assert batch.lengths.tolist() == [2]
    assert batch.item_ids.tolist() == [[1, 2]]
