import hashlib
from dataclasses import replace

import pytest
import torch

from hstu_kvcache.migration import (
    DRAMKVUpdateDestination,
    JaggedMigratedKVBatch,
    SemanticCanaryObservation,
    Stage5CohortPreflight,
    Stage5DeviceCapacity,
    Stage5PreflightMeasurement,
    Stage5PreparedExtent,
    Stage5ProducedExtent,
    Stage5RecordRequest,
    capture_manifest_snapshot,
    manifest_present_record_ids,
    observe_semantic_canary,
    run_stage5_job,
    run_stage5_preflight,
    verify_manifest_readback,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _batch(
    record_ids: tuple[int, ...],
    target: str,
    value: float,
    anchor: str = "theta0",
    tokens: int = 2,
) -> JaggedMigratedKVBatch:
    lengths = torch.tensor(
        [tokens for _ in record_ids],
        dtype=torch.long,
    )
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)))
    shape = (2, int(offsets[-1]), 3)
    k = torch.full(shape, value, dtype=torch.float32)
    v = torch.full(shape, -value, dtype=torch.float32)
    return JaggedMigratedKVBatch(
        record_ids=record_ids,
        migration_anchor_version=anchor,
        served_kv_target=target,
        k=k,
        v=v,
        lengths=lengths,
        offsets=offsets,
    )


def _canary(
    cohort_id: str,
    passed: bool = True,
) -> SemanticCanaryObservation:
    return SemanticCanaryObservation(
        cohort_id=cohort_id,
        record_ids=(10,),
        source_version="theta0",
        target_version="theta1",
        program_sha256=_sha("program"),
        metric="kv_relative_l2",
        observed_relative_l2=0.01 if passed else 0.25,
        maximum_relative_l2=0.05,
        candidate_sha256=_sha(f"{cohort_id}-candidate"),
        reference_sha256=_sha(f"{cohort_id}-reference"),
        threshold_artifact_sha256=_sha("threshold"),
    )


def test_semantic_canary_serializes_record_ids_as_json_array() -> None:
    value = _canary("old-a").to_dict()

    assert value["record_ids"] == [10]
    assert isinstance(value["record_ids"], list)


def _cohort(
    cohort_id: str,
    passed: bool = True,
) -> Stage5CohortPreflight:
    return Stage5CohortPreflight(
        cohort_id=cohort_id,
        source_version="theta0",
        target_version="theta1",
        expected_artifact_sha256=_sha("artifact"),
        observed_artifact_sha256=_sha("artifact"),
        expected_program_sha256=_sha("program"),
        observed_program_sha256=_sha("program"),
        expected_program_shape=(2, 3, 4),
        observed_program_shape=(2, 3, 4),
        expected_threshold_artifact_sha256=_sha("threshold"),
        expected_old_record_ids=(10,),
        present_old_record_ids=(10,),
        device_capacity=(
            Stage5DeviceCapacity(
                device="cpu",
                model_and_program_bytes=10,
                old_kv_bytes=20,
                complete_new_kv_bytes=20,
                transient_bytes=5,
                allocator_margin_bytes=5,
                capacity_bytes=100,
            ),
        ),
        canary=_canary(cohort_id, passed),
        measurement=Stage5PreflightMeasurement(0.0, 0.0, 0.0, 0.0),
    )


def _requests() -> tuple[Stage5RecordRequest, ...]:
    return (
        Stage5RecordRequest(
            record_id=10,
            cohort_id="old-a",
            requested_action="migrate",
            source_version="theta0",
            target_version="theta1",
            last_exact_version="theta0",
            migration_depth=2,
            requested_reason="scheduler_migrate",
            retained_tokens=2,
            final_tokens=3,
        ),
        Stage5RecordRequest(
            record_id=11,
            cohort_id="old-a",
            requested_action="exact",
            source_version="theta0",
            target_version="theta1",
            last_exact_version="theta0",
            migration_depth=1,
            requested_reason="scheduled_exact",
            retained_tokens=2,
            final_tokens=3,
        ),
    )


def _exact_only_cohort() -> Stage5CohortPreflight:
    return Stage5CohortPreflight(
        cohort_id="cold",
        source_version="theta0",
        target_version="theta1",
        expected_artifact_sha256=_sha("unused-artifact"),
        observed_artifact_sha256=_sha("unused-artifact"),
        expected_program_sha256=None,
        observed_program_sha256=None,
        expected_program_shape=(),
        observed_program_shape=(),
        expected_threshold_artifact_sha256=None,
        expected_old_record_ids=(),
        present_old_record_ids=(),
        device_capacity=(),
        canary=None,
        measurement=Stage5PreflightMeasurement(0.0, 0.0, 0.0, 0.0),
        migration_required=False,
    )


def _old_destination():
    destination = DRAMKVUpdateDestination()
    transaction = destination.begin(
        "old-job",
        "theta0",
        (10, 11),
    )
    transaction.stage("old-00000000", _batch((10,), "theta0", 1.0))
    transaction.stage("old-00000001", _batch((11,), "theta0", 2.0))
    manifest = transaction.commit()
    snapshot = capture_manifest_snapshot(destination, manifest)
    return destination, manifest, snapshot


def _retained_producer(
    record_ids: tuple[int, ...],
    action: str,
    cohort_id: str,
) -> Stage5PreparedExtent:
    assert cohort_id == "old-a"
    return Stage5PreparedExtent(
        record_ids=record_ids,
        action=action,
        cohort_id=cohort_id,
        source_version="theta0",
        target_version="theta1",
        artifact_sha256=_sha("artifact"),
        program_sha256=_sha("program") if action == "migrate" else None,
        program_shape=(2, 3, 4) if action == "migrate" else (),
        retained_lengths=tuple(2 for _ in record_ids),
        retained_batch=_batch(
            record_ids,
            "theta1",
            2.0,
            anchor="theta0" if action == "migrate" else "theta1",
        ),
        num_layers=2,
        kv_width=3,
        dtype="float32",
    )


def _target_appender(
    prepared: Stage5PreparedExtent,
) -> Stage5ProducedExtent:
    return Stage5ProducedExtent(
        _batch(
            prepared.record_ids,
            "theta1",
            3.0 if prepared.action == "migrate" else 4.0,
            anchor="theta1",
            tokens=3,
        ),
        source_guard_hook=prepared.guard_hook,
    )


def _guard(prepared, result) -> None:
    assert prepared.retained_batch is not None
    assert prepared.retained_lengths == tuple(
        int(value) for value in prepared.retained_batch.lengths
    )
    assert result.cohort_id == prepared.cohort_id


def test_semantic_canary_uses_only_matched_kv_endpoints() -> None:
    reference = _batch((10, 11), "theta1", 2.0, anchor="theta1")
    candidate = _batch((10, 11), "theta1", 2.01)

    observation = observe_semantic_canary(
        "old-a",
        "theta0",
        "theta1",
        candidate,
        reference,
        maximum_relative_l2=0.02,
        threshold_artifact_sha256=_sha("threshold"),
        program_sha256=_sha("program"),
    )

    assert observation.labels_used is False
    assert observation.metric == "kv_relative_l2"
    assert observation.passed
    assert observation.candidate_sha256 != observation.reference_sha256


def test_preflight_resolves_all_actions_before_execution() -> None:
    report = run_stage5_preflight(
        _requests(),
        (_cohort("old-a"),),
    )

    migrate, exact = report.decisions
    assert report.all_cohorts_passed
    assert migrate.final_action == "migrate"
    assert migrate.last_exact_version_after == "theta0"
    assert migrate.migration_depth_after == 3
    assert migrate.state_kind_after == "migrated"
    assert exact.final_action == "exact"
    assert exact.last_exact_version_after == "theta1"
    assert exact.migration_depth_after == 0
    assert exact.state_kind_after == "exact"


def test_semantic_preflight_failure_routes_whole_cohort_to_exact() -> None:
    report = run_stage5_preflight(
        _requests(),
        (_cohort("old-a", passed=False),),
    )

    assert not report.all_cohorts_passed
    assert {value.final_action for value in report.decisions} == {"exact"}
    assert report.decisions[0].fallback_reason == "semantic_canary"
    assert report.decisions[1].fallback_reason is None
    assert all(
        value.last_exact_version_after == "theta1"
        and value.migration_depth_after == 0
        for value in report.decisions
    )


def test_normal_mixed_job_commits_only_full_post_append_cache() -> None:
    destination, old_manifest, snapshot = _old_destination()

    report = run_stage5_job(
        "target-job",
        _requests(),
        (_cohort("old-a"),),
        destination,
        _retained_producer,
        _target_appender,
        _guard,
        old_manifest,
        snapshot,
    )

    assert report.outcome == "committed"
    assert report.target_visible
    assert report.target_manifest is not None
    assert report.target_manifest.record_ids == (10, 11)
    assert (
        report.target_manifest.destination_manifest.metadata_sha256
        == report.target_manifest.to_dict()["lineage_sha256"]
    )
    atomic_metadata = (
        report.target_manifest.destination_manifest.to_dict()["metadata"]
    )
    assert atomic_metadata["commit_hook"] == "post_append_full_cache"
    assert [value["record_id"] for value in atomic_metadata["lineage"]] == [
        10,
        11,
    ]
    assert report.staged_extents == 2
    assert report.guard_invocations == 2
    assert report.target_readback is not None
    assert report.target_readback.passed
    assert report.target_readback.expected_records == 2
    assert report.target_readback.read_records == 2
    assert destination.manifest("theta0") == old_manifest
    assert verify_manifest_readback(
        destination,
        old_manifest,
        snapshot,
    ).passed


def test_semantic_fallback_commits_complete_exact_target() -> None:
    destination, old_manifest, snapshot = _old_destination()
    actions = []

    def retained_producer(record_ids, action, cohort_id):
        actions.append((record_ids, action, cohort_id))
        return _retained_producer(record_ids, action, cohort_id)

    report = run_stage5_job(
        "fallback-job",
        _requests(),
        (_cohort("old-a", passed=False),),
        destination,
        retained_producer,
        _target_appender,
        _guard,
        old_manifest,
        snapshot,
    )

    assert report.outcome == "committed"
    assert actions == [((10, 11), "exact", "old-a")]
    assert report.target_manifest is not None
    assert report.target_manifest.record_ids == (10, 11)
    assert {
        value.final_action for value in report.preflight.decisions
    } == {"exact"}
    assert report.preflight.decisions[0].fallback_reason == "semantic_canary"
    assert report.preflight.decisions[1].fallback_reason is None


def test_cold_exact_record_needs_no_fake_old_version_or_program_canary() -> None:
    destination = DRAMKVUpdateDestination()
    request = Stage5RecordRequest(
        record_id=12,
        cohort_id="cold",
        requested_action="exact",
        source_version="theta0",
        target_version="theta1",
        last_exact_version=None,
        migration_depth=0,
        requested_reason="cold_exact",
        retained_tokens=2,
        final_tokens=3,
    )
    events = []

    def retained(record_ids, action, cohort_id):
        events.append("retained")
        return Stage5PreparedExtent(
            record_ids=record_ids,
            action=action,
            cohort_id=cohort_id,
            source_version="theta0",
            target_version="theta1",
            artifact_sha256=_sha("unused-artifact"),
            program_sha256=None,
            program_shape=(),
            retained_lengths=(2,),
            retained_batch=_batch(
                record_ids,
                "theta1",
                4.0,
                anchor="theta1",
            ),
            num_layers=2,
            kv_width=3,
            dtype="float32",
        )

    def guard(prepared, result):
        assert prepared.guard_hook == "post_retained_prefix_pre_append"
        assert result.passed
        events.append("guard")

    def append(prepared):
        events.append("append")
        return Stage5ProducedExtent(
            _batch(
                prepared.record_ids,
                "theta1",
                5.0,
                anchor="theta1",
                tokens=3,
            ),
            source_guard_hook=prepared.guard_hook,
        )

    report = run_stage5_job(
        "cold-job",
        (request,),
        (_exact_only_cohort(),),
        destination,
        retained,
        append,
        guard,
    )

    assert events == ["retained", "guard", "append"]
    assert report.outcome == "committed"
    decision = report.preflight.decisions[0]
    assert decision.last_exact_version_before is None
    assert decision.last_exact_version_after == "theta1"


@pytest.mark.parametrize(
    ("fault", "expected_staged"),
    (("mid_job", 1), ("pre_commit", 2)),
)
def test_fault_aborts_target_and_reads_every_old_record(
    fault: str,
    expected_staged: int,
) -> None:
    destination, old_manifest, snapshot = _old_destination()

    report = run_stage5_job(
        f"{fault}-job",
        _requests(),
        (_cohort("old-a"),),
        destination,
        _retained_producer,
        _target_appender,
        _guard,
        old_manifest,
        snapshot,
        fault=fault,
    )

    assert report.outcome == "aborted"
    assert not report.target_visible
    assert not report.partial_target_visible
    assert report.target_manifest is None
    assert report.staged_extents == expected_staged
    assert report.old_readback is not None
    assert report.old_readback.passed
    assert report.old_readback.expected_records == 2
    assert report.old_readback.read_records == 2
    with pytest.raises(KeyError, match="not published"):
        destination.manifest("theta1")


def test_readback_detects_mutated_old_data_not_just_manifest_identity() -> None:
    destination, old_manifest, snapshot = _old_destination()
    destination._committed["theta0"]["old-00000000"].k.add_(1)

    report = verify_manifest_readback(
        destination,
        old_manifest,
        snapshot,
    )

    assert report.manifest_equal
    assert report.read_records == 2
    assert not report.all_checksums_equal
    assert not report.passed


def test_destination_readback_isolated_from_committed_payload() -> None:
    destination, old_manifest, snapshot = _old_destination()
    loaded = destination.load_extent("theta0", "old-00000000")
    loaded.k.add_(1)

    assert verify_manifest_readback(
        destination,
        old_manifest,
        snapshot,
    ).passed


def test_readback_rejects_snapshot_from_another_version() -> None:
    destination, old_manifest, snapshot = _old_destination()

    report = verify_manifest_readback(
        destination,
        old_manifest,
        replace(snapshot, target_version="theta-wrong"),
    )

    assert not report.passed


def test_destination_staging_isolated_from_producer_payload() -> None:
    destination = DRAMKVUpdateDestination()
    source = _batch((10,), "theta1", 1.0, anchor="theta1")
    transaction = destination.begin("isolated-job", "theta1", (10,))
    transaction.stage("isolated-extent", source)
    source.k.add_(7)
    manifest = transaction.commit()
    loaded = destination.load_extent("theta1", "isolated-extent")

    assert torch.all(loaded.k == 1)
    assert capture_manifest_snapshot(destination, manifest).records[0].finite


def test_readback_rejects_batch_anchor_that_differs_from_manifest() -> None:
    destination, old_manifest, _ = _old_destination()
    old = destination.load_extent("theta0", "old-00000000")
    destination._committed["theta0"]["old-00000000"] = replace(
        old,
        migration_anchor_version="wrong-anchor",
    )

    with pytest.raises(RuntimeError, match="differs from its manifest"):
        capture_manifest_snapshot(destination, old_manifest)
    assert manifest_present_record_ids(destination, old_manifest) == (11,)


def test_prepared_extent_binds_anchor_and_retained_lengths() -> None:
    with pytest.raises(ValueError, match="retained extent"):
        replace(
            _retained_producer((10,), "migrate", "old-a"),
            retained_lengths=(1,),
        )
    with pytest.raises(ValueError, match="retained extent"):
        replace(
            _retained_producer((11,), "exact", "old-a"),
            retained_batch=_batch(
                (11,),
                "theta1",
                1.0,
                anchor="theta0",
            ),
        )


def test_job_rejects_unbound_program_before_publication() -> None:
    destination, old_manifest, snapshot = _old_destination()

    def producer(record_ids, action, cohort_id):
        prepared = _retained_producer(record_ids, action, cohort_id)
        if action == "migrate":
            return replace(prepared, program_sha256=_sha("other-program"))
        return prepared

    with pytest.raises(RuntimeError, match="producer differs"):
        run_stage5_job(
            "wrong-program-job",
            _requests(),
            (_cohort("old-a"),),
            destination,
            producer,
            _target_appender,
            _guard,
            old_manifest,
            snapshot,
        )
    with pytest.raises(KeyError, match="not published"):
        destination.manifest("theta1")


def test_job_rejects_retained_only_output_as_post_append_cache() -> None:
    destination, old_manifest, snapshot = _old_destination()

    def retained_only(prepared):
        return Stage5ProducedExtent(
            prepared.retained_batch,
            source_guard_hook=prepared.guard_hook,
        )

    with pytest.raises(RuntimeError, match="append differs"):
        run_stage5_job(
            "retained-only-job",
            _requests(),
            (_cohort("old-a"),),
            destination,
            _retained_producer,
            retained_only,
            _guard,
            old_manifest,
            snapshot,
        )
    with pytest.raises(KeyError, match="not published"):
        destination.manifest("theta1")


def test_canary_records_must_belong_to_migration_cohort() -> None:
    with pytest.raises(ValueError, match="preflight input"):
        replace(
            _cohort("old-a"),
            canary=replace(_canary("old-a"), record_ids=(99,)),
        )


def test_job_requires_real_guard_callback() -> None:
    destination, old_manifest, snapshot = _old_destination()

    with pytest.raises(ValueError, match="guard must be callable"):
        run_stage5_job(
            "no-guard-job",
            _requests(),
            (_cohort("old-a"),),
            destination,
            _retained_producer,
            _target_appender,
            None,
            old_manifest,
            snapshot,
        )


def test_job_binds_old_manifest_version_to_migration_source() -> None:
    destination, old_manifest, snapshot = _old_destination()
    requests = tuple(
        replace(value, source_version="theta-wrong")
        for value in _requests()
    )
    cohort = replace(
        _cohort("old-a"),
        source_version="theta-wrong",
        canary=replace(
            _canary("old-a"),
            source_version="theta-wrong",
        ),
    )

    with pytest.raises(ValueError, match="provenance"):
        run_stage5_job(
            "wrong-source-job",
            requests,
            (cohort,),
            destination,
            _retained_producer,
            _target_appender,
            _guard,
            old_manifest,
            snapshot,
        )


@pytest.mark.parametrize("failed_check", ("artifact", "capacity"))
def test_job_does_not_use_exact_as_an_unsafe_fallback(
    failed_check: str,
) -> None:
    destination, old_manifest, snapshot = _old_destination()
    cohort = _cohort("old-a")
    if failed_check == "artifact":
        cohort = replace(
            cohort,
            observed_artifact_sha256=_sha("wrong-artifact"),
        )
    else:
        capacity = replace(
            cohort.device_capacity[0],
            capacity_bytes=1,
        )
        cohort = replace(cohort, device_capacity=(capacity,))

    with pytest.raises(RuntimeError, match="no safe fallback"):
        run_stage5_job(
            f"unsafe-{failed_check}-job",
            _requests(),
            (cohort,),
            destination,
            _retained_producer,
            _target_appender,
            _guard,
            old_manifest,
            snapshot,
        )
    with pytest.raises(KeyError, match="not published"):
        destination.manifest("theta1")
