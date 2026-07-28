from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass

ROLLOUT_BOUNDARY_PROTOCOL = (
    "cohortkv_single_config_stage4_9_rollout_boundary_v1"
)
RETAINED_PREFIX_ABI = "cohortkv_retained_prefix_rollout_abi_v1"


def _segment_hash(domain: str, values: tuple[str, ...]) -> str:
    payload = json.dumps(
        {
            "domain": domain,
            "values": values,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _suffix_prefix_overlap(
    old_values: tuple[str, ...],
    new_values: tuple[str, ...],
) -> int:
    if not old_values or not new_values:
        return 0
    separator = object()
    combined = [*new_values, separator, *old_values]
    prefix = [0] * len(combined)
    for index in range(1, len(combined)):
        length = prefix[index - 1]
        while length and combined[index] != combined[length]:
            length = prefix[length - 1]
        if combined[index] == combined[length]:
            length += 1
        prefix[index] = length
    return min(prefix[-1], len(old_values), len(new_values))


@dataclass(frozen=True)
class RetainedPrefixPlan:
    protocol: str
    record_id: int
    user_id: int
    status: str
    old_history_sha256: str | None
    target_history_sha256: str | None
    old_tokens: int
    target_prefix_tokens: int
    final_tokens: int
    potential_overlap_tokens: int
    retained_start: int
    retained_tokens: int
    evicted_tokens: int
    delta_start: int
    delta_tokens: int
    latest_tokens: int
    previous_cache_expected: bool
    previous_cache_present: bool
    missing_expected_cache: bool
    timed_retained_rebuild: bool
    migration_eligible: bool
    retained_identity_sha256: str
    delta_identity_sha256: str
    target_prefix_identity_sha256: str

    def __post_init__(self) -> None:
        allowed = {
            "reusable",
            "zero_overlap",
            "cold",
            "missing_cache",
            "short_no_prefix",
            "expired",
            "absent",
        }
        if (
            self.protocol != RETAINED_PREFIX_ABI
            or self.record_id < 0
            or self.user_id < 1
            or self.status not in allowed
            or self.old_tokens < 0
            or self.target_prefix_tokens < 0
            or self.final_tokens < 0
            or self.latest_tokens not in {0, 1}
            or self.target_prefix_tokens + self.latest_tokens
            != self.final_tokens
            or not 0
            <= self.potential_overlap_tokens
            <= min(self.old_tokens, self.target_prefix_tokens)
            or self.retained_tokens
            != (
                self.potential_overlap_tokens
                if self.status in {"reusable", "missing_cache"}
                else 0
            )
            or self.retained_start != self.old_tokens - self.retained_tokens
            or self.evicted_tokens != self.retained_start
            or self.delta_start != self.retained_tokens
            or self.delta_tokens
            != self.target_prefix_tokens - self.retained_tokens
            or (
                self.previous_cache_present
                and not self.previous_cache_expected
            )
            or self.missing_expected_cache
            != (self.status == "missing_cache")
            or self.timed_retained_rebuild
            != (
                self.status == "missing_cache"
                and self.retained_tokens > 0
            )
            or self.migration_eligible
            != (
                self.status == "reusable"
                and self.previous_cache_present
                and self.retained_tokens > 0
            )
        ):
            raise ValueError("retained-prefix rollout plan is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def plan_retained_prefix(
    record_id: int,
    user_id: int,
    old_history_identities: Sequence[str],
    target_history_identities: Sequence[str],
    old_history_sha256: str | None,
    target_history_sha256: str | None,
    previous_cache_expected: bool,
    previous_cache_present: bool,
) -> RetainedPrefixPlan:
    old_values = tuple(str(value) for value in old_history_identities)
    target_values = tuple(str(value) for value in target_history_identities)
    final_tokens = len(target_values)
    latest_tokens = int(final_tokens > 0)
    target_prefix = target_values[: final_tokens - latest_tokens]
    potential_overlap = _suffix_prefix_overlap(old_values, target_prefix)
    if final_tokens == 0:
        status = (
            "expired"
            if previous_cache_expected or previous_cache_present
            else "absent"
        )
    elif not target_prefix:
        status = "short_no_prefix"
    elif (
        previous_cache_expected
        and not previous_cache_present
        and potential_overlap > 0
    ):
        status = "missing_cache"
    elif not previous_cache_present:
        status = "cold"
    elif potential_overlap == 0:
        status = "zero_overlap"
    else:
        status = "reusable"
    retained_tokens = (
        potential_overlap
        if status in {"reusable", "missing_cache"}
        else 0
    )
    retained_values = target_prefix[:retained_tokens]
    delta_values = target_prefix[retained_tokens:]
    return RetainedPrefixPlan(
        protocol=RETAINED_PREFIX_ABI,
        record_id=int(record_id),
        user_id=int(user_id),
        status=status,
        old_history_sha256=old_history_sha256,
        target_history_sha256=target_history_sha256,
        old_tokens=len(old_values),
        target_prefix_tokens=len(target_prefix),
        final_tokens=final_tokens,
        potential_overlap_tokens=potential_overlap,
        retained_start=len(old_values) - retained_tokens,
        retained_tokens=retained_tokens,
        evicted_tokens=len(old_values) - retained_tokens,
        delta_start=retained_tokens,
        delta_tokens=len(delta_values),
        latest_tokens=latest_tokens,
        previous_cache_expected=bool(previous_cache_expected),
        previous_cache_present=bool(previous_cache_present),
        missing_expected_cache=status == "missing_cache",
        timed_retained_rebuild=status == "missing_cache",
        migration_eligible=status == "reusable",
        retained_identity_sha256=_segment_hash(
            "retained-prefix-v1",
            retained_values,
        ),
        delta_identity_sha256=_segment_hash(
            "post-migration-delta-v1",
            delta_values,
        ),
        target_prefix_identity_sha256=_segment_hash(
            "target-prefix-v1",
            target_prefix,
        ),
    )


def retained_population_sha256(
    plans: Sequence[RetainedPrefixPlan],
) -> str:
    payload = json.dumps(
        [
            {
                "record_id": value.record_id,
                "retained_tokens": value.retained_tokens,
                "retained_identity_sha256": value.retained_identity_sha256,
            }
            for value in sorted(plans, key=lambda item: item.record_id)
            if value.migration_eligible or value.timed_retained_rebuild
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
