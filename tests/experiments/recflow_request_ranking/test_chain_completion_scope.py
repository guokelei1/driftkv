import pytest

from experiments.recflow_request_ranking.code.verify_version_chain import (
    EXPECTED_WINDOWS,
    validate_completed_scope,
)


def complete_scope():
    return {"versions": [{"day": day, "sha256": f"model-{day}"} for day in range(26, 36)],
            "evaluations": [{"role": role, "day": day} for role, day in sorted(EXPECTED_WINDOWS)]}


def test_full_scope_includes_all_models_nine_edges_both_roles_and_extra_day():
    validate_completed_scope(complete_scope())


@pytest.mark.parametrize("missing", ["later_versions", "confirmation", "extra_window", "duplicated_window", "same_checkpoint"])
def test_cannot_claim_completion_from_partial_or_reused_evidence(missing):
    progress = complete_scope()
    if missing == "later_versions":
        progress["versions"] = progress["versions"][:2]
    elif missing == "confirmation":
        progress["evaluations"] = [row for row in progress["evaluations"] if row["role"] == 2]
    elif missing == "extra_window":
        progress["evaluations"] = [row for row in progress["evaluations"] if row["day"] != 37]
    elif missing == "duplicated_window":
        progress["evaluations"].append(progress["evaluations"][0])
    else:
        progress["versions"][-1]["sha256"] = progress["versions"][-2]["sha256"]
    with pytest.raises(ValueError, match="Completion requires"):
        validate_completed_scope(progress)
