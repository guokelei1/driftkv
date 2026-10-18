"""The new single-pass bootstrap must preserve the retained estimator."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from partial_refresh_2026_10.analyze import summarize
from query_relay_2026_10.aggregate import aggregate, joint_quality, within_round
from query_relay_2026_10.analyze_ranking import evaluate_groups


def artificial_matched_panel():
    rng = np.random.default_rng(21)
    rows = []
    # UID8 is deliberately idle. Some tied candidate logits test tie handling.
    for uid in range(1, 8):
        for version in ("v3", "v4", "v5"):
            for round_index in range(3):
                for candidate in range(4):
                    label = int(candidate == 0)
                    reuse = round(float(rng.normal()), 1)
                    full = reuse + .75 * label
                    dynamic = reuse + .6 * label + .08 * (uid % 3 - 1)
                    mean = reuse + .35 * label + .13 * (round_index - 1)
                    if uid == 2 and round_index == 1:
                        reuse = full = dynamic = mean = .2
                    rows.append(dict(request_id=f"toy:{uid}:{version}:{round_index}:{candidate}",
                        uid=uid, version=version, label=label, reuse_logit=reuse,
                        full_logit=full, partial_logit=dynamic, mean_gate_logit=mean,
                        first_request_rebuild_logit=full))
    rng.shuffle(rows)
    return pa.Table.from_pylist(rows), list(range(1, 9)), ("v3", "v4", "v5")


def test_single_pass_matches_retained_paired_bootstrap():
    table, users, releases = artificial_matched_panel()
    dynamic, mean, difference = joint_quality(table, users, releases=releases,
        repetitions=1000, seed=17)
    original = summarize(table, users, repetitions=1000, seed=17,
                         releases=releases, preservation_threshold=.8)
    mean_table = table.set_column(table.schema.get_field_index("partial_logit"),
                                 "partial_logit", table["mean_gate_logit"])
    original_mean = summarize(mean_table, users, repetitions=1000, seed=17,
                              releases=releases, preservation_threshold=.8)
    comparison_table = table.set_column(table.schema.get_field_index("full_logit"),
                                       "full_logit", table["mean_gate_logit"])
    original_comparison = summarize(comparison_table, users, repetitions=1000,
                                    seed=17, releases=releases)
    for value, reference in ((dynamic, original), (mean, original_mean)):
        assert value["users"] == 8
        assert value["idle_uids"] == 1
        assert value["overall"]["equal_release_mean_auc"] == reference["overall"]["equal_release_mean_auc"]
        assert value["overall"]["paired"] == reference["overall"]["paired"]
        assert value["overall"]["score_deviation"] == reference["overall"]["score_deviation"]
        assert value["by_release"] == reference["by_release"]
    assert difference["overall"] == original_comparison["overall"]["paired"]["partial_minus_full_auc"]
    for version in releases:
        assert difference["by_release"][version] == original_comparison["by_release"][version]["paired"]["partial_minus_full_auc"]


def test_vectorized_round_metrics_match_retained_ranking():
    table, _, _ = artificial_matched_panel()
    reports = within_round(table, candidates=4)["diagnostics"]
    for column, actual in reports.items():
        reference = evaluate_groups(table, column)
        assert actual["rounds"] == reference["rounds"]
        for version, values in actual["by_release"].items():
            for metric, value in values.items():
                np.testing.assert_allclose(value, reference["by_release"][version][metric], atol=1e-15)
        for metric, value in actual["equal_release_mean"].items():
            np.testing.assert_allclose(value, reference["equal_release_mean"][metric], atol=1e-15)


def test_merge_requires_complete_disjoint_users_and_same_weights(tmp_path):
    table, users, releases = artificial_matched_panel()
    panel = tmp_path / "panel"
    panel.mkdir()
    roles = panel / "users.json"
    roles.write_text(json.dumps(dict(evaluation=users, fit=[9], validation=[10])))
    binding = panel / "binding.json"
    binding.write_text(json.dumps(dict(users_file=dict(path=str(roles),
        sha256=hashlib.sha256(roles.read_bytes()).hexdigest()))))
    panel_sha = hashlib.sha256(binding.read_bytes()).hexdigest()
    weights = {version: dict(sha256="weights-" + version) for version in releases}
    published = {version: dict(kind="query_relay_full_catalogue_release_v1",
        artifact_sha256="publication-" + version,
        bindings=dict(calibration_sha256=weights[version]["sha256"])) for version in releases}
    folders = []
    for i, selected in enumerate((users[:4], users[4:])):
        directory = tmp_path / f"shard{i}"
        directory.mkdir()
        subset = table.filter(pc.is_in(table["uid"], value_set=pa.array(selected)))
        configuration = dict(selected_uids=selected, panel_sha256=panel_sha, versions=list(releases),
            code_preparation="compiled", native_writes="corrected", storage_floats_per_layer_row=48,
            read_shared=None, read_private=None, mean_gate_control=True, role="development_exploration",
            no_online_labels_or_teacher=True, workload=dict(kind="behavior", stride=16, candidates=4),
            source_sha256={"adapter.py": "fixed-adapter", "session.py": "fixed-session", "evaluate.py": "fixed-observer"},
            calibration_root="toy-calibration", published_prefixes="toy-publication")
        units = [dict(uid=uid, by_release={version: dict(requests=12 if uid != 8 else 0)
                 for version in releases}) for uid in selected]
        summary = dict(status="complete", selected_uids=selected, units=units, weights=weights,
            compiled_subset=published, shared_feature_banks={}, gate_statistics=dict(rows=[]),
            checks=dict(frozen_user_roles=True, matched_request_identities=True,
                        corrected_writes=True, new_current_rows_zero_code=True),
            actual_service_counters=dict(ranking_queries=subset.num_rows),
            native_ledger=[dict(action="append", version=3, before=4, width=2, count=len(selected))],
            code_read_ledger=[dict(kind="ranking", version=3, old_length=4, queries=4,
                                  coded_rows=4, native_rows=4, count=len(selected))])
        (directory / "configuration.json").write_text(json.dumps(configuration))
        (directory / "summary.json").write_text(json.dumps(summary))
        pq.write_table(subset, directory / "scores.parquet")
        folders.append(directory)
    merged_table, settings, result = aggregate(folders, panel=panel, repetitions=9)
    assert merged_table.num_rows == table.num_rows
    assert settings["selected_uids"] == users
    assert settings["anchor_inverse"] == "transpose"
    assert result["actual_service_counters"]["ranking_queries"] == table.num_rows
    assert result["native_ledger"][0]["count"] == 8
    assert result["code_read_ledger"][0]["count"] == 8
    assert len(result["units"]) == 8
    direct = joint_quality(table, users, releases=releases, repetitions=9)[0]
    assert result["quality"]["overall"] == direct["overall"]
    with pytest.raises(ValueError, match="overlapping"):
        aggregate([folders[0], folders[0]], panel=panel, repetitions=1)
    with pytest.raises(ValueError, match="coverage incomplete"):
        aggregate(folders[:1], panel=panel, repetitions=1)
    different = json.loads((folders[1] / "configuration.json").read_text())
    different["anchor_inverse"] = "actual"
    (folders[1] / "configuration.json").write_text(json.dumps(different))
    with pytest.raises(ValueError, match="replay protocol"):
        aggregate(folders, panel=panel, repetitions=1)
    different.pop("anchor_inverse")
    (folders[1] / "configuration.json").write_text(json.dumps(different))
    wrong = json.loads((folders[1] / "summary.json").read_text())
    wrong["weights"]["v3"]["sha256"] = "different-weights"
    (folders[1] / "summary.json").write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="calibration or published"):
        aggregate(folders, panel=panel, repetitions=1)
