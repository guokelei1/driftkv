"""Tiny saved-score evidence: cross-rank joins, cost denominators and signs."""

import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from selective_recompute_2026_09.adjudicate import adjudicate, collect_summary
from selective_recompute_2026_09.common import METHODS, budgets, sha256, signature, write_json
from selective_recompute_2026_09.cost import CostModel


def saved_fixture(tmp_path, *, single_class=False):
    output, panels = tmp_path / "run", tmp_path / "panels"
    panel_dir = panels / "medium/v0_to_v1"
    panel_dir.mkdir(parents=True)
    requests = [
        {"request_id": f"r{i}", "uid": uid, "query_timestamp": 200 + i, "label": label,
         "full_logit": full, "reuse_logit": reuse}
        for i, (uid, label, full, reuse) in enumerate([
            (11, 1, 2., 0.), (11, 0, 0., 1.), (22, 1, 3., 2.), (22, 0, 1., 3.),
            # Available canary users not scored by this probe: never mix them into endpoints.
            (33, 1, -100., 100.), (33, 0, 100., -100.),
        ])
    ]
    if single_class:
        for request in requests:
            request["label"] = 1
    request_path = panel_dir / "canary_requests.parquet"
    pq.write_table(pa.Table.from_pylist(requests), request_path)
    panel_path = panel_dir / "binding.json"
    write_json(panel_path, {"scale": "medium", "edge": "v0_to_v1", "files": {
        "canary_requests": {"path": str(request_path), "sha256": sha256(request_path)},
    }})
    sources = {"fixture_worker.py": "fixed-source-version"}
    cal_path = output / "layer/medium/v0_to_v1/calibration.json"
    write_json(cal_path, {"status": "complete", "scale": "medium", "edge": "v0_to_v1",
                         "probe_only": True, "panel_binding_sha256": sha256(panel_path),
                         "execution_source_hashes": sources, "cost": {"calibration_flops": 12345}})
    for rank, uid in enumerate((11, 22)):
        selected = [row for row in requests if row["uid"] == uid]
        inputs = {"execution_sources": sources, "panel_binding": sha256(panel_path),
                  "requests": sha256(request_path), "calibration": sha256(cal_path),
                  "uids": [uid], "partition": "canary", "scale": "medium", "edge": "v0_to_v1"}
        bound = signature(inputs)
        outputs = {}
        for method in METHODS:
            rows = []
            for budget in budgets(6)[method]:
                for request in selected:
                    selection = 7 if method in ("deviation", "query") else 0
                    # Tail is worse than Reuse; other methods are equal to saved Full.
                    score = (1 - 2 * request["label"]) * 4. if method == "tail" else request["full_logit"]
                    rows.append({**{key: request[key] for key in ("request_id", "uid", "query_timestamp")},
                                 "budget": budget["name"], "hstu_logit": score,
                                 "recompute_flops": 11, "selection_flops": selection,
                                 "total_flops": 11 + selection,
                                 "selection_sort_comparisons_estimate": 2.5 if selection else 0.})
            path = output / method / "medium/v0_to_v1" / f"rank{rank}/shard_00000.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows), path)
            outputs[method] = {"path": str(path), "sha256": sha256(path), "rows": len(rows)}
        shared = output / "runtime/medium/v0_to_v1" / f"rank{rank}"
        write_json(shared / "unit_00000.json", {
            "source_signature": bound, "uids": [uid], "requests": 2, "outputs": outputs,
            "stats": {"full_history_hist": {10: 2}, "append_prefix_hist": {9: 3}, "initial_history_hist": {10: 1}},
        })
        write_json(shared / "complete.json", {
            "status": "complete", "rank": rank, "scale": "medium", "edge": "v0_to_v1",
            "partition": "canary", "inputs": inputs, "source_signature": bound,
            "users": 1, "requests": 2, "units": 1,
        })
    return output, panels


def test_joined_probe_uses_actual_users_and_preserves_negative_recovery(tmp_path):
    output, panels = saved_fixture(tmp_path)
    result = adjudicate("medium", 1, output, panels, world_size=2)
    assert result["requests"] == 4
    assert result["users"] == 2 and result["available_partition_users"] == 3
    assert result["current_full"]["ROC_AUC"] == 1
    assert result["current_reuse"]["ROC_AUC"] == .25
    cost = CostModel.for_scale("medium")
    denominator = 4 * cost.full_cache(10) - 6 * cost.append(9, new_kv_only=False)
    assert result["cost"]["full_minus_reuse_flops"] == denominator
    for point in result["points"]:
        if point["kind"] == "reference":
            expected = 100 if point["budget"] == "full" else 0
            assert point["relative_flops_percent"] == expected
            assert point["recovery_percent"] == expected
        elif point["baseline"] == "tail":
            assert point["recovery_percent"] == pytest.approx(-100 / 3)
        elif point["baseline"] == "layer":
            assert point["recovery_percent"] == 100
            assert point["calibration_flops"] == 12345
            assert point["extra_flops"] == 4 * 11 + 12345
        else:
            assert point["extra_flops"] == 4 * (11 + 7)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["completed_curves"] == 4 and summary["formal_evaluation"] is False
    with pytest.raises(RuntimeError, match="all 15 edges"):
        collect_summary(output)


def test_duplicate_request_cannot_enter_auc_even_with_updated_shard_seal(tmp_path):
    output, panels = saved_fixture(tmp_path)
    unit_path = output / "runtime/medium/v0_to_v1/rank0/unit_00000.json"
    unit = json.loads(unit_path.read_text())
    path = Path(unit["outputs"]["tail"]["path"])
    rows = pq.read_table(path).to_pylist()
    rows[1]["request_id"] = rows[0]["request_id"]
    pq.write_table(pa.Table.from_pylist(rows), path)
    unit["outputs"]["tail"]["sha256"] = sha256(path)
    write_json(unit_path, unit)
    with pytest.raises(RuntimeError, match="duplicate requests"):
        adjudicate("medium", 1, output, panels, world_size=2)


def test_short_history_torch_costs_replace_only_their_workload_subset(tmp_path):
    output, panels = saved_fixture(tmp_path)
    unit_path = output / "runtime/medium/v0_to_v1/rank0/unit_00000.json"
    unit = json.loads(unit_path.read_text())
    unit["stats"].update(torch_full_history_hist={10: 2}, torch_append_prefix_hist={9: 3})
    write_json(unit_path, unit)
    result = adjudicate("medium", 1, output, panels, world_size=2)
    native, eager = CostModel.for_scale("medium"), CostModel.for_scale("medium", "torch")
    assert result["cost"]["full_minus_reuse_flops"] == (
        2 * native.full_cache(10) + 2 * eager.full_cache(10)
        - 3 * native.append(9, new_kv_only=False) - 3 * eager.append(9, new_kv_only=False)
    )


def test_single_label_canary_keeps_cost_and_reports_auc_as_undefined(tmp_path):
    output, panels = saved_fixture(tmp_path, single_class=True)
    result = adjudicate("medium", 1, output, panels, world_size=2)
    assert result["status"] == "complete" and result["auc_defined"] is False
    assert result["requests"] == 4 and result["users"] == 2
    for point in result["points"]:
        assert point["full_auc"] is None and point["reuse_auc"] is None
        assert point["baseline_auc"] is None and point["recovery_percent"] is None
        assert point["relative_flops_percent"] is not None
    assert "no user reselection" in result["auc_note"]


def test_band_costs_added_once_and_torch_subset_uses_eager_execution(tmp_path):
    output, panels = saved_fixture(tmp_path)
    unit_path = output / "runtime/medium/v0_to_v1/rank0/unit_00000.json"
    unit = json.loads(unit_path.read_text())
    unit["stats"].update(band_append_hist={"9:3": 2}, torch_band_append_hist={"9:3": 1})
    write_json(unit_path, unit)
    result = adjudicate("medium", 1, output, panels, world_size=2)
    native, eager = CostModel.for_scale("medium"), CostModel.for_scale("medium", "torch")
    assert result["cost"]["full_minus_reuse_flops"] == (
        4 * native.full_cache(10) - 6 * native.append(9, new_kv_only=False)
        - native.band_append(9, 3) - eager.band_append(9, 3)
    )
