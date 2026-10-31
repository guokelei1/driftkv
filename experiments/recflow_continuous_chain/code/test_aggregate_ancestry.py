"""Synthetic CPU evidence for direct-parent and multi-update publications."""
import copy
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.recflow_continuous_chain.code import aggregate_full_population as agg
from experiments.recflow_release_design.code.interval_protocol import prepare_selection


def save(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def case(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(agg, "DATA", data)
    train = np.arange(30000, dtype=np.int64)
    np.savez(data / "population.npz", train=train)
    permutation = np.random.default_rng(agg.SEED).permutation(train)
    lo, _ = agg.day_bounds_ms(36)
    dtype = [("uid", "i8"), ("rid", "i8"), ("day", "i2"), ("ts", "i8"),
             ("start", "i8"), ("stop", "i8"), ("positive", "i4"),
             ("observed", "i4"), ("history_length", "i4")]
    records = np.zeros(13, dtype=dtype)
    # Both UID halves and both time halves occur within each fixed cohort.
    for index, position in enumerate((0, 1, 4096, 4097, 12288, 12289)):
        records[2*index:2*index+2]["uid"] = permutation[position]
        records[2*index]["ts"] = lo + 1000
        records[2*index+1]["ts"] = lo + 13*3600*1000
    records[-1]["uid"], records[-1]["ts"] = permutation[0], lo + 2000
    records["day"], records["rid"] = 36, np.arange(13) + 100
    records["start"], records["stop"] = np.arange(13)*40, (np.arange(13)+1)*40
    records["positive"], records["observed"], records["history_length"] = 1, 2, 1500
    records[-1]["positive"] = 0
    np.save(data / "full.npy", records)
    save(data / "selection_manifest.json", {
        "datasets": {"2:36": {"file": "full.npy", "requests": len(records)}}})
    proposal = {"architecture": {"max": {"layers": 16, "hidden_size": 320,
                                        "heads": 10, "head_dim": 32},
                                 "max_seq_len_including_query": 1025},
                "history": {"max_events": 1024}}

    def checkpoint(name, day, parent=None):
        path = tmp_path / f"{name}.pt"
        path.touch()
        save(path.with_suffix(".json"), {
            "status": "committed", "metadata": {
                "days": [day], "scale": "max", "proposal": proposal,
                "quality_use_permitted": True,
                "arguments": {"data": str(data), "parent": str(parent) if parent else None}}})
        return path

    source = checkpoint("published", 33)
    intermediate = checkpoint("unpublished", 34, source)
    target = checkpoint("target", 35, intermediate)
    sibling = checkpoint("sibling", 34, source)
    unrelated = checkpoint("unrelated", 33)
    plan = {"checkpoint": str(target), "cache_source": str(source), "data": str(data),
            "eval_day": 36, "history_limit": 1024, "batch_size": 16,
            "protocol": "matched-prefix complete-candidate snapshot",
            "primary_metric": "NDCG@10", "secondary_metric": "Recall@10",
            "evaluation_interval_ms_half_open": list(agg.day_bounds_ms(36))}
    return SimpleNamespace(root=tmp_path, data=data, records=records, source=source,
                           intermediate=intermediate, target=target, sibling=sibling,
                           unrelated=unrelated, plan=plan)


def revise_metadata(checkpoint, change):
    path = checkpoint.with_suffix(".json")
    document = json.loads(path.read_text())
    change(document)
    save(path, document)


def completed_sources(case, cache_source=None):
    """Small synthetic audited CSV fixtures; these are not experiment results."""
    base_plan = dict(case.plan, cache_source=str(cache_source or case.source))
    context = agg.load_context(base_plan)
    runs = []
    for number, (offset, count) in enumerate(((0, 4096), (4096, 8192), (12288, 17712))):
        run = case.root / f"sample_{number}"
        selection = prepare_selection(case.data, 36, run, count, agg.SEED, offset)
        records = np.load(run / "selection.npy")
        chosen = np.load(run / "chosen_uids.npy")
        rows = {}
        for record in records:
            row = {name: int(record[name]) for name in ("uid", "rid", "day")}
            row.update(candidates=40, exposed=int(record["observed"]),
                       positives=int(record["positive"]), history_length=1024)
            for method in ("current", "reuse"):
                for metric in agg.METRICS:
                    # Positive at rank 1 under current and rank 40 under reuse.
                    row[f"{method}_{metric}"] = (
                        float(method == "current") if row["positives"] else None)
            rows[agg.key(record)] = row
        overall, slices = agg.slices_for(records, rows, chosen, context["bounds"])
        plan = dict(base_plan, selection=selection, actual_training_path=context["lineage"])
        summary = {"status": "completed", "plan": copy.deepcopy(plan),
                   "actual_training_path": context["lineage"], "history_limit": 1024,
                   "precision": "float32_tf32_disabled", "world_size": 4,
                   "protocol": plan["protocol"], "users": overall["active_users"],
                   **{k: overall[k] for k in ("requests", "positive_requests",
                                              "zero_positive_requests", "positive_users")},
                   "metrics": {method: {m: overall["metrics"][m][label] for m in agg.METRICS}
                               for method, label in (("current", "rebuild"),
                                                     ("reuse", "ancestor_reuse"))}}
        analysis = {"status": "completed", "checkpoint": plan["checkpoint"],
                    "cache_source": plan["cache_source"], "actual_training_path": context["lineage"],
                    "evaluation_day": 36, "evaluation_interval_ms_half_open": list(context["bounds"]),
                    "overall": overall, "predetermined_slices": slices,
                    "uncertainty": "synthetic UID bootstrap fixture", "elapsed": {
                        "gpu_evaluator_wall_seconds": 0}}
        audit = {"status": "passed", "actual_training_path": context["lineage"],
                 "all_chosen_users_complete_day_requests": True,
                 **{k: overall[k] for k in agg.COUNT_FIELDS},
                 "metric_files": 4, "raw_score_files": 4,
                 "candidate_instances": int((records["stop"]-records["start"]).sum()),
                 "observed_positive_instances": int(records["positive"].sum())}
        completion = {"status": "completed", "completed_at": "synthetic fixture",
                      "audit": str(run / "audit.json"), "analysis": str(run / "analysis.json"),
                      **{k: overall[k] for k in agg.COUNT_FIELDS},
                      "primary_result": overall["comparisons"]["ndcg@10"]}
        for name, document in (("plan", plan), ("summary", summary), ("analysis", analysis),
                               ("audit", audit), ("completion", completion)):
            save(run / f"{name}.json", document)
        values = list(rows.values())
        for rank in range(4):
            with (run / f"rank_{rank}.csv").open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values[rank::4])
            score_dir = run / f"rank_{rank}_scores"
            score_dir.mkdir()
            # Aggregation consumes prior audits, not these raw score contents.
            np.savez(score_dir / "batch_000000.npz", synthetic_fixture=True)
        runs.append(run)
    return runs


@pytest.mark.parametrize("direct", [True, False])
def test_aggregate_keeps_real_parent_and_published_source_distinct(case, direct):
    source = case.intermediate if direct else case.source
    analysis, audit = agg.aggregate(completed_sources(case, source))
    assert analysis["cache_source"] == str(source)
    assert analysis["immediate_optimizer_parent"] == str(case.intermediate)
    assert [r["checkpoint"] for r in analysis["actual_training_path"]] == list(map(str, (
        [case.intermediate, case.target] if direct
        else [case.source, case.intermediate, case.target])))
    assert audit["chosen_users"] == 30000
    assert audit["requests"] == 13 and audit["positive_requests"] == 12
    assert audit["candidate_instances"] == 520
    assert audit["source_uids_pairwise_disjoint"]
    assert audit["union_matches_complete_day_selection_ledger"]
    assert analysis["overall"]["metrics"]["ndcg@10"]["rebuild"] == 1.0
    assert analysis["overall"]["metrics"]["ndcg@10"]["ancestor_reuse"] == 0.0
    assert analysis["independent_confirmation"] is False
    assert all(s["chosen_users"] == 15000 for n, s in analysis["predetermined_slices"].items()
               if n.startswith("chosen_uid"))


@pytest.mark.parametrize("name", ["sibling", "unrelated", "target"])
def test_rejects_nonancestor_or_same_checkpoint_source(case, name):
    with pytest.raises(ValueError, match="not an actual training ancestor|earlier ancestor"):
        agg.load_context(dict(case.plan, cache_source=str(getattr(case, name))))


def test_rejects_future_source(case):
    revise_metadata(case.source, lambda d: d["metadata"].update(days=[36]))
    with pytest.raises(ValueError, match="cutoff is later"):
        agg.load_context(case.plan)


@pytest.mark.parametrize("day", [27, 35, 38])
def test_requires_available_strictly_future_evaluation_day(case, day):
    with pytest.raises(ValueError, match="future available full day"):
        agg.load_context(dict(case.plan, eval_day=day))


@pytest.mark.parametrize("missing", ["checkpoint", "sidecar"])
def test_requires_intermediate_checkpoint_and_metadata(case, missing):
    (case.intermediate if missing == "checkpoint" else case.intermediate.with_suffix(".json")).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        agg.load_context(case.plan)


def test_rejects_different_data_at_unpublished_intermediate(case):
    revise_metadata(case.intermediate, lambda d: d["metadata"]["arguments"].update(
        data=str(case.root / "other_data")))
    with pytest.raises(ValueError, match="full-population Max/1024"):
        agg.load_context(case.plan)


def test_rejects_noncommitted_intermediate(case):
    revise_metadata(case.intermediate, lambda d: d.update(status="writing"))
    with pytest.raises(ValueError, match="not committed"):
        agg.load_context(case.plan)


def test_rejects_missing_source_completion_before_reading_scores(case):
    runs = [case.root / str(i) for i in range(3)]
    for run in runs:
        run.mkdir()
    with pytest.raises(ValueError, match="completion.json does not exist"):
        agg.aggregate(runs)


@pytest.mark.parametrize("mutation", ["uid", "candidate", "lineage", "audit", "pair"])
def test_ancestor_support_preserves_source_evidence_checks(case, mutation):
    runs = completed_sources(case)
    run = runs[1]
    if mutation == "uid":
        chosen = np.load(run / "chosen_uids.npy")
        chosen[0] = np.load(runs[0] / "chosen_uids.npy")[0]
        np.save(run / "chosen_uids.npy", chosen)
    elif mutation == "candidate":
        path = run / "rank_0.csv"
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["candidates"] = "41"
        with path.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    elif mutation == "lineage":
        path = run / "analysis.json"
        document = json.loads(path.read_text())
        document["actual_training_path"].pop(1)
        save(path, document)
    elif mutation == "audit":
        (run / "audit.json").unlink()
    else:
        path = run / "plan.json"
        document = json.loads(path.read_text())
        document["cache_source"] = str(case.intermediate)
        save(path, document)
    with pytest.raises(ValueError):
        agg.aggregate(runs)
