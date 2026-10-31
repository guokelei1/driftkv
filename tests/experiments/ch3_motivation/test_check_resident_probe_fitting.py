"""CPU entry/oracle checks; these are not GPU residency/speed evidence."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code import check_resident_probe_fitting as check
from experiments.ch3_motivation.code.probe_variants import build_variant
from hstu_kvcache.models import HSTUConfig, HSTUKVCache
from tests.experiments.ch3_motivation.test_resident_probe_fitting import world


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND", "torch")
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("width", [8, 12])
def test_actual_nested_fitter_gradient_state_rng_and_cost_oracles(width):
    model, train, validation = world()
    torch.manual_seed(17)
    initial = build_variant("full_item", model, stage="f", hidden_width=width, depth=1,
                            row_chunk=8, query_chunk=3)
    for stage in check.STAGES:
        assert initial.stage == stage
        before = {name: value.clone() for name, value in initial.state_dict().items()}
        best, result = check.numeric_pair(model, initial, train, validation, steps=4)
        assert result["passed"] and result["rng_unchanged"]
        assert result["original_report"]["counters_arithmetic_and_best_step_exact"]
        assert result["reference_fit"]["completed_steps"] == 4
        assert result["reference_fit"]["operation_counts"]["training_backward_calls"] == 4
        for gradients in result["gradients"]["parameters"].values():
            assert gradients["reference_steps"] == gradients["resident_steps"] == 4
            assert all(row["max_abs"] == 0 for row in gradients["steps"])
        assert all(torch.equal(value, initial.state_dict()[name]) for name, value in before.items())
        assert all(not parameter._backward_hooks for parameter in best.parameters())
        assert result["resident_execution_adapter"]["metrics"]["cuda_memory"] is None
        assert bool(best.statistics_initialized)
        if stage != "fgh":
            initial = best.expanded("fg" if stage == "f" else "fgh")


def test_comparisons_keep_missing_gradient_step_and_counter_failures():
    values = {"h.weight": [torch.zeros(2), torch.tensor([1., 2.])]}
    missing = {"h.weight": [torch.zeros(2)]}
    assert not check.compare_gradients(values, missing, 2, rtol=1e-5, atol=1e-7)["passed"]
    changed = copy.deepcopy(values)
    changed["h.weight"][1][0] += .01
    assert not check.compare_gradients(values, changed, 2, rtol=1e-5, atol=1e-7)["passed"]
    original = dict(best_step=0, completed_steps=4, operation_counts={"training_probe_rows": 32},
                    arithmetic={"training_probe_forward_flops": 128}, total_calibration_flops_estimate=128)
    altered = copy.deepcopy(original)
    altered["operation_counts"]["training_probe_rows"] += 1
    assert not check.compare_reports(original, altered, rtol=1e-5, atol=1e-7)["passed"]


def test_two_actual_32_step_timing_pairs_have_no_hooks_and_include_placement(monkeypatch):
    model, train, validation = world()
    initial = build_variant("full_item", model, stage="fgh", hidden_width=8, depth=1)

    def forbidden_hook(*args, **kwargs):
        raise AssertionError("a timing path installed a gradient hook")

    monkeypatch.setattr(torch.Tensor, "register_hook", forbidden_hook)
    result = check.timing_pairs(model, initial, train, validation, steps=32, repeats=2)
    assert result["passed"]
    assert [row["order"] for row in result["matched_replays"]] == [
        ["reference", "resident"], ["resident", "reference"]]
    for row in result["matched_replays"]:
        assert row["reference_over_resident_wall"] > 0  # CPU value never reported as GPU speed.
        for mode, run in row["runs"].items():
            assert run["fit"]["completed_steps"] == 32
            assert run["fit"]["operation_counts"]["training_backward_calls"] == 32
            assert len(run["fit"]["history"]) == 5  # validation at0,8,16,24,32.
            assert run["peak_memory"] is None
            if mode == "resident":
                metrics = run["fit"]["execution_adapter"]["metrics"]
                assert run["wall_seconds"] >= metrics["total_seconds"]
                assert metrics["preplacement_seconds"] >= 0
                assert metrics["storage"]["cross_device_logical_bytes"] == 0


def preparation_world(tmp_path, monkeypatch):
    directory = tmp_path / "edge_2"
    directory.mkdir()
    fit, val = list(range(256)), list(range(256, 272))
    population = dict(fit=fit, validation=val)
    (tmp_path / "population.json").write_text(json.dumps(population))
    for name in ("histories.npz", "requests.parquet", "calibration_feedback_candidates.json"):
        (directory / name).write_text("CPU fixture only; not real input")
    manifest = tmp_path / "sampling_manifest.json"
    manifest.write_text("CPU fixture manifest")
    binding = dict(population_sha256=check.sha256(tmp_path / "population.json"),
        history=dict(path="histories.npz", sha256=check.sha256(directory / "histories.npz")),
        requests=dict(path="requests.parquet", sha256=check.sha256(directory / "requests.parquet")),
        settings=dict(edge=2, history_length=1024, release_day=245))
    (directory / "inputs.json").write_text(json.dumps(binding))
    provenance = dict(fit_uids=fit, validation_uids=val, catalog_source_users=256, time_law_source_users=256,
        source_cutoff=245*86400, gap_rule="gap_empirical100", manifest_sha=check.sha256(manifest),
        view_key="feedback50_gap_empirical100")
    views = {uid: dict(candidates=torch.arange(1, 17)[None], query_delta=torch.arange(16)[None].float()+uid)
             for uid in fit+val}
    calls = []

    def sampling(path, root, edge, fraction, rule, *, fit_users):
        assert (edge, fraction, rule, fit_users) == (2, .5, "gap_empirical100", 256)
        calls.append("sampling")
        return views, provenance

    model = SimpleNamespace(cfg=HSTUConfig(num_items=20, num_behaviors=2, hidden_size=320,
        num_heads=10, num_layers=10), blocks=[SimpleNamespace(attn=SimpleNamespace(head_dim=32))]*10)

    def models(edge, device, backend):
        calls.append("models")
        assert edge == 2 and str(device) == "cpu" and backend == "torch"
        return object(), model, {"fixture_model": "fixture_sha"}

    def capture(parent, current, histories, uids, cfg, device):
        assert uids == fit+val
        calls.append("capture272")
        cache = HSTUKVCache(torch.zeros(1, 1, 1, 8), torch.zeros(1, 1, 1, 8), 1)
        # Actual view helper must remove these stale targets before preparation.
        return {uid: dict(parent_cache=cache, teacher_cache=cache, item_ids=torch.ones(1, 1).long(),
            candidates=torch.zeros(1, 16).long(), query_delta=torch.ones(1), teacher_scores=torch.ones(1, 16),
            read_scales=torch.ones(1), score_scale=torch.ones(()), target_moment_version="stale") for uid in uids}

    def targets(current, samples):
        calls.append(f"targets{len(samples)}")
        for sample in samples:
            assert "teacher_scores" not in sample and "target_moment_version" not in sample
            assert "read_scales" not in sample and "score_scale" not in sample
            assert torch.equal(sample["candidates"], views[sample["uid"]]["candidates"])
            assert torch.equal(sample["query_delta"], views[sample["uid"]]["query_delta"])
        return samples

    monkeypatch.setattr(check.calibration_sampling, "load_sampling_manifest", sampling)
    monkeypatch.setattr(check, "load_models", models)
    monkeypatch.setattr(check, "restore_pack", lambda path: object())
    monkeypatch.setattr(check, "capture_samples", capture)
    monkeypatch.setattr(check.probes, "prepare_training_samples", targets)
    args = SimpleNamespace(input_root=tmp_path, sampling_manifest=manifest, backend="torch")
    return args, calls, provenance


def test_entry_capture_once_actual_roles_view_before_targets_and_stale_purge(tmp_path, monkeypatch):
    args, calls, provenance = preparation_world(tmp_path, monkeypatch)
    _, train, val, record = check.prepare_inputs(args, torch.device("cpu"))
    assert calls == ["sampling", "models", "capture272", "targets256", "targets16"]
    assert len(train) == 256 and len(val) == 16
    assert record["fit_uids"] == provenance["fit_uids"]
    assert record["validation_uids"] == provenance["validation_uids"]
    assert record["sampling_provenance"]["source_cutoff"] == 245*86400
    assert len(record["input_sha256"]) == 6
    assert record["shared_preparation_seconds"]["total"] >= 0


def test_wrong_law_source_rejected_before_model_capture(tmp_path, monkeypatch):
    args, calls, provenance = preparation_world(tmp_path, monkeypatch)
    provenance["time_law_source_users"] = 64
    with pytest.raises(ValueError, match="sampling role/source/cutoff"):
        check.prepare_inputs(args, torch.device("cpu"))
    assert calls == ["sampling"]


def test_cli_is_explicit_and_diagnostic_without_cuda_initialization(tmp_path, monkeypatch):
    def forbidden_cuda(*args, **kwargs):
        raise AssertionError("CPU entry test attempted CUDA initialization")

    monkeypatch.setattr(torch.cuda, "set_device", forbidden_cuda)
    args = check.parser().parse_args(["--gpu", "0", "--output", str(tmp_path)])
    assert args.numeric_steps == 4 and args.timing_steps == 32 and args.repeats == 2
    assert check.WIDTHS == (128, 256) and check.STAGES == ("f", "fg", "fgh")
    check.check_frozen_fitter()
    args.timing_steps = 31
    with pytest.raises(ValueError, match="matched 32-step"):
        check.run(args)
    assert not list(tmp_path.iterdir())
    assert check.DIAGNOSTIC["diagnostic_only"]
    assert not check.DIAGNOSTIC["quality_use_permitted"]
    assert not check.DIAGNOSTIC["formal_weights_written"]
