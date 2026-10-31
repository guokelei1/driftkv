"""Explicit optimizer ancestry and published-parent identity for epoch branches."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.recflow_full_population.code.data_common import sha256

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
DATA = REPO / "experiments/recflow_full_population/data"
CONFIG = REPO / "experiments/recflow_full_population/configs/full.json"
ORIGINAL = REPO / "experiments/recflow_full_population/runs/attempt_1"


def read(path):
    return json.loads(Path(path).read_text())


def branch_binding():
    return {str(p): sha256(p) for p in sorted((BASE / "code").glob("*.py"))}


def validate_metadata(spec, first, published):
    """No weights are relabeled: the first pass really descends from the release."""
    f, p = first["metadata"], published["metadata"]
    if first["status"] != "committed" or published["status"] != "committed":
        raise ValueError("Expected completed checkpoints")
    actual_parent = f.get("arguments", {}).get("parent")
    if not actual_parent or Path(actual_parent).resolve() != Path(spec["previous_published_checkpoint"]).resolve():
        raise ValueError("First pass does not descend from the declared published parent")
    if (f["phase"] != "update" or f["days"] != [spec["train_day"]]
            or max(p["days"]) != spec["train_day"] - 1
            or spec["eval_day"] != spec["train_day"] + 1):
        raise ValueError("Branch changes its training/evaluation window")
    if (f["next_epoch"] != 1 or f["next_step"] != 0 or f["arguments"]["epochs"] != 1
            or not f["quality_use_permitted"] or not p["quality_use_permitted"]):
        raise ValueError("Expected a complete, quality-permitted first pass")
    for key in ("scale", "proposal"):
        if f[key] != p[key]:
            raise ValueError(f"Original parent and first pass differ in {key}")


def load_spec(path, data=DATA):
    path = Path(path).resolve()
    spec = read(path)
    first = read(Path(spec["first_pass_checkpoint_path"]).with_suffix(".json"))
    published = read(Path(spec["previous_published_checkpoint"]).with_suffix(".json"))
    validate_metadata(spec, first, published)
    return spec


def release_metadata(path):
    spec = read(path)
    return {
        "branch": spec["branch"], "spec_path": str(Path(path).resolve()), "spec_sha256": sha256(path),
        "previous_published_checkpoint": spec["previous_published_checkpoint"],
        "previous_published_checkpoint_sha256": spec["previous_published_checkpoint_sha256"],
        "first_pass_checkpoint": spec["first_pass_checkpoint_path"],
        "first_pass_checkpoint_sha256": spec["first_pass_checkpoint_sha256"],
        "total_update_epochs": 2, "extra_epoch_shuffle_seed": 18,
        "learning_rate": "one extra complete pass; restart original 5% warmup/cosine 1e-4 to 5e-6",
        "branch_source_binding": branch_binding(),
    }


def validate_extension(payload, spec_path, *, baseline=False):
    spec = read(spec_path)
    expected_parent = spec["previous_published_checkpoint" if baseline else "first_pass_checkpoint_path"]
    actual_parent = payload.get("arguments", {}).get("parent")
    if not actual_parent or Path(actual_parent).resolve() != Path(expected_parent).resolve():
        raise ValueError("Wrong optimizer parent for the declared branch")
    if not baseline:
        expected = release_metadata(spec_path)
        branch = payload.get("release_branch", {})
        fields = ("branch", "previous_published_checkpoint", "first_pass_checkpoint",
                  "total_update_epochs", "extra_epoch_shuffle_seed", "learning_rate")
        if (any(branch.get(key) != expected[key] for key in fields)
                or payload["next_epoch"] != 1 or payload["next_step"] != 0
                or not payload.get("quality_use_permitted")):
            raise ValueError("Incomplete extension or different training recipe")
    if payload["days"] != [spec["train_day"]]:
        raise ValueError("Extension trained on a different day")
    return Path(spec["previous_published_checkpoint"]).resolve()


def verify_selection(dataset, spec):
    directory = Path(spec["first_pass_checkpoint_path"]).parent
    plan = read(directory / "plan.json")
    selected = directory / "request_selection.npz"
    with np.load(selected) as expected:
        if any(not np.array_equal(expected[key], dataset.records[key]) for key in expected.files):
            raise ValueError("Second pass does not cover the exact first-pass training requests")
    if len(dataset) != plan["requests"]:
        raise ValueError("Second pass changes request count")
    return {"status": "passed", "requests": len(dataset), "original_selection_sha256": sha256(selected),
            "extra_optimizer_steps": math.ceil(len(dataset) / 64), "coverage": "all original requests exactly once"}


def verify_saved_state(model, optimizer, payload):
    """Compare every restored parameter/buffer/moment element before a new update."""
    counts = {"model_elements": 0, "optimizer_elements": 0}

    def same(actual, expected, category):
        if isinstance(expected, torch.Tensor):
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise ValueError("Restored tensor shape/dtype changed")
            a, b = actual.detach().reshape(-1), expected.detach().reshape(-1)
            for start in range(0, b.numel(), 2**20):
                if not torch.equal(a[start:start+2**20].cpu(), b[start:start+2**20].cpu()):
                    raise ValueError("Restored saved state differs before the extra pass")
            counts[category] += b.numel()
        elif actual != expected:
            raise ValueError("Restored scalar optimizer state differs")

    actual_model = model.state_dict()
    if actual_model.keys() != payload["model"].keys():
        raise ValueError("Restored model keys differ")
    for key, expected in payload["model"].items():
        same(actual_model[key], expected, "model_elements")
    actual_opt = optimizer.state_dict()
    expected_opt = payload["optimizer"]
    if actual_opt["param_groups"] != expected_opt["param_groups"] or actual_opt["state"].keys() != expected_opt["state"].keys():
        raise ValueError("Optimizer groups or state IDs differ")
    for index, state in expected_opt["state"].items():
        if actual_opt["state"][index].keys() != state.keys():
            raise ValueError("Optimizer state fields differ")
        for name, expected in state.items():
            same(actual_opt["state"][index][name], expected, "optimizer_elements")
    return dict(status="passed", equality="bitwise, every saved tensor element", **counts)
