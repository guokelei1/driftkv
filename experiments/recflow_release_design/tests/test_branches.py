import copy

import numpy as np
import pytest
import torch

from experiments.recflow_full_population.code.batches import GlobalBatchSampler
from experiments.recflow_release_design.code.common import validate_metadata, verify_saved_state


def metadata():
    shared = dict(scale="max", proposal={"history": 1024}, vocabulary_sha256="v", source_binding={"core": "f"}, quality_use_permitted=True)
    previous = dict(status="committed", checkpoint_sha256="p", metadata=dict(shared, days=[31]))
    first = dict(status="committed", checkpoint_sha256="f", metadata=dict(shared, phase="update", days=[32],
                 parent_checkpoint_sha256="p", next_epoch=1, next_step=0,
                 arguments={"epochs": 1, "parent": "/models/m4.pt"}))
    spec = dict(first_pass_checkpoint_sha256="f", previous_published_checkpoint_sha256="p",
                previous_published_checkpoint="/models/m4.pt", train_day=32, eval_day=33)
    return spec, first, previous


def test_valid_branch_and_wrong_published_parent_rejected():
    spec, first, previous = metadata()
    validate_metadata(spec, first, previous)
    first["metadata"]["arguments"]["parent"] = "/models/another_parent.pt"
    with pytest.raises(ValueError, match="descend"):
        validate_metadata(spec, first, previous)


def test_historical_hash_fields_do_not_block_compatible_branch():
    spec, first, previous = metadata()
    first["checkpoint_sha256"] = "changed-checkpoint-record"
    first["metadata"]["parent_checkpoint_sha256"] = "changed-parent-record"
    first["metadata"]["source_binding"] = {"core": "edited-source"}
    first["metadata"]["vocabulary_sha256"] = "edited-manifest"
    validate_metadata(spec, first, previous)


@pytest.mark.parametrize("field,value", [("next_epoch", 0), ("next_step", 12), ("quality_use_permitted", False), ("days", [33])])
def test_incomplete_or_wrong_window_rejected(field, value):
    spec, first, previous = metadata()
    first["metadata"][field] = value
    with pytest.raises(ValueError):
        validate_metadata(spec, first, previous)


def test_second_epoch_order_covers_every_request_once_on_four_ranks():
    count = 159607
    parts = []
    for rank in range(4):
        sampler = GlobalBatchSampler(count, global_batch=64, micro_batch=16, world_size=4, rank=rank, seed=18)
        parts.extend(x for batch in sampler for x in batch if x >= 0)
    assert np.array_equal(np.sort(parts), np.arange(count))
    original_second = GlobalBatchSampler(count, global_batch=64, micro_batch=16, world_size=4, seed=17, epoch=1)
    extra = GlobalBatchSampler(count, global_batch=64, micro_batch=16, world_size=4, seed=18, epoch=0)
    assert np.array_equal(original_second.order, extra.order)
    first = GlobalBatchSampler(count, global_batch=64, micro_batch=16, world_size=4, seed=17)
    assert not np.array_equal(extra.order, first.order)


def test_restoration_checks_every_model_and_optimizer_value():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    payload = copy.deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict()))
    assert verify_saved_state(model, optimizer, payload)["optimizer_elements"] > 0
    payload["optimizer"]["state"][0]["exp_avg"].view(-1)[-1] += 1
    with pytest.raises(ValueError, match="differs"):
        verify_saved_state(model, optimizer, payload)
