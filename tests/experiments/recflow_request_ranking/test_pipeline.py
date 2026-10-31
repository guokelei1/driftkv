import json

import numpy as np
import torch

from experiments.recflow_request_ranking.code.batches import GlobalBatchSampler
from experiments.recflow_request_ranking.code.data_common import RAW_DTYPE, REQUEST_DTYPE
from experiments.recflow_request_ranking.code.index_data import hash_item, history_index, time_keys
from experiments.recflow_request_ranking.code.model import RequestRanker


def test_all_world_sizes_see_same_targets_without_padding_duplicates():
    for world in [1, 2, 4]:
        seen = []
        for rank in range(world):
            sampler = GlobalBatchSampler(70, global_batch=64, micro_batch=4, world_size=world,
                                         rank=rank, shuffle=False)
            seen.extend(i for batch in sampler for i in batch if i >= 0)
        assert sorted(seen) == list(range(70))


def test_entire_same_timestamp_block_excluded_and_first_independent_of_labels():
    events = np.zeros(5, RAW_DTYPE)
    events["uid"] = [1, 1, 1, 1, 3]
    events["ts"] = [100, 200, 200, 300, 400]
    req = np.zeros(4, REQUEST_DTYPE)
    req["uid"], req["ts"], req["rid"] = [1, 1, 2, 3], [200, 300, 100, 400], [7, 8, 9, 10]
    offsets = np.r_[0, np.cumsum(np.bincount(events["uid"]))]
    result = history_index(events, req, offsets, time_keys(events["uid"], events["ts"]), 27, np.array([1, 3]))
    assert result["history_length"].tolist() == [1, 3, 0, 0]
    assert result["first_request"].tolist() == [True, False, True, True]


def test_feature_padding_oov_and_dead_last_history_attention_equivalence(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    item_map = (17 + hash_item(np.arange(100), 32)).astype(np.int32)
    item_map[:16] = np.arange(1, 17)
    np.save(tmp_path / "item_map.npy", item_map)
    (tmp_path / "manifest.json").write_text(json.dumps({"explicit_ids": 16, "hash_buckets": 32}))
    model = RequestRanker(tmp_path).eval()
    features = torch.randint(0, 30, (2, 33, 5), dtype=torch.int32)
    features[1, 17:] = -1
    candidates = torch.randint(0, 300, (2, 7, 5), dtype=torch.int32)
    batch = {"history_features": features, "behaviors": torch.ones((2, 33), dtype=torch.long),
             "deltas": torch.zeros(2, 33), "lengths": torch.tensor([33, 17]),
             "candidate_features": candidates, "query_deltas": torch.tensor([1., 2.])}
    fast, reference = model.encode_history(batch), model.encode_history(batch, reference=True)
    torch.testing.assert_close(fast.k, reference.k, rtol=0, atol=0)
    torch.testing.assert_close(fast.v, reference.v, rtol=0, atol=0)
    assert (model.features(features)[1, 17:] == 0).all()
    score = model.read(batch, fast)
    reference_score = model.read(batch, reference)
    torch.testing.assert_close(score, reference_score, rtol=0, atol=0)
    parameters = [p for p in model.parameters() if p.requires_grad]
    gradients = torch.autograd.grad(score.square().mean(), parameters)
    reference_gradients = torch.autograd.grad(reference_score.square().mean(), parameters)
    for actual, expected in zip(gradients, reference_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
