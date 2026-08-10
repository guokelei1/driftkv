import torch

from hstu_kvcache.streaming.kuairand_engagement import interleave_batch


def test_candidate_query_precedes_observed_behavior_without_label_leakage() -> None:
    batch = {
        "item_ids": torch.tensor([[4, 5, 0]]),
        "behaviors": torch.tensor([[2, 7, 0]]),
        "time_deltas": torch.tensor([[1.0, 2.0, 0.0]]),
        "lengths": torch.tensor([2]),
        "labels": torch.tensor([[1, 0, 0]]),
        "train_mask": torch.tensor([[True, True, False]]),
    }
    value = interleave_batch(batch, 9, torch.device("cpu"))
    assert value["item_ids"].tolist() == [[4, 4, 5, 5, 0, 0]]
    assert value["behaviors"].tolist() == [[9, 2, 9, 7, 0, 0]]
    assert value["lengths"].tolist() == [4]
    assert value["target_mask"].tolist() == [[True, True, False]]
