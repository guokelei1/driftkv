import torch

from hstu_kvcache.data import event_time_deltas


def collate_histories(histories, item_map):
    max_history = 512
    items = torch.zeros((len(histories), max_history), dtype=torch.long)
    behaviors = torch.zeros_like(items)
    deltas = torch.zeros((len(histories), max_history), dtype=torch.float32)
    lengths = torch.zeros(len(histories), dtype=torch.long)
    for row, history in enumerate(histories):
        items[row, :len(history)] = torch.tensor([item_map[item] for item, _, _ in history])
        behaviors[row, :len(history)] = torch.tensor([behavior for _, _, behavior in history])
        deltas[row, :len(history)] = torch.from_numpy(event_time_deltas(history))
        lengths[row] = len(history)
    return items, behaviors, deltas, lengths


def test_yambda_collate_uses_leading_valid_prefix() -> None:
    items, behaviors, deltas, lengths = collate_histories(
        [[(101, 10, 1), (102, 25, 2)]], {101: 1, 102: 2}
    )
    assert lengths.tolist() == [2]
    assert items[0, :2].tolist() == [1, 2]
    assert behaviors[0, :2].tolist() == [1, 2]
    assert deltas[0, :2].tolist() == [0.0, 15.0]
    assert torch.count_nonzero(items[0, 2:]) == 0
    assert torch.count_nonzero(behaviors[0, 2:]) == 0
