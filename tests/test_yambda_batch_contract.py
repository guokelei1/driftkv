from pathlib import Path
import sys

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from train_yambda_theta0_medium import collate_histories  # noqa: E402


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
