"""Fresh cost probes cannot silently alter warm or historical endpoints."""
from argparse import Namespace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from design_one.calibrate_nonlinear import training_epochs


def test_fresh_epoch_override_and_unchanged_endpoints():
    defaults = dict(method="kv_item", epochs=None, small=False, initial_calibration=None,
        rolling_scenes=False, mixed_refinement=False, producer_scale_refinement=False,
        centered_logit_refinement=False, recent_only_queries=False, refinement_epochs=None,
        refinement_objective="logit", hidden_width=None, queries_per_user=None, item_read_view=None)

    def endpoint(**updates):
        return training_epochs(Namespace(**(defaults | updates)))

    assert endpoint() == 64
    assert endpoint(method="nonlinear_response") == 200
    assert endpoint(initial_calibration=Path("prior")) == 8
    assert endpoint(method="kv_view", initial_calibration=Path("prior")) == 16
    for epochs in (32, 48, 64):
        assert endpoint(epochs=epochs) == epochs
        assert endpoint(epochs=epochs, small=True) == 2
    for epochs in (100, 200):
        assert endpoint(method="nonlinear_response", epochs=epochs, item_read_view=Path("fresh_item")) == epochs
        assert endpoint(method="nonlinear_response", epochs=epochs, small=True) == 2
    for incompatible in (dict(initial_calibration=Path("prior")), dict(recent_only_queries=True),
                         dict(rolling_scenes=True), dict(method="kv_view"), dict(epochs=100)):
        with pytest.raises(ValueError, match="--epochs"):
            endpoint(**(dict(epochs=32) | incompatible))
