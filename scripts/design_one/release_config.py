"""Small configuration-selection hook for fresh Design 1 calibration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationSettings:
    hidden_width: int
    epochs: int
    compact_training: bool


def select_release_config(
    *, release_id: str, stage: str, fixed: CalibrationSettings,
) -> CalibrationSettings:
    """Return the fixed settings; no adaptive selection is implemented yet.

    The caller supplies the resolved settings, including any small-probe epoch
    limit. A future strategy can use release/stage context and calibration-only
    evidence here. Benchmark users, labels and scoring controls stay outside
    this interface.
    """
    return fixed
