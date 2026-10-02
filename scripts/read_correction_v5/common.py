"""Isolated v5 Q-feature and mixed-history development settings."""
from pathlib import Path

from read_correction_v4.common import ROOT, PANEL_ROOT, RESERVATIONS, edge_name, sha256, write_json
from read_correction_v4.common import configuration as previous_configuration

OUTPUT = ROOT / "results/read_correction_2026_09/v5"


def configuration():
    return {**previous_configuration(), "revision": "v5",
        "authorization": "2026-09-29 user: optimize query-only and explicit-history correction, small tests then all15edges with4GPUs and tmux",
        "candidate_mode": "mixed_recent_uniform", "calibration_users": 128,
        "validation_users": 16, "query_ridge": .01, "feature_mode": "head_phi",
        "nonlinear_epochs": 6, "nonlinear_learning_rate": .001, "weight_decay": .0001,
        "nonlinear_width_rule": "model dimension / 2",
        "mixed_append_targets": [0, 256, 512, 896],
        "fit_objective": "query: same-actual-query read residual ridge; history: normalized token and read residual",
        "selection": "four fixed Medium/Large development edges; no Max quality tuning",
        "evaluation_users": 3000, "unit_users": 256}


def sources():
    from read_correction_v4.common import sources as prior
    result = prior()
    for folder in (ROOT / "scripts/read_correction_v5", ROOT / "src/hstu_kvcache/read_correction_v5"):
        for path in sorted(folder.rglob("*.py")):
            result[str(path.relative_to(ROOT))] = sha256(path)
    return result
