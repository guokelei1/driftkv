"""Paths and essential provenance for the second correction exploration."""
from pathlib import Path
import json

from selective_recompute_2026_09.common import edge_name, sha256, signature, write_json

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "results/read_correction_2026_09/v2"
PANEL_ROOT = ROOT / "results/selective_recompute_2026_09/panels"
RESERVATIONS = OUTPUT / "panels"
PLAN = ROOT / "configs/read_correction_2026_09/v2/plan.json"
METHODS = ("query_only", "history_conditioned")


def plan():
    return json.loads(PLAN.read_text())


def method_directory(method, scale, edge, *, output_root=OUTPUT):
    return Path(output_root) / method / scale / edge_name(edge)


def runtime_directory(output_root, scale, edge):
    return Path(output_root) / "runtime" / scale / edge_name(edge)


def sources():
    from read_correction_2026_09.common import sources as original_sources
    paths = original_sources()
    paths[str(PLAN.relative_to(ROOT))] = sha256(PLAN)
    return paths


def build_correction(kind, config, state_dict=None):
    from hstu_kvcache.read_correction.query_only import QueryCorrection
    from hstu_kvcache.read_correction.history_conditioned.v2 import HistoryCorrectionV2
    cls = {"query_only": QueryCorrection, "history_conditioned_v2": HistoryCorrectionV2}[kind]
    module = cls(**config)
    if state_dict is not None:
        example = next(iter(state_dict.values()))
        module.to(device=example.device, dtype=example.dtype)
        module.load_state_dict(state_dict)
    return module
