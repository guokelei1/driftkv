from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence

from .policy import cache_fidelity_recovery

D1_SAME_SLA_PROTOCOL = "d1_same_sla_baseline_development_v0"
D1_SAME_SLA_MANIFEST_SCHEMA = "d1_same_sla_candidate_manifest_v0"
D1_SAME_SLA_RECOVERY_TARGET = 0.5


def _fraction_tag(value: float) -> int:
    return int(round(100 * value))


def _canonical_sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def int_sequence_sha256(values: Sequence[int]) -> str:
    payload = b"".join(
        int(value).to_bytes(8, "little", signed=True)
        for value in values
    )
    return hashlib.sha256(payload).hexdigest()


def build_d1_same_sla_candidate_manifest(
    num_layers: int,
    *,
    rectangle_depth_fractions: Sequence[float] = (
        1 / 3,
        2 / 3,
        1.0,
    ),
    rectangle_recent_fractions: Sequence[float] = (0.25, 0.5),
) -> tuple[dict[str, object], str]:
    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    if not rectangle_depth_fractions or any(
        not 0 < float(value) <= 1
        for value in rectangle_depth_fractions
    ):
        raise ValueError("rectangle depth fractions must be in (0, 1]")
    if not rectangle_recent_fractions or any(
        not 0 < float(value) < 1
        for value in rectangle_recent_fractions
    ):
        raise ValueError("rectangle recent fractions must be in (0, 1)")
    depths = sorted(
        {
            max(
                1,
                min(
                    num_layers,
                    int(round(num_layers * float(fraction))),
                ),
            )
            for fraction in rectangle_depth_fractions
        }
    )
    recencies = sorted(
        {float(value) for value in rectangle_recent_fractions}
    )
    actions: dict[str, dict[str, object]] = {
        "current_projection": {
            "kind": "recent_suffix",
            "top_n_full": 0,
            "depth_fraction": 0.0,
            "recent_fraction": 0.0,
            "baseline_family": "shared_endpoint",
        }
    }
    suffix_names = []
    prefix_names = []
    rectangle_names = []
    interval_names = []
    for depth in range(1, num_layers):
        suffix_name = f"fixed_suffix_d{depth}"
        actions[suffix_name] = {
            "kind": "recent_suffix",
            "top_n_full": depth,
            "depth_fraction": depth / num_layers,
            "recent_fraction": 1.0,
            "baseline_family": "fixed_deep_suffix",
        }
        suffix_names.append(suffix_name)
        prefix_name = f"plain_prefix_d{depth}"
        actions[prefix_name] = {
            "kind": "interval",
            "start_layer": 0,
            "end_layer": depth - 1,
            "depth_fraction": depth / num_layers,
            "recent_fraction": 1.0,
            "baseline_family": "plain_progressive_prefix",
        }
        prefix_names.append(prefix_name)
    for depth in depths:
        for recent in recencies:
            name = (
                f"recent_rectangle_d{depth}_r"
                f"{_fraction_tag(recent)}"
            )
            actions[name] = {
                "kind": "recent_suffix",
                "top_n_full": depth,
                "depth_fraction": depth / num_layers,
                "recent_fraction": recent,
                "baseline_family": "recent_token_rectangles",
            }
            rectangle_names.append(name)
    for start in range(num_layers):
        for end in range(start, num_layers):
            if start == 0 and end == num_layers - 1:
                continue
            name = f"contiguous_interval_l{start + 1}_l{end + 1}"
            actions[name] = {
                "kind": "interval",
                "start_layer": start,
                "end_layer": end,
                "depth_fraction": (end - start + 1) / num_layers,
                "recent_fraction": 1.0,
                "baseline_family": "arbitrary_contiguous_intervals",
            }
            interval_names.append(name)
    actions["recompute"] = {
        "kind": "recent_suffix",
        "top_n_full": num_layers,
        "depth_fraction": 1.0,
        "recent_fraction": 1.0,
        "baseline_family": "exact_endpoint",
    }
    shared = ["current_projection"]
    families = {
        "fixed_deep_suffix": {
            "candidate_names": shared + suffix_names,
            "fallback": "recompute",
        },
        "plain_progressive_prefix": {
            "candidate_names": shared + prefix_names,
            "fallback": "recompute",
        },
        "recent_token_rectangles": {
            "candidate_names": shared + rectangle_names,
            "fallback": "recompute",
        },
        "arbitrary_contiguous_intervals": {
            "candidate_names": shared + interval_names,
            "fallback": "recompute",
        },
    }
    manifest: dict[str, object] = {
        "schema": D1_SAME_SLA_MANIFEST_SCHEMA,
        "protocol_status": "development",
        "num_layers": num_layers,
        "recovery_target": D1_SAME_SLA_RECOVERY_TARGET,
        "selection_signal": "probe cache fidelity",
        "selection_cost": "measured GPU migration cost",
        "endpoints": {
            "stale_reuse": "reuse",
            "current_projection": "current_projection",
            "exact": "recompute",
        },
        "rectangle_depth_fractions": [
            float(value) for value in rectangle_depth_fractions
        ],
        "rectangle_recent_fractions": recencies,
        "actions": actions,
        "families": families,
    }
    return manifest, _canonical_sha256(manifest)


def _candidate_measurement(
    configs: Mapping[str, Mapping[str, object]],
    name: str,
) -> dict[str, float | str]:
    reuse = configs["reuse"]
    exact = configs["recompute"]
    value = configs[name]
    recovery = cache_fidelity_recovery(
        float(value["cache_error_rel"]),
        float(reuse["cache_error_rel"]),
        float(exact["cache_error_rel"]),
    )
    return {
        "name": name,
        "cache_fidelity_recovery": recovery,
        "migration_ratio_to_recompute": float(
            value["migration_ratio_to_recompute"]
        ),
    }


def select_d1_same_sla_families(
    probe_configs: Mapping[str, Mapping[str, object]],
    test_configs: Mapping[str, Mapping[str, object]],
    manifest: Mapping[str, object],
    *,
    recovery_target: float = D1_SAME_SLA_RECOVERY_TARGET,
) -> dict[str, object]:
    if not 0 < recovery_target <= 1:
        raise ValueError("recovery_target must be in (0, 1]")
    required = {"reuse", "recompute", "current_projection"}
    if not required.issubset(probe_configs) or not required.issubset(
        test_configs
    ):
        raise ValueError("shared endpoint measurements are missing")
    families = manifest.get("families")
    if not isinstance(families, Mapping) or not families:
        raise ValueError("candidate manifest families are missing")
    output: dict[str, object] = {}
    for family_name, family_value in families.items():
        if not isinstance(family_value, Mapping):
            raise ValueError("candidate family is invalid")
        names = family_value.get("candidate_names")
        fallback = family_value.get("fallback")
        if (
            not isinstance(names, list)
            or not names
            or fallback != "recompute"
            or any(
                not isinstance(name, str)
                or name not in probe_configs
                or name not in test_configs
                for name in names
            )
        ):
            raise ValueError("candidate family measurements are incomplete")
        measurements = [
            _candidate_measurement(probe_configs, name)
            for name in names
        ]
        eligible = [
            value
            for value in measurements
            if math.isfinite(
                float(value["cache_fidelity_recovery"])
            )
            and float(value["cache_fidelity_recovery"])
            >= recovery_target
        ]
        fallback_used = not eligible
        if eligible:
            selected = min(
                eligible,
                key=lambda value: (
                    float(value["migration_ratio_to_recompute"]),
                    -float(value["cache_fidelity_recovery"]),
                    str(value["name"]),
                ),
            )
            selected_name = str(selected["name"])
        else:
            selected_name = "recompute"
        selected_probe = _candidate_measurement(
            probe_configs,
            selected_name,
        )
        selected_test = _candidate_measurement(
            test_configs,
            selected_name,
        )
        output[str(family_name)] = {
            "recovery_target": recovery_target,
            "candidate_names": list(names),
            "eligible_candidate_names": [
                str(value["name"]) for value in eligible
            ],
            "selected": selected_name,
            "fallback": "recompute",
            "fallback_used": fallback_used,
            "probe": {
                **selected_probe,
                "metrics": probe_configs[selected_name],
            },
            "test": {
                **selected_test,
                "metrics": test_configs[selected_name],
            },
        }
    return {
        "rule": (
            "minimum measured probe GPU cost meeting 50% cache-fidelity "
            "recovery within each family; otherwise exact"
        ),
        "task_labels_used_for_selection": False,
        "recovery_target": recovery_target,
        "families": output,
    }
