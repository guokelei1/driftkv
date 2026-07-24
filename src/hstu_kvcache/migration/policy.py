from __future__ import annotations

import math


def cache_fidelity_recovery(
    cache_error: float,
    reuse_error: float,
    full_error: float,
) -> float:
    denominator = reuse_error - full_error
    if not math.isfinite(denominator) or denominator <= 1e-12:
        return float("nan")
    return (reuse_error - cache_error) / denominator


def select_minimum_cost_actions(
    configs: dict[str, dict],
    fidelity_targets: tuple[float, ...],
    reuse_name: str = "reuse",
    full_name: str = "recompute",
) -> dict[str, dict]:
    if reuse_name not in configs or full_name not in configs:
        raise ValueError("reuse and full configurations are required")
    reuse_error = float(configs[reuse_name]["cache_error_rel"])
    full_error = float(configs[full_name]["cache_error_rel"])
    annotated = {}
    for name, value in configs.items():
        recovery = cache_fidelity_recovery(
            float(value["cache_error_rel"]),
            reuse_error,
            full_error,
        )
        annotated[name] = {
            "cache_fidelity_recovery": recovery,
            "migration_ratio_to_recompute": float(
                value["migration_ratio_to_recompute"]
            ),
        }
    output = {}
    for target in fidelity_targets:
        if not 0 < target <= 1:
            raise ValueError("fidelity targets must be in (0, 1]")
        eligible = [
            name
            for name, value in annotated.items()
            if math.isfinite(value["cache_fidelity_recovery"])
            and value["cache_fidelity_recovery"] >= target
        ]
        selected = min(
            eligible,
            key=lambda name: (
                annotated[name]["migration_ratio_to_recompute"],
                -annotated[name]["cache_fidelity_recovery"],
                name,
            ),
            default=full_name,
        )
        output[str(target)] = {
            "selected": selected,
            "target": target,
            **annotated[selected],
        }
    return output


def select_maximum_fidelity_actions(
    configs: dict[str, dict],
    budget_targets: tuple[float, ...],
    reuse_name: str = "reuse",
    full_name: str = "recompute",
) -> dict[str, dict]:
    if reuse_name not in configs or full_name not in configs:
        raise ValueError("reuse and full configurations are required")
    reuse_error = float(configs[reuse_name]["cache_error_rel"])
    full_error = float(configs[full_name]["cache_error_rel"])
    annotated = {
        name: {
            "cache_fidelity_recovery": cache_fidelity_recovery(
                float(value["cache_error_rel"]),
                reuse_error,
                full_error,
            ),
            "migration_ratio_to_recompute": float(
                value["migration_ratio_to_recompute"]
            ),
        }
        for name, value in configs.items()
    }
    output = {}
    for budget in budget_targets:
        if not 0 <= budget <= 1:
            raise ValueError("budget targets must be in [0, 1]")
        eligible = [
            name
            for name, value in annotated.items()
            if value["migration_ratio_to_recompute"] <= budget
            and math.isfinite(value["cache_fidelity_recovery"])
        ]
        selected = max(
            eligible,
            key=lambda name: (
                annotated[name]["cache_fidelity_recovery"],
                -annotated[name]["migration_ratio_to_recompute"],
                name,
            ),
            default=reuse_name,
        )
        output[str(budget)] = {
            "selected": selected,
            "budget": budget,
            **annotated[selected],
        }
    return output
