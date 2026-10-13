"""Shared full-panel and cost checks for the retained Q/H figure sources."""
import math


def validate_point(point):
    if point["partition"] != "evaluation" or point["users"] != 3000:
        raise RuntimeError("formal summary requires the complete 3000-user evaluation panel")
    for name in ("full_auc", "reuse_auc", "baseline_auc", "relative_flops_percent", "recovery_percent"):
        if point[name] is None or not math.isfinite(point[name]):
            raise RuntimeError(f"undefined or nonfinite {name} must be reported, not silently omitted")
    if point["full_minus_reuse_flops"] <= 0 or point["full_auc"] <= point["reuse_auc"]:
        raise RuntimeError("frozen full-panel gap/compute denominator must remain positive")
    if point["kind"] == "measurement":
        if point["extra_flops"] != point["calibration_flops"] + point["correction_flops"]:
            raise RuntimeError("correction and calibration costs do not sum to extra compute")
    expected_x = 100 * point["extra_flops"] / point["full_minus_reuse_flops"]
    expected_y = 100 * (point["baseline_auc"] - point["reuse_auc"]) / (point["full_auc"] - point["reuse_auc"])
    if not math.isclose(point["relative_flops_percent"], expected_x, rel_tol=1e-10, abs_tol=1e-10):
        raise RuntimeError("reported cost uses a different denominator")
    if not math.isclose(point["recovery_percent"], expected_y, rel_tol=1e-10, abs_tol=1e-10):
        raise RuntimeError("reported AUC recovery does not match retained controls")
