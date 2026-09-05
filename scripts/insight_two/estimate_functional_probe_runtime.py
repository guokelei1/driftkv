#!/usr/bin/env python3
"""Seal a linear 512-user makespan estimate after the estimator canary."""

from __future__ import annotations

import json

from insight_two.common import RESULT_ROOT, sha256_file
from insight_two.run_functional_probe_canary import CONTRACT, OUTPUT_ROOT


def main() -> None:
    source = OUTPUT_ROOT / "canary/summary.json"
    output = OUTPUT_ROOT / "resource_estimate.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    summary = json.loads(source.read_text(encoding="utf-8"))
    if not summary.get("passed") or summary.get("users") != 32:
        raise RuntimeError("a passing 32-user estimator canary is required")
    if summary.get("contract_sha256") != sha256_file(CONTRACT):
        raise RuntimeError("estimator canary contract differs")
    estimated_seconds = float(summary["elapsed_seconds"]) * 512 / 32
    payload = {
        "status": "functional_probe_discovery_resource_estimate",
        "contract_sha256": sha256_file(CONTRACT),
        "source_summary": str(source.relative_to(RESULT_ROOT.parent.parent.parent)),
        "source_summary_sha256": sha256_file(source),
        "method": "linear_scale_from_32_user_four_GPU_canary",
        "estimated_512_user_seconds": estimated_seconds,
        "estimated_512_user_minutes": estimated_seconds / 60,
        "peak_reserved_mib_upper_bound": float(summary["peak_reserved_mib"]),
        "interactive_limit_minutes": 30,
        "passes_interactive_limit": estimated_seconds / 60 <= 30,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
