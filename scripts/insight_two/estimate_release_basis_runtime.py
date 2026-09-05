#!/usr/bin/env python3
"""Seal the release-basis 512-user runtime estimate."""

from __future__ import annotations

import json

from insight_two.common import RESULT_ROOT, sha256_file
from insight_two.run_release_basis_diagnostic import CONTRACT, OUTPUT_ROOT


def main() -> None:
    source = OUTPUT_ROOT / "canary_v2/summary.json"
    output = OUTPUT_ROOT / "resource_estimate.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    summary = json.loads(source.read_text(encoding="utf-8"))
    if not summary.get("passed") or summary.get("contract_sha256") != sha256_file(CONTRACT):
        raise RuntimeError("valid release-basis canary under current contract required")
    seconds = float(summary["elapsed_seconds"]) * 512 / 32
    payload = {
        "status": "release_basis_discovery_resource_estimate",
        "contract_sha256": sha256_file(CONTRACT),
        "source_summary": str(source.relative_to(RESULT_ROOT.parent.parent.parent)),
        "source_summary_sha256": sha256_file(source),
        "method": "linear_scale_from_32_user_four_GPU_canary",
        "estimated_512_user_seconds": seconds,
        "estimated_512_user_minutes": seconds / 60,
        "interactive_limit_minutes": 30,
        "passes_interactive_limit": seconds / 60 <= 30,
        "peak_reserved_mib_upper_bound": float(summary["peak_reserved_mib"]),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
