#!/usr/bin/env python3
"""Document how the existing A/B probe panels differ before defining Q_main."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from build_yambda_release_snapshot import prepare_catalog_and_popularity
from train_yambda_theta0_medium import read_manifest


EDGES = ("theta0_theta1", "theta1_theta2")


def summary(ranks: np.ndarray) -> dict:
    output = {key: float(np.quantile(ranks, q)) for key, q in (("p10", .1), ("p50", .5), ("p90", .9))}
    output["mean"] = float(ranks.mean())
    return output


def main() -> None:
    raw = Path("data/raw/yambda/flat/50m/listens.parquet")
    _, popular = prepare_catalog_and_popularity(raw)
    rank = {int(item): index + 1 for index, item in enumerate(popular)}
    result = {
        "status": "panel_provenance_development",
        "shared_source": "foundation-only popularity-ranked catalog, filtered by each user's pre-foundation Listen+ seen set",
        "panel_A": "deterministic first 100 eligible items by popularity rank; no sampling seed",
        "panel_B": "deterministic next 100 eligible items after panel A; no sampling seed",
        "conclusion": "A and B are disjoint rank slices, not independent samples from a common proposal distribution; cross-panel results cannot estimate Monte-Carlo reliability.",
        "edges": {},
    }
    for edge in EDGES:
        a = {int(row["uid"]): row for row in read_manifest(Path(f"data/manifests/yambda50m_v2_cutover_probe_{edge}.jsonl"))}
        b = {int(row["uid"]): row for row in read_manifest(Path(f"data/manifests/yambda50m_v2_cutover_probe_panel_b_{edge}.jsonl"))}
        overlaps, ranks_a, ranks_b = [], [], []
        for uid in sorted(a):
            items_a, items_b = set(map(int, a[uid]["candidate_item_ids"])), set(map(int, b[uid]["candidate_item_ids"]))
            overlaps.append(len(items_a & items_b) / 100)
            ranks_a.extend(rank[item] for item in items_a)
            ranks_b.extend(rank[item] for item in items_b)
        result["edges"][edge] = {
            "states": len(a),
            "panel_size": 100,
            "target_injected": False,
            "user_conditioned_seen_filter": True,
            "mean_item_overlap": float(np.mean(overlaps)),
            "panel_a_popularity_rank": summary(np.asarray(ranks_a)),
            "panel_b_popularity_rank": summary(np.asarray(ranks_b)),
        }
    output = Path("results/data_audit/yambda50m_v2/panel_provenance_v1.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
