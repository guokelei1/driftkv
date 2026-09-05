#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_yambda500m_foundation_raw import load_histories  # noqa: E402
from insight_one_locality.common import (  # noqa: E402
    CONTRACT,
    CUTOVER_DAYS,
    DATASET,
    DAY,
    HISTORY,
    INPUT_MANIFEST,
    KNOWN_ITEMS,
    OOV_BUCKETS,
    POPULATION,
    USERS,
    candidate_panel,
    histories_at_cutover,
    sha256_file,
    verify_contract,
)


def select_population(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    listens = (DATASET.parent / dataset["shared_listens_glob"]).resolve()
    connection = duckdb.connect()
    try:
        frame = connection.execute(
            """
            WITH eligible AS (
              SELECT u.uid, u.selector_rank, count(*) AS events_before_first_cutover
              FROM read_parquet(?) u JOIN read_parquet(?) l USING(uid)
              WHERE l.timestamp < ?
              GROUP BY u.uid, u.selector_rank
              HAVING count(*) >= ?
            )
            SELECT uid, selector_rank, events_before_first_cutover
            FROM eligible
            ORDER BY selector_rank, uid
            LIMIT ?
            """,
            [str(USERS), str(listens), CUTOVER_DAYS[0] * DAY, HISTORY, count],
        ).fetchdf()
    finally:
        connection.close()
    if len(frame) != count:
        raise RuntimeError(f"only {len(frame)} users meet the frozen full-history rule")
    return (
        frame.uid.to_numpy(dtype=np.int64),
        frame.selector_rank.to_numpy(dtype=np.int64),
        frame.events_before_first_cutover.to_numpy(dtype=np.int64),
    )


def write_npz(path: Path, **arrays) -> None:
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=INPUT_MANIFEST)
    parser.add_argument("--history-threads", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_name(args.output.name + ".partial").exists():
        raise FileExistsError(f"refusing to overwrite existing input manifest: {args.output}")
    contract = verify_contract()
    partial = args.output.with_name(args.output.name + ".partial")
    partial.mkdir(parents=True)
    try:
        uids, selector_ranks, event_counts = select_population(POPULATION)
        if len(np.unique(uids)) != POPULATION:
            raise RuntimeError("selected population contains duplicate UIDs")
        print(json.dumps({"phase": "load_histories", "users": len(uids)}), flush=True)
        history = load_histories(
            uids.tolist(),
            oov_buckets=OOV_BUCKETS,
            dataset_path=DATASET,
            known_vocab_size=KNOWN_ITEMS,
            end_timestamp=(CUTOVER_DAYS[-1] + 1) * DAY,
            threads=args.history_threads,
        )
        candidates, modes, audits = [], [], {}
        for edge, cutover_day in enumerate(CUTOVER_DAYS):
            edge_name = f"v{edge}_to_v{edge + 1}"
            print(json.dumps({"phase": "candidate_panel", "edge": edge_name}), flush=True)
            _, items, _, _, _ = histories_at_cutover(history, uids, cutover_day * DAY)
            panel, panel_modes, audit = candidate_panel(items)
            candidates.append(panel)
            modes.append(panel_modes)
            audits[edge_name] = audit
        candidate_array = np.stack(candidates)
        mode_array = np.stack(modes)
        write_npz(
            partial / "population.npz",
            uids=uids,
            selector_ranks=selector_ranks,
            events_before_first_cutover=event_counts,
        )
        write_npz(
            partial / "candidate_panels.npz",
            uids=uids,
            cutover_days=np.asarray(CUTOVER_DAYS, dtype=np.int64),
            candidates=candidate_array,
            modes=mode_array,
        )
        (partial / "candidate_panel_audit.json").write_text(
            json.dumps(audits, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        artifacts = {}
        for name in ("population.npz", "candidate_panels.npz", "candidate_panel_audit.json"):
            path = partial / name
            artifacts[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        descriptor = {
            "status": "medium_insight1_locality_inputs_complete",
            "contract": str(CONTRACT.relative_to(ROOT)),
            "contract_sha256": sha256_file(CONTRACT),
            "users": POPULATION,
            "history_positions": HISTORY,
            "edges": list(contract["scope"]["edges"]),
            "candidate_shape": list(candidate_array.shape),
            "labels_read": False,
            "artifacts": artifacts,
        }
        (partial / "manifest.json").write_text(
            json.dumps(descriptor, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        partial.rename(args.output)
        print(json.dumps(descriptor, indent=2), flush=True)
    except BaseException:
        # Preserve partial evidence for audit; never silently overwrite or delete it.
        raise


if __name__ == "__main__":
    main()
