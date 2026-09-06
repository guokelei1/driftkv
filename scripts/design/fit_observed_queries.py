#!/usr/bin/env python3
"""Calibrate with the release's already observed CC items and time gaps.

Labels are not loaded. Four legal query-time groups and sixteen candidates per
group retain the existing teacher budget; the state/teacher pipeline is reused.
"""

import argparse
import json
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
from design import run
from design.data import DAY, REQUEST_ROOT, ROOT


def observed_panels(history, uids, cutover):
    table = pq.read_table(REQUEST_ROOT / "requests_quality.parquet",
        filters=[("uid", "in", uids), ("time_block", "=", "matrix_horizon"),
                 ("target_known", "=", True), ("query_timestamp", ">=", cutover-14*DAY),
                 ("query_timestamp", "<", cutover)],
        columns=["uid", "query_timestamp", "item_idx", "request_id"])
    frame = table.to_pandas().sort_values(["uid", "query_timestamp", "request_id"])
    assert len(frame) and frame.query_timestamp.max() < cutover
    frame["gap"] = [int(row.query_timestamp-history.rows[int(row.uid)][0][
        np.searchsorted(history.rows[int(row.uid)][0], row.query_timestamp, side="left")-1])
        for row in frame.itertuples(index=False)]
    assert frame.gap.min() > 0
    groups = {int(uid):part for uid, part in frame.groupby("uid", sort=False)}
    panels = {}
    for uid in uids:
        # Quiet users share the fitting cohort's observed request distribution.
        # This remains an unlabeled query probe, not a fabricated feedback row.
        part = groups.get(uid, frame)
        random = np.random.default_rng(17+uid)
        indices = random.choice(len(part), 16, replace=len(part) < 16)
        gaps = sorted(part.gap.astype(int).tolist())
        panels[uid] = dict(items=part.iloc[indices].item_idx.astype(int).tolist(),
            gap_quartiles=[gaps[(len(gaps)-1)*q//4] for q in (1, 2, 3)],
            requests=len(part), scope="own observed requests" if uid in groups else "fitting-cohort observed requests")
    return panels, dict(requests=len(frame), users_with_feedback=len(groups),
        minimum_timestamp=int(frame.query_timestamp.min()), maximum_timestamp=int(frame.query_timestamp.max()),
        source="preceding14days, known exposed CC requests; no label column loaded")


ORIGINAL_SCENES = run.make_scenes


def observed_scenes(current, history, uids, states, early_states, cutover, ledger, previous, target, args):
    assert args.temporal and args.representation == "ridge" and not args.score_steps
    scenes = ORIGINAL_SCENES(current, history, uids, states, early_states, cutover, ledger, previous, target, args)
    panels, record = run.timed(lambda: observed_panels(history, uids, cutover), ledger, "calibration_query_selection")
    for scene in scenes:
        gap = float(scene.query_delta.max())
        panel = panels[scene.uid]
        offsets = [gap]+[min(gap, value) for value in panel["gap_quartiles"]]
        assert all(1 <= value <= gap for value in offsets)
        scene.candidates = run.tensor([panel["items"]*4], scene.candidates.device)
        scene.query_delta = run.tensor([offsets], scene.query_delta.device, floating=True).repeat_interleave(16, dim=1)
    run.write_json(ROOT / "results/design" / args.run_id / f"observed_queries_m{target}.json",
        dict(target=target, cutover=cutover, **record, panels=panels,
             times="release gap plus three empirical observed-gap quartiles, clipped to the visible prefix's legal gap"))
    return scenes


def main(cli):
    config = json.loads((ROOT / "results/design/v9_stratified_stable64_01/configuration.json").read_text())
    args = SimpleNamespace(**config)
    args.run_id, args.fit_users, args.lifetime_users = cli.run_id, cli.users, cli.users//4
    args.targets = cli.targets
    args.source_confidence = args.second_moments = False
    args.observed_calibration_queries = dict(items="16 sampled past CC request rows per UID, cohort pool for quiet users",
        gaps="past observed-gap quartiles clipped to each scene's legal prefix gap, plus release gap",
        labels_read=False, seed="unchanged17+uid", inference_change=False,
        costs="request IO and construction recorded as calibration_query_selection, in addition to existing fitting costs")
    args.prospective_resources = dict(expected_seconds=[25, 90] if cli.users == 16 else [50, 180],
        expected_gpu_mib=12000, estimate_basis="fixed16/64 calibration24/44s plus five fitting-cohort request scans")
    run.make_scenes = observed_scenes
    run.main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--users", type=int, choices=(16, 64), default=64)
    parser.add_argument("--targets", type=int, choices=(1, 2, 3, 4, 5), default=5)
    main(parser.parse_args())
