#!/usr/bin/env python3
"""Check finite native-write dependency clearance on real calibration histories."""

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from design.data import DATASET, DAY, fixed_split, frozen_model, histories  # noqa: E402
from design.run import append_events, cache_at, event_range, timed, write_json  # noqa: E402


@torch.no_grad()
def main(run_id):
    out = ROOT / "results/design" / run_id
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    capacity, layers = 1024, 6
    opened = fixed_split()["calibration"][:128]
    table = pq.read_table(DATASET.parent / "users.parquet",
                          columns=["uid", "n_theta0"], filters=[("uid", "in", opened)])
    counts = dict(zip(table["uid"].to_pylist(), table["n_theta0"].to_pylist(), strict=True))
    uids = [uid for uid in opened if counts[uid] >= capacity * (layers + 1)][:4]
    assert len(uids) == 4, "The opened calibration prefix lacks four eligible real histories"
    reference = ROOT / "results/design/v5_budget64_01/configuration.json"
    previous_config = json.loads(reference.read_text())
    sources = [Path(__file__), ROOT / "scripts/design/run.py", ROOT / "scripts/design/data.py",
               *[ROOT / "src/hstu_kvcache/models" / name for name in
                 ("hstu.py", "block.py", "attention.py", "embeddings.py", "state_transition.py")]]
    write_json(out / "configuration.json", dict(
        purpose="real native-write layer clearance; no candidate labels or learned fitting",
        selection="first four of the opened first128 calibration UIDs with n_theta0 >= 7168",
        uids=uids, n_theta0={str(uid): counts[uid] for uid in uids},
        parent=0, current=1, latest_visible_timestamp=231 * DAY, capacity=capacity,
        real_writes=layers * capacity, replay_chunk=128,
        claim="layer j (zero-based) clears by (j+1)*1024 native writes; arbitrary initial cache error",
        comparison="Parent and Current Exact on the same real prefix, then identical Current native writes",
        expected_seconds=[15, 90], expected_peak_gib=4, confirmation_read=False,
        teacher_scope="diagnostic only; no teacher state enters an executable method",
        reference_configuration_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        checkpoint_inputs={k: previous_config["checkpoint_inputs"][k] for k in ("v0", "v1")},
        checkpoint_seals={k: previous_config["checkpoint_seals"][k] for k in ("v0", "v1")},
        dataset_input=previous_config["dataset_input"],
        item_mapping_input=previous_config["item_mapping_input"],
        split_sha256=previous_config["split_sha256"],
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    ))
    for path in sources:
        (out / path.name).write_bytes(path.read_bytes())
    ledger = defaultdict(float)
    history = timed(lambda: histories(uids, 231), ledger, "history_io")
    previous = timed(lambda: frozen_model(0, device), ledger, "model_io")
    current = timed(lambda: frozen_model(1, device), ledger, "model_io")
    assert len(current.blocks) == layers and not current.cfg.relative_position_bias
    rows = []
    for uid in uids:
        stamps = history.rows[uid][0]
        end = np.searchsorted(stamps, 231 * DAY, side="left")
        snapshot = int(stamps[end - layers * capacity])
        source, times = timed(partial(cache_at, previous, history, uid, snapshot), ledger, "parent_prefix")
        exact, exact_times = timed(partial(cache_at, current, history, uid, snapshot), ledger, "teacher_prefix")
        assert source.seq_len == exact.seq_len == capacity
        assert np.array_equal(times, exact_times)
        # A timestamp tie can add earlier same-time events. Replay the first
        # 6144 actual events, never insert a candidate query inside that group.
        events = event_range(history, uid, snapshot, 231 * DAY)[:layers * capacity]
        assert len(events) == layers * capacity
        last = int(times[-1])
        for step in range(layers + 1):
            if step:
                part = events[(step - 1) * capacity:step * capacity]
                source = timed(partial(append_events, current, source, part, last, 128), ledger, "source_native_replay")
                exact = timed(partial(append_events, current, exact, part, last, 128), ledger, "teacher_native_replay")
                last = part[-1][0]
            differences = {field: (getattr(source, field) - getattr(exact, field)).abs()
                           .flatten(1).amax(1).tolist() for field in ("k", "v")}
            cleared_equal = all(torch.equal(getattr(source, field)[:step], getattr(exact, field)[:step])
                                for field in ("k", "v"))
            row = dict(uid=uid, snapshot=snapshot, real_writes=step * capacity,
                       expected_cleared_layers=step, expected_layers_bitwise_equal=cleared_equal,
                       layer_max_abs=differences)
            rows.append(row)
            print(json.dumps(row), flush=True)
    nontrivial = any(max(row["layer_max_abs"]["k"] + row["layer_max_abs"]["v"]) > 0
                     for row in rows if row["real_writes"] == 0)
    passed = nontrivial and all(row["expected_layers_bitwise_equal"] for row in rows)
    result = dict(status="diagnostic_complete" if passed else "diagnostic_failed", passed=passed,
                  initial_state_difference_observed=nontrivial, comparisons=rows,
                  ledger_seconds=dict(ledger), elapsed_seconds=time.perf_counter() - start,
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / (1 << 20),
                  confirmation_read=False,
                  limitations=["four real pre-release calibration histories, frozen V0/V1 seed17",
                               "native writes and fixed Current model, cap1024, six layers, no position bias",
                               "supports the structural bound; does not measure quality or correction policy"])
    write_json(out / "summary.json", result)
    assert passed, "Observed cache differences contradict the prospective clearance bound"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    main(parser.parse_args().run_id)
