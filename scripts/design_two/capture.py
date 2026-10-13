"""Capture matched native/Current histories for pre-release calibration."""
from collections import defaultdict
import json
import torch
from design_two.common import RELEASES, ledger_flops
from hstu_kvcache.design_two.replay import NativeTrajectory
from read_correction_2026_09.cost import CostModel
from read_correction_2026_09.v2.calibrate import mixed_candidates


@torch.no_grad()
def capture_rows(models, histories, uids, target, known):
    """CPU keeps one source/teacher pair per calibration UID."""
    rows, costs, records = {}, defaultdict(int), []
    current = models[target]
    device = next(current.parameters()).device
    cutover = dict(RELEASES)[target]
    teacher_costs = {name: CostModel(current.cfg.hidden_size, len(current.blocks), current.cfg.num_heads, name)
                     for name in ("torch", "triton")}
    for index, uid in enumerate(uids):
        trajectory = NativeTrajectory(models["v2"], 2, histories.rows[uid], RELEASES[0][1])
        for version, at in RELEASES:
            trajectory.switch(models[version], int(version[1:]), at, rebuild=False)
            if version == target:
                break
        events = trajectory.retained_events
        if not trajectory.cache.seq_len:
            raise ValueError(f"calibration UID {uid} has no pre-release history")
        teacher = trajectory.fresh_cache()
        ids = torch.as_tensor(events[1], dtype=torch.long, device=device)[None]
        rows[uid] = dict(parent=trajectory.cache.to("cpu"), teacher=teacher.to("cpu"),
            candidates=torch.tensor(mixed_candidates(uid, events[1], known, 16, seed=17)),
            query_delta=float(cutover - trajectory.last_timestamp),
            item_features=current.lookup_item_embeddings(ids).detach().cpu())
        costs["source_trajectory_flops"] += sum(ledger_flops(trajectory.ledger, current, "triton").values())
        costs["teacher_cache_capture_flops"] += teacher_costs[
            "triton" if teacher.seq_len == current.cfg.max_seq_len else "torch"].full_cache(teacher.seq_len)
        records.append(dict(uid=uid, history_length=teacher.seq_len,
            last_history_timestamp=int(trajectory.last_timestamp),
            producer_counts={str(int(p)): int((trajectory.producers == p).sum())
                             for p in set(trajectory.producers.tolist())}))
        if (index + 1) % 16 == 0 or index + 1 == len(uids):
            print(json.dumps(dict(status="cutover_capture", target=target, users=index + 1,
                                  total=len(uids))), flush=True)
    return rows, costs, records
