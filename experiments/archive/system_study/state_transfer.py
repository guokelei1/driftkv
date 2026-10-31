"""Measured page traffic for actual final user states, separate from service.

No compression, cache policy or concurrency is introduced. A page is the full
K/V plus the compact buffer, when present. A hypothetical consumer can append
and score while that page is resident, so callers must not charge two misses
for those stages without a matching eviction policy.
"""
from collections import defaultdict
import time

import torch

from experiments.archive.system_study.batched_session import grouped_uids


@torch.no_grad()
def profile_state_transfer(sessions, device, *, batches=(1, 8), repeats=3):
    started = time.perf_counter()
    pages = {}
    for uid, session in sessions.items():
        fields = dict(k=session.cache.k, v=session.cache.v)
        if getattr(session, "state", None) is not None:
            fields["codes"] = session.state.codes
        pages[uid] = {name: value.detach().cpu().pin_memory() for name, value in fields.items()}
    torch.cuda.synchronize(device)
    backing_prepare_seconds = time.perf_counter()-started
    results = []
    for maximum in batches:
        groups = grouped_uids(sessions, lambda uid: sessions[uid].cache.seq_len, maximum)
        for members in groups:
            first = pages[members[0]]
            started = time.perf_counter()
            input_stage, output_stage = {}, {}
            for name, value in first.items():
                shape = list(value.shape)
                shape[1] = len(members)
                input_stage[name] = torch.empty(shape, dtype=value.dtype, pin_memory=True)
                output_stage[name] = torch.empty(shape, dtype=value.dtype, pin_memory=True)
            stage_allocation_seconds = time.perf_counter()-started
            payload = {name: value.numel()*value.element_size() for name, value in input_stage.items()}
            for repeat in range(-1, repeats):
                torch.cuda.synchronize(device)
                all_start = time.perf_counter()
                started = time.perf_counter()
                for name, staged in input_stage.items():
                    for i, uid in enumerate(members):
                        staged[:, i:i+1].copy_(pages[uid][name])
                host_pack_seconds = time.perf_counter()-started
                before = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                gpu = {name: value.to(device, non_blocking=True) for name, value in input_stage.items()}
                torch.cuda.synchronize(device)
                h2d_seconds = time.perf_counter()-started
                started = time.perf_counter()
                for name, value in gpu.items():
                    output_stage[name].copy_(value, non_blocking=True)
                torch.cuda.synchronize(device)
                d2h_seconds = time.perf_counter()-started
                started = time.perf_counter()
                for name, staged in output_stage.items():
                    for i, uid in enumerate(members):
                        pages[uid][name].copy_(staged[:, i:i+1])
                host_scatter_seconds = time.perf_counter()-started
                wall_seconds = time.perf_counter()-all_start
                peak = torch.cuda.max_memory_allocated(device)-before
                # Exact transport identity; outside every reported timed phase.
                for name in first:
                    if not torch.equal(input_stage[name], output_stage[name]):
                        raise AssertionError("state page changed in host/device transport")
                if repeat >= 0:
                    results.append(dict(max_batch=maximum, batch=len(members), uids=list(members),
                        version=f"v{sessions[members[0]].trajectory.version}", history_length=sessions[members[0]].cache.seq_len,
                        repeat=repeat, host_pack_seconds=host_pack_seconds, h2d_seconds=h2d_seconds,
                        d2h_seconds=d2h_seconds, host_scatter_seconds=host_scatter_seconds,
                        wall_seconds=wall_seconds, field_bytes=payload, h2d_bytes=sum(payload.values()),
                        d2h_bytes=sum(payload.values()), peak_gpu_increment_bytes=peak,
                        stage_allocation_seconds=stage_allocation_seconds, pinned=True,
                        host_stage_reused=True, gpu_allocation_in_h2d=True, exact_round_trip=True))
                del gpu
    return dict(backing_prepare_seconds=backing_prepare_seconds, rows=results,
        resident_backing_bytes=sum(t.numel()*t.element_size() for page in pages.values() for t in page.values()),
        scope="Independent full-state nonoverlapped transfer coefficients on actual final sessions; no CPU-backing service integration or measured cache hit rate",
        miss_unit="One user history-consumer visit; append and score can share one page-in/page-out",
        preparation="Pinned page backing and staging allocation separate; each measured visit includes host pack, GPU allocation+H2D, D2H and host scatter; no model compute")
