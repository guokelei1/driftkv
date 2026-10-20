#!/usr/bin/env python3
"""Prepare and audit real development arrivals, without loading model weights."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np

from design_two.common import END, RELEASES
from design_two.prepare import load_history, load_panel
from query_relay_2026_10.workload import candidate_rows
from system_study_2026_10.events import build_trace
from unified_reuse_2026_09.common import sha256, write_json


def audit_trace(trace, histories, expected_rows):
    """Check consumers against arrived rows, independently of model execution.

    This accounts for logical history rows only, not GPU append bands or FLOPs.
    At a consumer timestamp, every strictly earlier row must have arrived and
    no tied row may have arrived. Advancing on demand must consume each row once.
    """
    start, end = trace[0].logical_time, trace[-1].logical_time
    initial = {u: int(np.searchsorted(h[0], start, side="left"))
               for u, h in histories.items()}
    arrived, consumed = initial.copy(), initial.copy()
    by_release = defaultdict(Counter)
    kinds, widths = Counter(), Counter()
    request_ids = set()
    digest = hashlib.sha256()

    def advance(uid, event):
        target = int(np.searchsorted(histories[uid][0], event.logical_time, side="left"))
        if arrived[uid] != target or consumed[uid] > target:
            raise RuntimeError(f"strict-past violation for uid={uid} event={event.sequence}")
        work = target - consumed[uid]
        consumed[uid] = target
        return work

    for event in trace:
        digest.update(json.dumps(asdict(event), sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
        kinds[event.kind] += 1
        counts = by_release[event.version]
        counts[event.kind + "_events"] += 1
        if event.kind == "behavior":
            indices = event.behavior_indices
            if indices != tuple(range(arrived[event.uid], arrived[event.uid] + len(indices))):
                raise RuntimeError("behavior arrivals changed original row order")
            arrived[event.uid] += len(indices)
            counts["behavior_rows"] += len(indices)
        elif event.kind in ("publication", "drain"):
            counts["logical_rows_consumed"] += sum(advance(uid, event) for uid in histories)
        elif event.kind == "recommendation":
            counts["logical_rows_consumed"] += advance(event.uid, event)
            counts["candidate_scores"] += len(event.requests)
            widths[len(event.requests)] += 1
            for row in event.requests:
                if row["request_id"] in request_ids:
                    raise RuntimeError("recommendation candidate was duplicated")
                request_ids.add(row["request_id"])
                if int(row["uid"]) != event.uid or int(row["query_timestamp"]) != event.logical_time:
                    raise RuntimeError("recommendation grouping changed user or logical time")
    expected_ids = [r["request_id"] for rows in expected_rows.values() for r in rows
                    if start <= int(r["query_timestamp"]) < end]
    if len(expected_ids) != len(set(expected_ids)) or request_ids != set(expected_ids):
        raise RuntimeError("trace did not preserve exactly the input candidate identities")
    total_rows = sum(int(np.searchsorted(h[0], end, side="left")) - initial[u]
                     for u, h in histories.items())
    if total_rows != sum(consumed[u] - initial[u] for u in histories):
        raise RuntimeError("final drain lost behavior rows")
    return dict(status="passed_cpu_causality_audit", users=len(histories),
        event_counts=dict(kinds), recommendation_rounds=kinds["recommendation"],
        candidate_scores=len(request_ids), behavior_rows=total_rows,
        candidates_per_round={str(k): v for k, v in sorted(widths.items())},
        by_release={v: dict(c) for v, c in by_release.items()},
        trace_sha256=digest.hexdigest(), logical_start=start, logical_end=end,
        scheduled_arrival_span_seconds=trace[-1].arrival_time,
        checks=["strict-past arrivals at every history consumer",
                "same-time behavior excluded from recommendation",
                "candidate identities preserved exactly once",
                "original behavior order and final drain preserved"],
        scope="Logical events only; no model execution, append FLOPs, quality or service measurements")


def run(args):
    if args.output.exists():
        raise FileExistsError(f"choose a fresh output directory: {args.output}")
    config = json.loads(args.config.read_text())
    if config["behavior_processing"] != "on_demand" or config["seed"] != 17:
        raise ValueError("initial trace uses the existing on-demand, seed17 protocol")
    panel_dir = ROOT / config["panel"]
    selection_path = ROOT / config["selection"]
    selection = json.loads(selection_path.read_text())
    candidate = ROOT / config["candidate"]
    if selection["panel_sha256"] != sha256(panel_dir / "binding.json"):
        raise ValueError("selected panel differs from retained engineering selection")
    if selection["freeze_sha256"] != sha256(candidate / "freeze.json"):
        raise ValueError("reference candidate differs from retained engineering selection")
    panel, groups, feedback = load_panel(panel_dir)
    chosen = selection["selected_uids"]
    if len(set(chosen)) != config["expected_users"] or len(chosen) != len(set(chosen)):
        raise ValueError("wrong or duplicate user selection")
    if not set(chosen).issubset(groups["evaluation"]):
        raise ValueError("trace users must retain the development evaluation role")
    histories = load_history(panel_dir, requested_uids=chosen).rows
    args.output.mkdir(parents=True, exist_ok=False)
    settings = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        command=sys.argv, settings=config, selected_uids=chosen,
        input_sha256={"config": sha256(args.config), "selection": sha256(selection_path),
                      "panel_binding": sha256(panel_dir / "binding.json"),
                      "candidate_freeze": sha256(candidate / "freeze.json")},
        sources={str(p.relative_to(ROOT)): sha256(p) for p in (
            Path(__file__), Path(__file__).with_name("events.py"),
            ROOT / "scripts/query_relay_2026_10/workload.py")},
        releases=list(RELEASES), end_time=END, model_weights_loaded=False,
        gpu_used=False, evidence_role="development_trace_preparation")
    write_json(args.output / "configuration.json", settings)
    summaries, preview = {}, {}
    for protocol in config["protocols"]:
        if protocol == "behavior_stride16_candidates16":
            if (config["stride"], config["candidates"]) != (16, 16):
                raise ValueError("protocol name and actual stride/candidates differ")
            requests = {u: candidate_rows(u, histories[u], stride=config["stride"],
                candidates=config["candidates"], known_vocab_size=panel["known_vocab_size"])
                for u in chosen}
        elif protocol == "original_feedback":
            requests = {u: feedback[u] for u in chosen}
        else:
            raise ValueError(f"unsupported initial protocol: {protocol}")
        trace = build_trace(histories, requests, RELEASES, END,
                            arrival_scale=config["arrival_scale"])
        summaries[protocol] = audit_trace(trace, histories, requests)
        examples = {e.sequence: e for e in trace[:6] + trace[-2:]
                    + [e for e in trace if e.kind == "publication"]}
        preview[protocol] = [asdict(examples[i]) for i in sorted(examples)]
    summary = dict(status="complete_cpu_trace_preparation", protocols=summaries,
        quality_measured=False, gpu_service_measured=False,
        storage="Only configuration, digest, summary and preview persisted; rebuild the trace from retained inputs",
        next_step="Connect existing synchronous model sessions; measure GPU completion and actual operation shapes")
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "preview.json", preview)
    lines = ["# 真实事件流准备检查", "", "仅 CPU 因果与分组检查，无新质量或 GPU 性能结论。", "",
             "| 协议 | 用户 | 推荐轮次 | 候选评分 | 行为行 | 检查 |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for protocol, row in summaries.items():
        lines.append(f"| {protocol} | {row['users']} | {row['recommendation_rounds']} | "
                     f"{row['candidate_scores']} | {row['behavior_rows']} | 通过 |")
    lines.extend(["", "到达间隔缩放仅用于准备 trace，未据此测量或选择生产负载。",
                  "每个 consumer 已核对严格过去；末尾 drain 覆盖全部区间内行为。",
                  "暂存配置、事件摘要、哈希和少量预览，不复制原始历史或全部候选。",
                  "下一步：接入真实会话，单独核对实际追加带、发布工作、GPU完成与内存。", ""])
    (args.output / "summary.md").write_text("\n".join(lines))
    print(json.dumps(dict(status=summary["status"], output=str(args.output),
                          users=len(chosen), protocols=list(summaries)), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
