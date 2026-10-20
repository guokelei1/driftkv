#!/usr/bin/env python3
"""Summarize the retained population's native operation mix, not GPU runtime."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from read_correction_2026_09.cost import CostModel, eager_read

LEDGER = ROOT / "results/query_relay_2026_10/workloads/stride16_candidates16_ledger.json"
QUERY_BATCH_NOTE = ("Retained native query calls may contain 32 candidates; current replay uses batches of 16. "
                    "For runtime weighting, split width 32 into two width-16 calls; candidate-linear FLOPs are unchanged.")


def length_bin(n):
    if n <= 128:
        return "1-128"
    if n <= 512:
        return "129-512"
    if n < 1024:
        return "513-1023"
    return "1024"


def summarize(ledger):
    cfg = ledger["model_config"]
    models = {name: CostModel(cfg["hidden_size"], cfg["num_layers"], cfg["num_heads"], name)
              for name in ("torch", "triton")}
    groups, totals, shapes = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    fields = {"rebuild": "once_rebuild_flops", "query": "native_query_flops", "append": "native_append_flops"}
    checks = {}
    for version, counts in ledger["by_release"].items():
        actual = Counter()
        for key, count in counts.items():
            if key.startswith("once_rebuild_length_"):
                op, n = "rebuild", int(key.removeprefix("once_rebuild_length_"))
                width = 1
                cost = models["triton" if n >= 1024 else "torch"].full_cache(n)
            elif key.startswith("native_query_shape_"):
                op = "query"
                n, width = map(int, key.removeprefix("native_query_shape_").split(":"))
                cost = eager_read(models["torch"], n, queries=width)
            elif key.startswith("native_append_shape_"):
                op = "append"
                n, width = map(int, key.removeprefix("native_append_shape_").split(":"))
                cost = models["triton" if n >= 1024 else "torch"].band_append(n, width, window_size=1024)
            else:
                continue
            values = dict(calls=count, rows_or_candidates=count * (n if op == "rebuild" else width),
                          flops=count * cost)
            groups[(op, length_bin(n))].update(values)
            totals[op].update(values)
            shapes[(op, n, width)].update(values)
            actual[fields[op]] += count * cost
        expected = ledger["native_by_release"][version]
        if any(actual[key] != expected[key] for key in fields.values()):
            raise AssertionError(f"native shape totals do not reproduce {version}")
        checks[version] = dict(actual)
    rows = [dict(operation=op, history_bin=band, **dict(values),
                 call_percent=100 * values["calls"] / totals[op]["calls"],
                 flop_percent=100 * values["flops"] / totals[op]["flops"])
            for (op, band), values in sorted(groups.items())]
    top = {op: [dict(history_rows=n, width=width, **dict(values))
                for (_, n, width), values in sorted(
                    ((key, value) for key, value in shapes.items() if key[0] == op),
                    key=lambda pair: pair[1]["flops"], reverse=True)[:8]] for op in fields}
    return dict(groups=rows, totals=dict(totals), top_shapes_by_flops=top,
                reproduced_native_components_by_release=checks)


def run(output):
    ledger = json.loads(LEDGER.read_text())
    result = summarize(ledger)
    output.mkdir(parents=True, exist_ok=False)
    result.update(status="complete_retained_population_shape_mix", users=ledger["users"],
        created_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        source=dict(path=str(LEDGER), sha256=hashlib.sha256(LEDGER.read_bytes()).hexdigest()),
        scope="Native operation shapes from actual 30k next-item ledger; no timing or correction-runtime estimate",
        limitations=["Query width is candidates per retained native call, not cross-user concurrency.",
                     QUERY_BATCH_NOTE,
                     "Correction conversion row counts and coded-prefix shapes need their own timing coverage.",
                     "Microbenchmark samples must be weighted by workload, not averaged equally across bins.",
                     "Batching, CPU/GPU transfer, capacity and queueing require separate measurements."])
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 固定30k next-item的原生操作组成", "",
             "从实际事件账重新按形状求和，与三版原生FLOPs逐项一致；没有运行模型或外推时间。", "",
             "| 操作 | 历史长度 | 调用数 | 调用占比 | FLOPs占比 |",
             "| --- | --- | ---: | ---: | ---: |"]
    for row in result["groups"]:
        lines.append(f"| {row['operation']} | {row['history_bin']} | {row['calls']} | {row['call_percent']:.2f}% | {row['flop_percent']:.2f}% |")
    lines += ["", "这些比例用于安排测量覆盖和按真实工作量加权，不能把短／长历史小样本等权均值当总体。",
              "query的width是候选批量，不是并发用户数。码转换、纠正读取、传输和共享训练／准备仍需各自计时，",
              "不能仅凭本表给出CacheTide时间盈亏点。完整高频形状见JSON。",
              "原账部分query调用含32候选，当前回放每批16；时间加权须拆成两次16候选调用，按候选线性计算的FLOPs不变。", ""]
    (output / "summary.md").write_text("\n".join(lines))
    print(json.dumps(dict(status=result["status"], output=str(output), users=ledger["users"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
