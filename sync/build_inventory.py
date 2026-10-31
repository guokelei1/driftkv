#!/usr/bin/env python3
"""Build a categorized, numbered inventory of ignored data/model payloads."""

from __future__ import annotations

import datetime as dt
import pathlib
import subprocess
import sys


GROUPS = [
    ("DATA", "数据", "D001", "Yambda 原始数据", "Yambda 原始训练数据及下载缓存。用于重建或核对输入数据。", "data/raw/yambda"),
    ("DATA", "数据", "D002", "RecFlow 原始数据", "RecFlow 原始数据包；处理后数据见 D009。", "data/raw/recflow"),
    ("DATA", "数据", "D003", "原始数据下载记录", "数据下载清单和日志，体积较小，用于追溯下载来源。", "data/raw"),
    ("DATA", "数据", "D004", "Yambda 500M/Medium/大型请求清单", "Yambda 500M 规模、Medium/大型请求及流式窗口清单。", "data/manifests"),
    ("DATA", "数据", "D005", "Yambda 5B Max 请求清单", "Yambda 5B Max 规模的请求和人口清单。", "data/manifests"),
    ("DATA", "数据", "D006", "开发划分清单", "早期设计开发用的用户划分清单。", "data/manifests"),
    ("DATA", "数据", "D007", "Yambda 500M 处理后数据", "Yambda 500M Unified 数据，包括共享数据与 Small/Medium/Large 切分。", "data/processed/yambda500m_unified_v1"),
    ("DATA", "数据", "D008", "Yambda 5B Max 处理后数据", "Yambda 5B、20 万用户 Max 规模的共享数据与 Max 切分。", "data/processed/yambda5b_max_200k_v1"),
    ("DATA", "数据", "D009", "RecFlow 处理后数据", "RecFlow 模型训练和评测所用的处理后数组与人口数据。", "data/processed/recflow_v1"),
    ("DATA", "数据", "D010", "Query Relay 特征库", "Query Relay 读取校正实验使用的原始特征库及本地变体。", "results/query_relay_2026_10/feature_banks"),
    ("YAMBDA", "模型 / Yambda", "Y001", "Medium 规模训练模型", "Yambda 500M Medium，seed17，v0–v5 六个连续发布版本的训练 checkpoint。", "results/unified_training_2026_09/medium/seed17/checkpoints"),
    ("YAMBDA", "模型 / Yambda", "Y002", "Large 规模训练模型", "Yambda 500M Large，seed17，v0–v5 六个连续发布版本的训练 checkpoint。", "results/unified_training_2026_09/large/seed17/checkpoints"),
    ("YAMBDA", "模型 / Yambda", "Y003", "Max 规模训练模型", "Yambda 5B Max，seed17，v0–v5 六个连续发布版本的训练 checkpoint。", "results/unified_training_2026_09/max/seed17/checkpoints"),
]

# Query Relay source/publication snapshots are Yambda model payloads, grouped
# by candidate so each numbered entry has a clear identity.
QUERY_CANDIDATES = [
    "A_constantgate6_rms_roll4", "A_firstgate6_rms_roll4",
    "A_itemgate6_rms_roll4", "A_localorigin6_rms_roll4",
    "A_origin6_rms_roll4", "B_balanced6_rms_roll4", "B_origin6_rms_roll4",
    "B_r6_v3", "C_balanced6_rms_roll4",
]
for i, candidate in enumerate(QUERY_CANDIDATES, 4):
    GROUPS.append(("YAMBDA", "模型 / Yambda", f"Y{i:03d}",
                   f"Query Relay 候选快照：{candidate}",
                   "候选读取校正实验保留的源模型和/或发布版本快照；与 Y001–Y003 的基础训练 checkpoint 分开。",
                   f"results/query_relay_2026_10/probes/{candidate}"))

REC_FLOW_RELEASES = [
    ("A", "epoch3", "六层 RecFlow 基模型，A 阶段完整三轮训练终点。"),
    ("B", "epoch1", "RecFlow B 阶段模型；从 A 继续训练并保留优化器状态。"),
    ("C", "epoch1", "RecFlow C 阶段模型；从 B 继续训练并保留优化器状态。"),
    ("D", "epoch1", "RecFlow D 阶段模型；从 C 继续训练并保留优化器状态。"),
    ("E", "epoch1", "RecFlow E 阶段模型；从 D 继续训练并保留优化器状态。"),
    ("F", "epoch1", "RecFlow F 阶段模型；从 E 继续训练并保留优化器状态。"),
]
for i, (release, epoch, note) in enumerate(REC_FLOW_RELEASES, 1):
    GROUPS.append(("REC_FLOW", "模型 / RecFlow", f"R{i:03d}",
                   f"RecFlow 六层模型 {release}（{epoch}）", note,
                   f"results/recflow/development/window_6l_expanded_u4096_seed17/{release}/{release}_{epoch}"))

GROUPS.sort(key=lambda group: {"DATA": 0, "YAMBDA": 1, "REC_FLOW": 2}[group[0]])


def ignored_files(root: pathlib.Path, rel: str) -> list[pathlib.Path]:
    base = root / rel
    if not base.exists():
        return []
    found = []
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", "--", relative], cwd=root,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
        if ignored:
            # Parent roots can overlap; include only the payload types in scope.
            if rel.startswith("data/") or path.suffix == ".pt":
                found.append(path)
    return sorted(found)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"201", "202"}:
        raise SystemExit("Usage: build_inventory.py {201|202}")
    machine = sys.argv[1]
    root = pathlib.Path(__file__).resolve().parents[1]
    output = root / "sync" / machine

    lines = [
        f"# Machine {machine} inventory",
        f"Generated: {dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Repository: {root}",
        "编号按类别预留 001–100；FILE 行供 copy_to.sh 读取，路径相对于仓库根目录。",
        "",
        "## 数据（D001–D100）",
    ]
    emitted = set()
    for kind, title, code, name, note, rel in GROUPS:
        files = ignored_files(root, rel)
        if code == "D004":
            files = [p for p in files if "/yambda500m_" in p.as_posix()]
        elif code == "D005":
            files = [p for p in files if "/yambda5b_" in p.as_posix()]
        elif code == "D006":
            files = [p for p in files if "/evokv_design_" in p.as_posix()]
        if kind == "DATA":
            # Keep the manifest groups distinct while omitting duplicate paths
            # from broad parent directories.
            files = [p for p in files if p.relative_to(root).as_posix() not in emitted]
        elif kind in {"YAMBDA", "REC_FLOW"}:
            files = [p for p in files if p.suffix == ".pt" and p.relative_to(root).as_posix() not in emitted]
            if kind == "YAMBDA" and int(code[1:]) >= 4:
                files = [p for p in files if "/source_v" in p.as_posix() or "/publication/" in p.as_posix()]
        if not files:
            continue
        emitted.update(p.relative_to(root).as_posix() for p in files)
        if kind == "YAMBDA" and not any("## 模型 / Yambda" in line for line in lines):
            lines.extend(["", "## 模型 / Yambda（Y001–Y100；规模和实验候选分别编号）"])
        if kind == "REC_FLOW" and not any("## 模型 / RecFlow" in line for line in lines):
            lines.extend(["", "## 模型 / RecFlow（R001–R100）"])
        size = sum(p.stat().st_size for p in files)
        lines.extend(["", f"### {code} — {name}", note,
                      f"路径：`{rel}`；{len(files)} 个文件；合计 {size:,} bytes。",
                      "```text"])
        for path in files:
            relative = path.relative_to(root).as_posix()
            lines.append(f"FILE\t{code}\t{path.stat().st_size}\t{relative}")
        lines.append("```")
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
