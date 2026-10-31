#!/usr/bin/env python3
"""Read retained metadata and stat current assets; never load weights or run models."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
CANDIDATE = Path("results/query_relay_2026_10/probes/A_itemgate6_rms_roll4")
PANELS = {
    "development": "results/design_system_2026_10/phase2/representative4096/panel",
    "calibration": "results/design_two_2026_10/benchmark_panel",
    "revealed_layer_confirmation": "results/query_relay_2026_10/confirmation/panel",
    "revealed_item_confirmation": "results/query_relay_2026_10/confirmation_item/panel",
}
COMPLETED = {
    "item_development4096": CANDIDATE / "population4096/merged_actual/summary.json",
    "item_feedback4096": CANDIDATE / "feedback4096_actual/merged/summary.json",
    "item_confirmation1024": "results/query_relay_2026_10/confirmation_item/behavior1024/summary.json",
    "layer_confirmation1024": "results/query_relay_2026_10/confirmation/behavior1024/summary.json",
    "static_writer_development4096": "results/query_relay_2026_10/comparisons/item_vs_constant4096/summary.json",
    "static_writer_confirmation1024": "results/query_relay_2026_10/comparisons/item_vs_constant_confirmation1024/summary.json",
    "item_code_content128": CANDIDATE / "code_use128_v3/summary.json",
    "optimized_engineering_recheck1024": "results/implementation_2026_10/population_confirmation1024/merged/summary.json",
    "engineered_cost": "results/implementation_2026_10/cost/summary.json",
    "recflow_foundation_chain": "results/recflow/development/window_6l_expanded_u4096_seed17/chain_summary.json",
}


def collect():
    files, missing = {}, []

    def asset(path, *, required=False):
        p = Path(path)
        p = p if p.is_absolute() else ROOT / p
        key = str(p)
        present = p.is_file()
        files[key] = dict(path=key, exists=present,
                          bytes=p.stat().st_size if present else None,
                          required=required or files.get(key, {}).get("required", False))
        if required and not present and key not in missing:
            missing.append(key)
        return files[key]

    def read(path, *, required=False):
        record = asset(path, required=required)
        return json.loads(Path(record["path"]).read_text()) if record["exists"] else {}

    panels, roles = {}, {}
    for name, folder in PANELS.items():
        base = Path(folder)
        binding = read(base / "binding.json", required=True)
        groups = read(base / "users.json", required=True)
        roles[name] = {key: set(value) for key, value in groups.items()}
        panels[name] = dict(
            binding=str(ROOT / base / "binding.json"),
            roles=str(ROOT / base / "users.json"),
            counts={key: len(value) for key, value in groups.items()},
            model_chain=binding.get("model_chain"), model_config=binding.get("model_config"),
            state_rule=binding.get("state_rule"), request_rule=binding.get("request_rule"),
            inputs=[asset(base / name_, required=True) for name_ in ("histories.npz", "requests.parquet")],
        )
        if "confirmation" in name:
            panels[name]["revealed"] = True
            panels[name]["reuse_scope"] = "Existing outcomes are known; not a new blind confirmation after tuning."
            panels[name]["prior_role_inventory"] = asset(base / "excluded_roles.json")
            verification = read(base / "verification.json", required=True)
            panels[name]["recorded_role_checks"] = {
                k: v for k, v in verification.items() if "overlap" in k or "role" in k
            }
        for model in binding.get("models", {}).values():
            if model.get("checkpoint"):
                asset(model["checkpoint"], required=True)

    freeze = read(CANDIDATE / "freeze.json", required=True)
    current = dict(name=CANDIDATE.name, freeze=str(ROOT / CANDIDATE / "freeze.json"),
                   frozen_settings=freeze.get("settings"), releases={})
    for version, record in freeze.get("release_bindings", {}).items():
        config_path = CANDIDATE / "calibration" / version / "configuration.json"
        cfg = read(config_path, required=True)
        metadata = read(config_path.parent / "calibration.json", required=True)
        fields = ("mode", "fit_users", "validation_users", "local_epochs", "joint_epochs",
                  "batch_size", "learning_rate", "seed", "read_shared", "read_private",
                  "gate_source", "gate_normalization", "feature_mode", "rollout_steps",
                  "release_initialization", "source_mode", "objectives")
        current["releases"][version] = dict(
            configuration=str(ROOT / config_path),
            settings={k: cfg.get(k) for k in fields},
            checkpoint=asset(record["calibration_path"], required=True),
            recorded_weights_sha256=metadata.get("weights_sha256"),
            fit_uids=cfg.get("fit_uids", []), validation_uids=cfg.get("validation_uids", []),
        )
        for key in ("origin", "feature_bank_path"):
            if cfg.get(key):
                asset(cfg[key], required=True)
        if record.get("source_pack"):
            current["releases"][version]["prior_source_pack"] = asset(record["source_pack"])
    current["publication_tables"] = {
        version: asset(record["path"], required=True)
        for version, record in freeze.get("publication_bindings", {}).items()
    }
    if freeze.get("fixed_feature_bank", {}).get("path"):
        current["fixed_feature_bank"] = asset(freeze["fixed_feature_bank"]["path"], required=True)

    # Resolve the selected endpoints from the retained model list, not old raw paths.
    models, model_list = {}, ROOT / "docs/unified_training_2026_09/model_versions.md"
    if asset(model_list, required=True)["exists"]:
        scale = None
        for line in model_list.read_text().splitlines():
            if line.startswith("## "):
                scale = line[3:].lower().strip()
            match = re.search(r"\| (V[0-5]) \|.*?\[([0-9a-f]+)\]\(([^)]+checkpoint_100\.pt)\)", line)
            if match:
                version, digest, link = match.groups()
                path = (model_list.parent / link).resolve()
                models.setdefault(scale, {})[version.lower()] = dict(
                    checkpoint=asset(path, required=scale == "medium"),
                    recorded_sha256_prefix=digest, metadata_source=str(model_list),
                    selected_row=line,
                )
    for scale, manifest in {
        "medium": "results/unified_training_2026_09/medium/seed17/checkpoints/chain.manifest.json",
        "large": "results/unified_training_2026_09/large/seed17/checkpoints/chain.v0_v5.manifest.json",
    }.items():
        data = read(manifest)
        for row in data.get("versions", []):
            target = models.get(scale, {}).get(row["version"])
            if target is not None:
                target["model_config"] = row.get("config")
                target["manifest"] = str(ROOT / manifest)
                target["recorded_sha256"] = row.get("checkpoint_sha256")

    recflow_path = COMPLETED["recflow_foundation_chain"]
    recflow = read(recflow_path)
    models["recflow"] = {
        phase: dict(checkpoint=asset(record["checkpoint"]), model_config=record.get("model"),
                    recorded_sha256=record.get("recorded_checkpoint_sha256"),
                    metadata_source=str(ROOT / recflow_path))
        for phase, record in recflow.get("phases", {}).items()
    }
    completed = {
        name: dict(evidence=asset(path), disposition="Retained completed evidence; not a rerun queue.")
        for name, path in COMPLETED.items()
    }
    fit = set(freeze.get("fit_uids", []))
    val = set(freeze.get("validation_uids", []))
    overlap = {
        name: dict(current_fit_overlap=len(group.get("evaluation", set()) & fit),
                   current_validation_overlap=len(group.get("evaluation", set()) & val))
        for name, group in roles.items()
    }
    adaptation = {
        "medium": "Current reference and H6/D32 optimized paths exist; protocol/shape changes need matched controls.",
        "large_max": "Foundation endpoints exist. Prepare scale-specific panels and origin calibration; current H6 optimized reader is not a ready-made backend for H10 models.",
        "recflow": "Foundation A-F endpoints exist; no relay cache result yet. SiLU/reference block/relative bias and three-token joint scoring require adapter and teacher integration.",
        "service_replay": "Current integration is synchronous per-user numerical replay; arrival/queue/resident-capacity service measurements remain to implement.",
        "budgeted_rebuild": "Native rebuild primitives exist; persistent budgeted selection under the same event stream remains to implement.",
    }
    return dict(
        status="ready_for_preparation" if not missing else "missing_required_assets",
        created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        scope="Read-only preparation inventory, not training, quality evaluation, or checkpoint integrity verification.",
        method="Read retained JSON/Markdown and file stat only; no tensor loading, large-file hashing, GPU access, or historical raw restoration.",
        source_root=str(ROOT), current_candidate=current, panels=panels,
        current_calibration_vs_panel_evaluation_overlap=overlap, foundation_models=models,
        completed_evidence=completed, pending_adaptation=adaptation,
        files=list(files.values()), missing_required=missing,
        historical_note="AGENTS rank8 settings precede this candidate. Existing reports remain unchanged. Missing retired raw data are not reconstruction tasks.",
    )


def markdown(summary):
    settings = summary["current_candidate"]["frozen_settings"] or {}
    lines = ["# 本轮准备资产清单", "", summary["scope"], "",
             f"状态：`{summary['status']}`。只读元数据与文件大小，未加载模型、未使用 GPU；已记录哈希未重新验证。", "",
             f"当前候选：`{summary['current_candidate']['name']}`；fit/val={settings.get('fit_users')}/{settings.get('validation_users')}，"
             f"local/joint={settings.get('local_epochs')}/{settings.get('joint_epochs')}，"
             f"读取共享/私有={settings.get('read_shared')}/{settings.get('read_private')}，"
             f"续写={settings.get('rollout_steps')}，门控={settings.get('gate_source')}。", "",
             "| 面板 | evaluation | 保留 fit / validation | 角色文件 |", "|---|---:|---:|---|"]
    for name, panel in summary["panels"].items():
        counts = panel["counts"]
        lines.append(f"| {name} | {counts.get('evaluation', 0)} | {counts.get('fit', 0)} / {counts.get('validation', 0)} | `{panel['roles']}` |")
    lines += ["", "两个确认组均已揭晓；不能用于新候选调参后再次宣称独立确认。面板保留角色数不等于当前实际训练人数。", "",
              "| 基础模型 | 存在 / 已登记 | 权重字节（仅 stat） |", "|---|---:|---:|"]
    for name, models in summary["foundation_models"].items():
        count = sum(row["checkpoint"]["exists"] for row in models.values())
        size = sum(row["checkpoint"]["bytes"] or 0 for row in models.values())
        lines.append(f"| {name} | {count} / {len(models)} | {size:,} |")
    lines += ["", "已完成证据不列入重跑队列："]
    lines += [f"- {name}: `{row['evidence']['path']}`（存在：{row['evidence']['exists']}）"
              for name, row in summary["completed_evidence"].items()]
    lines += ["", "待适配："]
    lines += [f"- {name}: {note}" for name, note in summary["pending_adaptation"].items()]
    lines += ["", f"必需资产缺失：{len(summary['missing_required'])}。历史清理后的 raw 路径不是恢复任务。"]
    lines += [f"- `{path}`" for path in summary["missing_required"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing paths are never overwritten.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = collect()
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (args.output / "summary.md").write_text(markdown(summary))
    print(json.dumps(dict(status=summary["status"], output=str(args.output.resolve()),
                         files=len(summary["files"]), missing_required=summary["missing_required"])))
    return 1 if summary["missing_required"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
