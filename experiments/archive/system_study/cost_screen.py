#!/usr/bin/env python3
"""CPU shape forecast for two bounded relay changes; no fitting or GPU work."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
import numpy as np
import torch

from hstu_kvcache.methods.cachetide.adapter import UnifiedRelayAdapter
from hstu_kvcache.methods.cachetide.calibration_cost import OperationCounts, update_forward_flops
from hstu_kvcache.methods.cachetide.cost import price, read_flops
from hstu_kvcache.evaluation.cachetide_workload import trigger_indices
from hstu_kvcache.evaluation.cost import CostModel

BASE = ROOT / "results/query_relay_2026_10/probes/A_itemgate6_rms_roll4"
PANEL = ROOT / "results/design_two_2026_10/benchmark_panel"
LEDGERS = {
    "stride16_candidates16": ROOT / "results/query_relay_2026_10/workloads/stride16_candidates16_ledger.json",
    "original_feedback": ROOT / "results/query_relay_2026_10/cost/event_ledger.json",
}
REFERENCE = ROOT / "results/implementation_2026_10/cost/summary.json"
RELEASES = (("v3", 259 * 86400), ("v4", 273 * 86400), ("v5", 287 * 86400))
BRANCHES = {
    "current": dict(read_shared=3, read_private=3, fit_users=64, local_epochs=8, joint_epochs=4),
    "Q1_read6plus6": dict(read_shared=6, read_private=6, fit_users=64, local_epochs=8, joint_epochs=4),
    "Q2_calibration128": dict(read_shared=3, read_private=3, fit_users=128, local_epochs=16, joint_epochs=8),
}


def read(path):
    return json.loads(Path(path).read_text())


def histories_for(uids):
    """Read retained pre-existing events, without labels or foundation weights."""
    with np.load(PANEL / "histories.npz") as pack:
        positions = {int(u): i for i, u in enumerate(pack["uids"])}
        offsets, times, items = pack["offsets"], pack["timestamps"], pack["item_ids"]
        return {uid: (times[offsets[positions[uid]]:offsets[positions[uid] + 1]].copy(),
                      items[offsets[positions[uid]]:offsets[positions[uid] + 1]].copy()) for uid in uids}


def source_shapes(adapter, events_by_uid, uids, version, known):
    """Count the explicit capture path, including lazy preparation at read boundaries."""
    target = dict(RELEASES)[version]
    prior = tuple(row for row in RELEASES if row[1] < target)
    cm = {b: CostModel(192, 6, 6, b) for b in ("torch", "triton")}
    totals = Counter()
    for uid in uids:
        times, items = events_by_uid[uid]
        cursor = int(np.searchsorted(times, RELEASES[0][1], side="left"))
        length, coded, pending = min(cursor, 1024), 0, 0
        totals["native_initial"] += cm["triton" if length == 1024 else "torch"].full_cache(length)

        def prepare():
            nonlocal pending, coded
            if pending:
                totals["code_update"] += pending * update_forward_flops(adapter, rows=length)
                coded, pending = length, 0

        def advance(until):
            nonlocal cursor, length, coded
            stop = int(np.searchsorted(times, until, side="left"))
            while cursor < stop:
                width = 1 << (min(32, stop - cursor).bit_length() - 1)
                prepare()
                totals["native_append"] += cm["triton" if length == 1024 else "torch"].band_append(length, width)
                if coded:
                    totals["corrected_read"] += read_flops(pairs=width * coded, queries=width,
                        head_rank=adapter.accessible_rank, gate_normalization="rms")["total"]
                removed = max(0, length + width - 1024)
                coded = max(0, coded - removed)
                length, cursor = min(1024, length + width), cursor + width

        boundaries = trigger_indices((times, items), stride=16, known_vocab_size=known,
                                     releases=prior, end=target)
        for i, (name, at) in enumerate(prior):
            advance(at)
            pending += 1
            until = prior[i + 1][1] if i + 1 < len(prior) else target
            for timestamp in sorted(set(map(int, times[boundaries[name]]))):
                advance(timestamp)
                prepare()
            advance(until)
        advance(target)
        prepare()  # source_snapshot calls prepare_read even for an idle user.
    return dict(totals)


def calibration_shapes(adapter, settings, lengths, train, validation):
    """Replay calibrate.py counters and seeded padded batch order, never tensor values."""
    counts = OperationCounts()
    cm = {b: CostModel(192, 6, 6, b) for b in ("torch", "triton")}
    all_uids, batch, q = train + validation, 8, 16
    for uid in all_uids:
        n = lengths[uid]
        if n < 1:
            raise ValueError("Calibration requires strictly pre-release history")
        counts.add("teacher_cache_capture_flops", cm["triton" if n == 1024 else "torch"].full_cache(n))
        counts.record_query(adapter, length=n, queries=q, fitted_layers=0, stage="teacher_score_prepare")
    generator = torch.Generator().manual_seed(17)
    for index, layer in enumerate(adapter.layers):
        for uid in all_uids:
            counts.record_local_capture(adapter, length=lengths[uid], queries=q, layer_index=index)
        counts.add("local_residual_scale_flops", 3 * len(train) * 6 * q * 32 + 2 * 6)
        parameters = [p for p in layer.parameters() if p.requires_grad]
        size = sum(p.numel() for p in parameters)

        def local_rows(chosen, backward):
            for start in range(0, len(chosen), batch):
                group = chosen[start:start + batch]
                counts.record_local(adapter, batch=len(group), padded_length=max(lengths[u] for u in group),
                    queries=q, backward=backward, encoder_backward=backward,
                    stage="local_train" if backward else "local_validation")
                if backward:
                    counts.record_optimizer(parameters=size, stage="local")

        local_rows(validation, False)
        for _ in range(settings["local_epochs"]):
            order = torch.randperm(len(train), generator=generator).tolist()
            local_rows([train[i] for i in order], True)
            local_rows(validation, False)
        counts.record_gradient_diagnostic(parameters=size, tensors=len(parameters))

    def joint(uid, backward):
        n = lengths[uid]
        counts.record_update(adapter, length=n, backward=backward, stage="joint_static_encoder")
        counts.record_query(adapter, length=n, queries=q, backward=backward, stage="joint_static_query")
        counts.record_score_loss(outputs=q, backward=backward, stage="joint_static")
        tail = min(4, n - 1)
        if tail:
            prefix = n - tail
            counts.record_update(adapter, length=prefix, backward=backward, stage="joint_rollout_encoder")
            for i in range(tail):
                counts.record_event_write(adapter, old_length=prefix + i, width=1,
                    coded_rows=prefix, backward=backward, stage="joint_rollout_write")
            counts.record_query(adapter, length=n, queries=q, coded_rows=prefix,
                                backward=backward, stage="joint_rollout_query")
            counts.record_score_loss(outputs=q, backward=backward, stage="joint_rollout")
            counts.add("joint_two_view_combination_flops", 3)

    parameters = [p for p in adapter.parameters() if p.requires_grad]
    size = sum(p.numel() for p in parameters)
    for uid in validation:
        joint(uid, False)
    for _ in range(settings["joint_epochs"]):
        for uid in train:
            joint(uid, True)
        for _ in range((len(train) + batch - 1) // batch):
            counts.record_optimizer(parameters=size, stage="joint")
        for uid in validation:
            joint(uid, False)
    counts.record_gradient_diagnostic(parameters=size, tensors=len(parameters))
    for uid in train:
        counts.record_gate_statistics(adapter, length=lengths[uid], queries=q)
    counts.add("gate_statistics_final_moment_flops", 6 * 6 * 2 * 6 * adapter.accessible_rank)
    return dict(counts.components)


def packed_price(ledger, publications, calibration, source, rank):
    # Projection/composition geometry is independent of selected read coordinates.
    pub = deepcopy(publications)
    for metadata in pub.values():
        metadata["adapter_config"].update(read_shared=rank // 2, read_private=rank // 2)
    result = price(ledger, branch="A", xi_width=1, calibration_flops=calibration,
        source_reconstruction_flops=source, replicas=4, prefix_only=True, block_sparse=True,
        head_rank=rank, gate_normalization="rms", preparation_mode="independent",
        feature_mode="origin_encoder", publication_metadata=pub, gate_source="item", stable_anchor_inverse=True)
    removed = Counter()
    for version, data in result["by_release"].items():
        for role in ("ranking", "writer"):
            queries = ledger["by_release"][version][role + "_nonzero_prefix_queries"]
            saved = 5 * 6 * (4 * 32 + 2) * queries
            part = data[role + "_read_flops"]
            part["gate_query_normalization_flops"] -= saved
            part["total"] -= saved
            removed[role] += saved
    result["extra_ranking_flops"] -= removed["ranking"]
    result["extra_writer_flops"] -= removed["writer"]
    saved = sum(removed.values())
    result["complete_added_flops"] -= saved
    total, denominator = result["complete_added_flops"], result["once_rebuild_flops"]
    result["complete_percent_of_once_rebuild"] = 100 * total / denominator
    result["complete_with_source_reconstruction_percent"] = 100 * (total + source) / denominator
    for label, native in (("whole_query", "common_native_whole_query_flops"),
                          ("history_attention", "native_history_attention_flops"),
                          ("history_and_self_attention", "native_history_and_self_attention_flops")):
        result["extra_percent_of_native_" + label] = 100 * result["extra_ranking_flops"] / result[native]
    result["extra_percent_of_native_append"] = 100 * result["extra_writer_flops"] / result["common_native_incremental_append_flops"]
    result["total_modeled_service_relay_flops"] -= saved
    result["total_modeled_service_relay_with_source_reconstruction_flops"] -= saved
    result["net_saved_flops"] += saved
    result["net_saved_with_source_reconstruction_flops"] += saved
    full = result["total_modeled_service_full_flops"]
    result["net_saved_percent_of_total_full_service"] = 100 * result["net_saved_flops"] / full
    result["net_saved_with_source_percent_of_total_full_service"] = 100 * result["net_saved_with_source_reconstruction_flops"] / full
    result["net_kv_rebuild_replacement_saving_percent"] = 100 * result["net_saved_flops"] / denominator
    result["net_kv_rebuild_replacement_with_source_saving_percent"] = 100 * result["net_saved_with_source_reconstruction_flops"] / denominator
    result["remaining_calibration_budget_flops"] += saved
    result["saved_cross_layer_RMS_flops"] = saved
    result["preparation_flops"] = calibration + result["maintenance_flops"]
    result["service_correction_flops"] = result["extra_ranking_flops"] + result["extra_writer_flops"]
    result["publication_cost_scope"] = "Retained full-catalogue arithmetic reused by identical A/48-dimensional update geometry; new weights/tables not executed."
    return result


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    roles, binding = read(PANEL / "users.json"), read(PANEL / "binding.json")
    histories = histories_for(roles["fit"][:128] + roles["validation"][:16])
    publications = {v: read(BASE / "publication" / f"{v}.json") for v, _ in RELEASES}
    metadata = {v: read(BASE / "calibration" / v / "calibration.json") for v, _ in RELEASES}
    ledgers, reference = {name: read(p) for name, p in LEDGERS.items()}, read(REFERENCE)
    origin = metadata["v3"]["cost"]["warmstart_origin"]
    branches = {}
    for name, settings in BRANCHES.items():
        config = dict(publications["v3"]["adapter_config"], read_shared=settings["read_shared"],
                      read_private=settings["read_private"])
        adapter = UnifiedRelayAdapter(**config)  # Small random parameter geometry only; no bank/weights loaded.
        train, validation = roles["fit"][:settings["fit_users"]], roles["validation"][:16]
        stages = {}
        for version, cutover in RELEASES:
            lengths = {uid: min(1024, int(np.searchsorted(histories[uid][0], cutover, side="left")))
                       for uid in train + validation}
            components = calibration_shapes(adapter, settings, lengths, train, validation)
            source_components = source_shapes(adapter, histories, train + validation, version, binding["known_vocab_size"])
            if name == "current":
                expected = metadata[version]["cost"]["components"]
                differences = {k: (components.get(k, 0), expected.get(k, 0))
                               for k in components.keys() | expected.keys() if components.get(k, 0) != expected.get(k, 0)}
                if differences:
                    raise AssertionError((version, "calibration component mismatch", differences))
                expected_source = metadata[version]["cost"]["source_reconstruction_sensitivity"]["additional_flops"]
                if version == "v3":
                    expected_source -= origin["source_reconstruction_sensitivity_flops"]
                assert sum(source_components.values()) == expected_source, (version, source_components, expected_source)
            stages[version] = dict(components_flops=components, total_flops=sum(components.values()),
                source_reconstruction_components_flops=source_components,
                source_reconstruction_flops=sum(source_components.values()),
                fit_history_rows=sum(lengths[u] for u in train), validation_history_rows=sum(lengths[u] for u in validation),
                shortest_history=min(lengths.values()), longest_history=max(lengths.values()))
        calibration = origin["calibration_flops"] + sum(s["total_flops"] for s in stages.values())
        source = origin["source_reconstruction_sensitivity_flops"] + sum(s["source_reconstruction_flops"] for s in stages.values())
        protocols = {protocol: packed_price(ledger, publications, calibration, source, adapter.accessible_rank)
                     for protocol, ledger in ledgers.items()}
        if name == "current":
            for protocol, report in protocols.items():
                for field in ("calibration_flops", "maintenance_flops", "extra_ranking_flops", "extra_writer_flops",
                              "complete_added_flops", "source_reconstruction_sensitivity_flops", "once_rebuild_flops",
                              "native_history_and_self_attention_flops", "common_native_whole_query_flops"):
                    assert report[field] == reference["protocols"][protocol][field], (protocol, field)
        memory = deepcopy(reference["persistent_memory"])
        derived = 110592 * adapter.accessible_rank // 6
        memory.update(added_derived_bytes_per_replica=derived,
                      derived_gate_weights_bytes_per_replica=derived // 2,
                      derived_read_decoders_bytes_per_replica=derived // 2,
                      shared_state_bytes_per_replica=memory["original_shared_state_bytes_per_replica"] + derived,
                      publication_peak_shared_state_bytes_per_producer=memory["original_publication_peak_shared_state_bytes_per_producer"] + 2 * derived)
        memory["shared_state_bytes_four_replicas"] = 4 * memory["shared_state_bytes_per_replica"]
        memory["added_derived_bytes_four_replicas"] = 4 * derived
        memory["derived_payload_bytes_per_replica_release"] = derived
        memory["derived_payload_bytes_four_replicas_three_releases"] = 12 * derived
        branches[name] = dict(settings=dict(settings, validation_users=16, rollout_steps=4, seed=17, batch_size=8),
            evidence="Existing arithmetic reproduced" if name == "current" else "Untrained shape prediction; no quality or timing result",
            initial_origin=origin, calibration_stages=stages, calibration_flops=calibration,
            source_reconstruction_sensitivity_flops=source, protocols=protocols, memory=memory)
        print(json.dumps({"branch": name, "calibration_TF": calibration / 1e12,
            "protocols": {p: r["complete_percent_of_once_rebuild"] for p, r in protocols.items()}}), flush=True)
    sources = [Path(__file__), REFERENCE, PANEL / "binding.json", PANEL / "users.json",
               *LEDGERS.values(), *(BASE / "calibration" / v / "calibration.json" for v, _ in RELEASES),
               *(BASE / "publication" / f"{v}.json" for v, _ in RELEASES)]
    summary = dict(status="complete_cpu_cost_screen", created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        branches=branches, checks=dict(current_calibration_components_all_three_releases_match=True,
            current_source_reconstruction_all_three_releases_match=True, current_complete_protocol_components_match=True),
        assumptions=["MAC=2; padded shapes charged; backward conservatively two forward passes, as in retained accounting.",
            "Calibration uses fixed fit/validation roles, seed17 batches, 16 candidates, 4 causal tail steps; early-selected checkpoints do not shorten scheduled training.",
            "Prospective source capture uses own-state explicit updates and behavior stride16 boundaries, just as the frozen candidate; geometry is value independent.",
            "Q1 assumes the same operation-wide RMS reuse is available at width12; present optimized reader supports only width6, so kernel adaptation and timing remain untested.",
            "Four independently prepared replica tables, immutable feature bank, origin calibration once, teacher work and source reconstruction sensitivity are included.",
            "No quality target is used for acceptance;80% is historical context. Ratios above20% are tradeoff points, not automatic rejection.",
            "Logical FLOPs and persistent bytes are forecasts, not measured instructions, DRAM traffic, latency, peak training memory or experiment outcomes."],
        sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],
        history_source=dict(path=str(PANEL / "histories.npz"), bytes=(PANEL / "histories.npz").stat().st_size,
                            recorded_binding=binding["files"]["history"], rehashed=False))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    lines = ["# 第一批成本预筛", "", "CPU按真实事件和历史长度计价；Q1/Q2尚未训练，不含新质量或实测耗时结论。", "",
             "| 分支 | 协议 | 校准TF（含起点一次） | 准备TF | 服务纠正TF | 新增/重建% | 含来源重建% | query/原query% | query/attention% |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, branch in branches.items():
        for protocol, r in branch["protocols"].items():
            lines.append(f"| {name} | {protocol} | {r['calibration_flops']/1e12:.3f} | {r['preparation_flops']/1e12:.3f} | {r['service_correction_flops']/1e12:.3f} | {r['complete_percent_of_once_rebuild']:.5f} | {r['complete_with_source_reconstruction_percent']:.5f} | {r['extra_percent_of_native_whole_query']:.3f} | {r['extra_percent_of_native_history_and_self_attention']:.3f} |")
    lines += ["", "当前三版校准每个组件、来源重建和两个30k协议总账均与保留结果逐项一致。Q1改变读取宽度；Q2改变fit人数与轮次；没有把全部系统成本按倍数放大。准备包含校准、表准备及码转换；原生公共读写与各自重建分母在JSON单列。", "",
              "固定48维码与全目录表容量保持不变。Q1若扩展打包后端，仅派生门控/投影副本随读取宽度增加；这不是已完成的内核适配。完整行状态、四副本共享表与交接峰值见JSON。", "",
              "代价判断：先考虑Q2的小规模拟合，它增加一次性校准预算；Q1把附加工作放在每次读写，需更强质量收益才可能经济。20%与80%都不能代替质量—成本曲线。", ""]
    (output / "summary.md").write_text("\n".join(lines))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New isolated directory")
    run(parser.parse_args().output)
