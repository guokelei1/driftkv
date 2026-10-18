#!/usr/bin/env python3
"""Conservative complete FLOPs ledger for the frozen implementation backend.

Reuse the fixed 30k causal ledgers and completed calibration/publication costs.
Only the operation-wide reuse of item-query RMS removes logical arithmetic.
Table projection batching, attention fusion and in-place code conversion do
not receive invented FLOPs savings.  This script is CPU/JSON accounting only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from query_relay_2026_10.cost import read_flops, row_update_flops

BASE = ROOT / "results/query_relay_2026_10"
NAME = "A_itemgate6_rms_roll4"
BUDGET = BASE / "cost/item_and_static_actual_budget.json"
BRANCH_COST = BASE / "probes" / NAME / "cost.json"
FEEDBACK_QUALITY = BASE / "probes" / NAME / "feedback4096_actual/merged/summary.json"
LEDGERS = {
    "stride16_candidates16": BASE / "workloads/stride16_candidates16_ledger.json",
    "original_feedback": BASE / "cost/event_ledger.json",
}


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    return dict(path=str(Path(path).relative_to(ROOT)),
                sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def derive(ledger, old, compact, final_price):
    cfg = ledger["model_config"]
    layers, heads, width = cfg["num_layers"], cfg["num_heads"], cfg["hidden_size"]
    dim = width // heads
    assert (layers, heads, dim, old["read_head_rank"]) == (6, 6, 32, 6)
    assert old["prefix_only"] and old["gate_normalization"] == "rms"
    assert old["preparation_replicas"] == 4 and old["preparation_mode"] == "independent"
    assert old["once_rebuild_flops"] == ledger["once_rebuild_flops"]
    assert ledger["users"] == 30000
    assert row_update_flops("A", 1, layers=layers, block_sparse=True,
                            feature_mode="origin_encoder") == 93888
    for key in ("calibration_flops", "maintenance_flops", "complete_added_flops",
                "source_reconstruction_sensitivity_flops", "once_rebuild_flops"):
        assert old[key] == compact[key] == final_price[key], key

    per_release, savings = {}, dict(ranking=0, writer=0)
    old_normalization, new_normalization = 0, 0
    for version, counts in ledger["by_release"].items():
        stage = old["by_release"][version]
        result = dict(roles={}, maintenance_components_flops=deepcopy(stage["maintenance_components_flops"]))
        rows = counts.get("anchor_to_current_rows", 0) + counts.get("current_to_anchor_rows", 0)
        assert result["maintenance_components_flops"]["anchor_current_conversions"] == rows * 93888
        for role in ("ranking", "writer"):
            queries = counts[role + "_nonzero_prefix_queries"]
            reference = read_flops(pairs=counts[role + "_prefix_pairs"], queries=queries,
                head_rank=6, layers=layers, heads=heads, head_dim=dim,
                gates=True, gate_normalization="rms")
            assert reference == stage[role + "_read_flops"], (version, role)
            old_rms = layers * heads * (4 * dim + 2) * queries
            new_rms = heads * (4 * dim + 2) * queries
            saved = (layers - 1) * heads * (4 * dim + 2) * queries
            assert old_rms == reference["gate_query_normalization_flops"]
            assert saved == old_rms - new_rms and 0 <= saved <= old_rms
            optimized = {**reference, "gate_query_normalization_flops": new_rms,
                         "total": reference["total"] - saved}
            assert optimized["total"] == sum(value for key, value in optimized.items() if key != "total")
            for key in ("code_pair_flops", "code_fixed_query_flops", "nonlinear_gate_flops"):
                assert optimized[key] == reference[key]
            result["roles"][role] = dict(nonzero_prefix_queries=queries,
                original=reference, optimized=optimized, saved_rms_flops=saved)
            savings[role] += saved
            old_normalization += old_rms
            new_normalization += new_rms
        per_release[version] = result
    saved_total = sum(savings.values())
    assert saved_total == old_normalization - new_normalization
    ranking = old["extra_ranking_flops"] - savings["ranking"]
    writer = old["extra_writer_flops"] - savings["writer"]
    calibration, maintenance = old["calibration_flops"], old["maintenance_flops"]
    complete = calibration + maintenance + ranking + writer
    assert old["complete_added_flops"] - complete == saved_total
    source = old["source_reconstruction_sensitivity_flops"]
    denominator = old["once_rebuild_flops"]
    native_q = old["common_native_whole_query_flops"]
    native_w = old["common_native_incremental_append_flops"]
    full = native_q + native_w + denominator
    relay = native_q + native_w + complete
    assert full == old["total_modeled_service_full_flops"]
    assert old["total_modeled_service_relay_flops"] - relay == saved_total
    result = dict(users=ledger["users"], candidate_scores=ledger["requests"],
        recommendation_rounds=ledger.get("recommendation_rounds"),
        by_release=per_release, saved_rms_flops_by_role=savings,
        saved_rms_flops=saved_total, original_rms_flops=old_normalization,
        optimized_rms_flops=new_normalization,
        original_complete_added_flops=old["complete_added_flops"],
        original_complete_percent_of_once_rebuild=old["complete_percent_of_once_rebuild"],
        calibration_flops=calibration, maintenance_flops=maintenance,
        extra_ranking_flops=ranking, extra_writer_flops=writer,
        complete_added_flops=complete, once_rebuild_flops=denominator,
        complete_percent_of_once_rebuild=100 * complete / denominator,
        source_reconstruction_sensitivity_flops=source,
        complete_with_source_reconstruction_flops=complete + source,
        complete_with_source_reconstruction_percent=100 * (complete + source) / denominator,
        twenty_percent_cost_ceiling_met=complete <= denominator / 5,
        twenty_percent_with_source_cost_ceiling_met=(complete + source) <= denominator / 5,
        all_resident_eager_rebuild_flops=ledger["all_resident_eager_rebuild_flops"],
        common_native_whole_query_flops=native_q, common_native_incremental_append_flops=native_w,
        extra_percent_of_native_whole_query=100 * ranking / native_q,
        native_history_attention_flops=old["native_history_attention_flops"],
        native_history_and_self_attention_flops=old["native_history_and_self_attention_flops"],
        extra_percent_of_native_history_attention=100 * ranking / old["native_history_attention_flops"],
        extra_percent_of_native_history_and_self_attention=100 * ranking / old["native_history_and_self_attention_flops"],
        extra_percent_of_native_append=100 * writer / native_w,
        total_modeled_service_full_flops=full, total_modeled_service_relay_flops=relay,
        total_modeled_service_relay_with_source_reconstruction_flops=relay + source,
        net_saved_flops=full - relay, net_saved_with_source_reconstruction_flops=full - relay - source,
        net_saved_percent_of_total_full_service=100 * (full - relay) / full,
        net_saved_with_source_percent_of_total_full_service=100 * (full - relay - source) / full,
        extra_gate_item_lookup_bytes=old["extra_gate_item_lookup_bytes"],
        shared_state_publication_transfer_bytes=old["shared_state_publication_transfer_bytes"],
        checks=dict(original_stage_cost_matches_literal_ledger=True,
            all_normalization_components_equal_L_H_4dplus2_times_queries=True,
            savings_no_larger_than_original_normalization=True,
            all_non_RMS_read_work_unchanged=True, full_source_reconstruction_still_charged=True,
            original_93888_conversion_flops_per_row_retained=True,
            sum_cost_before_division_identity=True, denominator_and_native_service_unchanged=True))
    assert result["complete_with_source_reconstruction_flops"] - complete == source
    return result


def run(output):
    if output.exists():
        raise FileExistsError(output)
    audit, candidate_cost = read(BUDGET), read(BRANCH_COST)
    candidate = audit["candidates"][NAME]
    assert candidate_cost["audit_binding"]["sha256"] == bind(BUDGET)["sha256"]
    assert candidate["missing_calibration_versions"] == []
    assert candidate["publication_status"] == "all_full_catalogue_artifacts_observed"
    calibration = candidate["initial_origin"]["calibration_flops"] + sum(
        s["stage_calibration_flops"] for s in candidate["stages"].values())
    source = sum(s["omitted_source_reconstruction_flops"] for s in candidate["stages"].values())
    assert calibration == candidate_cost["calibration_flops"]
    assert source == candidate_cost["source_reconstruction_sensitivity_flops"]
    protocols = {}
    for protocol, path in LEDGERS.items():
        assert audit["protocols"][protocol]["binding"]["sha256"] == bind(path)["sha256"]
        protocols[protocol] = derive(read(path), candidate["detailed_primary_and_original"][protocol],
            candidate["protocol_prices"][protocol], candidate_cost["stable_anchor_inverse_prices"][protocol])
    # Exact parameter geometry of the frozen wrapper, not an uncharged cache.
    gate_bytes = 6 * 32 * 6 * (2 * 6) * 4
    decoder_bytes = 6 * 2 * 6 * 6 * 32 * 4
    assert gate_bytes == decoder_bytes == 55296
    derived = gate_bytes + decoder_bytes
    memory = deepcopy(candidate["persistent_memory"])
    original_global = memory["shared_state_bytes_per_replica"]
    original_peak = memory["publication_peak_shared_state_bytes_per_producer"]
    memory.update(original_shared_state_bytes_per_replica=original_global,
        derived_gate_weights_bytes_per_replica=gate_bytes,
        derived_read_decoders_bytes_per_replica=decoder_bytes,
        added_derived_bytes_per_replica=derived,
        added_derived_bytes_four_replicas=4 * derived,
        derived_payload_bytes_per_replica_release=derived,
        derived_payload_bytes_four_replicas_three_releases=4 * 3 * derived,
        derived_pack_floating_point_flops=0,
        derived_copy_scope="Logical source and destination payloads, not measured DRAM traffic; temporary cat/stack/permute copies and allocator work are not latency-free. No floating-point projection is added when selecting/copying weights.",
        shared_state_bytes_per_replica=original_global + derived,
        shared_state_bytes_four_replicas=4 * (original_global + derived),
        original_publication_peak_shared_state_bytes_per_producer=original_peak,
        publication_peak_shared_state_bytes_per_producer=original_peak + 2 * derived,
        per_row_extra_persistent_bytes=0,
        peak_scope="Original old/new D/P/Q/inverse and fixed bank accounting, plus both old/new derived gate and decoder buffers. Excludes runtime scratch and framework/model allocations. Integration retains all release tables for controls and is not steady loaded-service memory evidence.")
    feedback = read(FEEDBACK_QUALITY)["quality"]
    feedback_boundary = dict(users=feedback["users"], candidate_scores=feedback["requests"],
        recovery=feedback["overall"]["paired"]["full_gain_retained_percent"],
        preservation_margin=feedback["overall"]["paired"]["preservation_margin_auc"],
        noninferiority_supported_at_80_percent=feedback["overall"]["paired"]["noninferiority_supported_at_threshold"],
        joint_80_quality_20_cost_target_met=False,
        scope="Retained frozen-method quality, not a new optimized-backend quality evaluation. The original feedback80% noninferiority remains unestablished and complete cost remains above20%.")
    files = [Path(__file__), BUDGET, BRANCH_COST, FEEDBACK_QUALITY, *LEDGERS.values(),
        BASE / "probes" / NAME / "freeze.json",
        ROOT / "scripts/query_relay_2026_10/cost.py", ROOT / "scripts/implementation_2026_10/backend.py",
        ROOT / "scripts/implementation_2026_10/prepare.py", ROOT / "scripts/implementation_2026_10/state.py",
        ROOT / "scripts/implementation_2026_10/reader.py", ROOT / "scripts/implementation_2026_10/reader_kernel.py"]
    report = dict(status="complete_cpu_analytical_accounting", created_utc=datetime.now(timezone.utc).isoformat(),
        candidate=NAME, primary_protocol="stride16_candidates16", protocols=protocols,
        arithmetic_change="For each coded candidate or appended behavior row: old RMS L*H*(4d+2); packed gates reuse one H*(4d+2). Saved=(L-1)*H*(4d+2)*query_rows. Query projection, all-layer K/V gate projections/tanh, native attention and correction arithmetic remain charged.",
        unchanged_work="Full original calibration including origin once, teacher work, old conservative93888F per converted row, fixed-phi preparation, every full-catalogue D publication, actual small-block inverse preparation, and complete source reconstruction sensitivity are retained without deductions.",
        denominator_scope=audit["denominator_scope"], service_total_scope=audit["service_total_scope"],
        preparation_policy="Four independent replica preparations; serving populations partitioned across replicas, not multiplied by four.",
        persistent_memory=memory, lookup_memory=deepcopy(candidate_cost["extra_lookup_memory"]),
        original_feedback_boundary=feedback_boundary,
        limitations=["Fusion reduces history-sized intermediates and code expansion; it is not credited with lower logical attention/correction FLOPs.",
            "TF32x3, padded rank16 and query/history tiles may execute more hardware instruction work than logical FLOPs. This accounting is not a Tensor Core instruction counter.",
            "Table/gate batching reduces duplicate input reads and launch geometry; modelled bytes are separate from arithmetic and do not prove loaded throughput or p99.",
            "In-place conversion reduces scratch; existing append/FIFO interfaces may temporarily allocate new buffers.",
            "No new quality estimate or change to frozen models, users, requests, labels, training or acceptance workloads."],
        source_bindings=[bind(path) for path in files])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Implementation 完整理论成本", "", "固定30k因果主账本和原反馈账本独立计价，只扣跨层重复RMS。没有重新训练或改变质量任务。", "",
        "| 任务 | 旧完整新增成本% | 新完整新增成本% | 含完整来源重建% | 确定节省FLOPs |", "| --- | ---: | ---: | ---: | ---: |"]
    for protocol, data in protocols.items():
        lines.append(f"| {protocol} | {data['original_complete_percent_of_once_rebuild']:.8f} | {data['complete_percent_of_once_rebuild']:.8f} | {data['complete_with_source_reconstruction_percent']:.8f} | {data['saved_rms_flops']} |")
    lines += ["", "20%的分母仍是每用户、每发布首次推荐的一次K/V重建；普通增量写入和完整推荐查询另计。不能将上述比例解释为全服务FLOPs比例。", "",
        "优化仅把每个有效纠正Query的六次RMS减少到一次：`(L−1)×H×(4d+2)×查询数`。原非零前缀计数、原RMS组成、全部非RMS读取工作、93888 FLOPs/行转换、完整校准和来源重建均通过逐项整数核对。码准备的批量投影、融合扫描和原地转换不再扣减理论FLOPs。", "",
        f"发布级另加门控权重{gate_bytes:,}B和读取投影副本{decoder_bytes:,}B/副本，合计{derived:,}B；整理参数没有浮点投影，但有复制字节和临时分配。稳态共享空间为{memory['shared_state_bytes_per_replica']:,}B/副本，四副本{memory['shared_state_bytes_four_replicas']:,}B；旧/新交接共享空间另列。逐行码仍1152B，相对于9216B原生K/V为12.5%，额外逐行状态为0。", "",
        "Current物品门控lookup仍按包括零码操作在内的全部候选/行为收费；主账本41,760,119,808B，校准47,923,200B，来源捕获字节保留原记录。源码及输入哈希见summary.json。", "",
        "原反馈任务的80%恢复非劣性未建立，新完整成本仍高于20%；此工程优化不改变这一质量/成本边界。矩阵核TF32x3、补齐坐标和重复Query块的实际指令/流量不等于逻辑FLOPs，不据此宣称加载服务吞吐或p99。", ""]
    output.with_suffix(".md").write_text("\n".join(lines))
    for protocol, data in protocols.items():
        print(json.dumps(dict(protocol=protocol, saved_rms_flops=data["saved_rms_flops"],
            complete_percent=data["complete_percent_of_once_rebuild"],
            with_source_percent=data["complete_with_source_reconstruction_percent"],
            checks=data["checks"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
        default=ROOT / "results/implementation_2026_10/cost/summary.json")
    run(parser.parse_args().output)
