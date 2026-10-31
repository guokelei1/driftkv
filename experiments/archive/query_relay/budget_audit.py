#!/usr/bin/env python3
"""Price observed candidate stages against the retained causal CPU ledgers.

This is a small development audit, not a quality experiment. Missing stages
stay missing. The compiler's exact arithmetic geometry is shown separately
when its full-catalogue publication has not yet completed.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
from hstu_kvcache.methods.cachetide.cost import price

BASE = ROOT / "results/query_relay_2026_10"
VERSIONS = ("v3", "v4", "v5")
DEFAULT_BRANCHES = ("A_origin6_rms_roll4", "B_origin6_rms_roll4",
                    "A_localorigin6_rms_roll4")


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def publication_preview(configuration, rows, version):
    """Same branch geometry as publish.py; never labelled an executed artifact."""
    layers, embedding, code = 6, 192, 48
    xi = 1 if configuration["mode"] == "A" else 3
    widths = (12, 6, 6, 6, 6, 6, 6)
    squares, cubes = sum(w * w for w in widths), sum(w**3 for w in widths)
    cayley = layers * sum((8 * w**3 + 2) // 3 + 6 * w**2 for w in widths)
    work = dict(item_projection_flops=layers * rows * (2 * embedding * code * xi + code * xi),
                history_composition_flops=0 if version == "v3" else
                    layers * rows * (2 * squares * xi + code * xi),
                matrix_composition_flops=0 if version == "v3" else
                    layers * (4 * cubes + squares),
                cayley_preparation_flops_estimate=cayley)
    return dict(catalogue_table_rows=rows,
                adapter_config=dict(mode=configuration["mode"], xi_width=xi,
                    feature_mode=configuration["feature_mode"],
                    gate_normalization=configuration["gate_normalization"],
                    read_shared=configuration["read_shared"],
                    read_private=configuration["read_private"]),
                actual_compilation_components_flops=work,
                audit_scope="Arithmetic geometry preview only; full-catalogue artifact pending.")


def gather_candidate(name, rows):
    directory = BASE / "probes" / name
    stages, publications, publication_sources = {}, {}, {}
    first = read(directory / "calibration/v3/calibration.json")
    origin = first["cost"]["warmstart_origin"]
    if not origin["charge_separately"] or origin["calibration_flops"] is None:
        raise ValueError(f"{name}: originating calibration cost is not known")
    config = {key: first[key] for key in (
        "mode", "read_shared", "read_private", "feature_mode", "gate_normalization")}
    config["gate_source"] = first.get("gate_source", "layer")
    for version in VERSIONS:
        path = directory / "calibration" / version / "calibration.json"
        if path.exists():
            data = read(path)
            if any(data.get(key, "layer" if key == "gate_source" else None) != value
                   for key, value in config.items()):
                raise ValueError(f"{name}/{version}: candidate geometry changed")
            cost = data["cost"]
            if sum(cost["components"].values()) != cost["stage_calibration_flops"]:
                raise ValueError(f"{name}/{version}: stage component sum differs")
            if version != "v3" and cost["warmstart_origin"]["charge_separately"]:
                raise ValueError("A later originating fit needs explicit accounting")
            stages[version] = dict(binding=binding(path),
                stage_calibration_flops=cost["stage_calibration_flops"],
                components_flops=cost["components"],
                omitted_source_reconstruction_flops=cost["source_reconstruction_sensitivity"]["additional_flops"],
                fit_users=data["fit_users"], validation_users=data["validation_users"],
                local_epochs=data["local_epochs"], joint_epochs=data["joint_epochs"],
                rollout_steps=data["rollout_steps"], weights_sha256=data["weights_sha256"])
        path = directory / "publication" / (version + ".json")
        if path.exists():
            publications[version] = read(path)
            if version not in stages:
                raise ValueError(f"{name}/{version}: published calibration is absent")
            if publications[version]["bindings"]["calibration_sha256"] != stages[version]["weights_sha256"]:
                raise ValueError(f"{name}/{version}: publication binds different calibrated weights")
            if publications[version]["feature_bank"]["sha256"] != first["feature_bank_sha256"]:
                raise ValueError(f"{name}/{version}: immutable origin bank changed")
            expected = publication_preview(config, rows, version)
            if publications[version]["actual_compilation_components_flops"] != expected["actual_compilation_components_flops"]:
                raise ValueError(f"{name}/{version}: compiler preview differs from actual arithmetic")
            publication_sources[version] = dict(status="completed_full_catalogue_publication",
                binding=binding(path), artifact_sha256=publications[version]["artifact_sha256"],
                producer_had_full_cuda_table=publications[version]["producer_had_full_cuda_table"])
        else:
            publications[version] = publication_preview(config, rows, version)
            publication_sources[version] = dict(status="unexecuted_compiler_geometry_preview")
    calibration = origin["calibration_flops"] + sum(s["stage_calibration_flops"] for s in stages.values())
    source = sum(s["omitted_source_reconstruction_flops"] for s in stages.values())
    return dict(name=name, config=config, initial_origin=origin, stages=stages,
        missing_calibration_versions=[v for v in VERSIONS if v not in stages],
        observed_calibration_flops_including_origin_once=calibration,
        observed_source_reconstruction_sensitivity_flops=source,
        source_scope="Stage V3 sensitivity already includes originating reconstruction; do not add that origin again. Later stage sensitivities are paid once each.",
        publication_sources=publication_sources), publications


def summarize_price(result):
    fields = ("calibration_flops", "maintenance_flops", "extra_ranking_flops",
              "extra_writer_flops", "complete_added_flops", "once_rebuild_flops",
              "complete_percent_of_once_rebuild", "source_reconstruction_sensitivity_flops",
              "complete_with_source_reconstruction_percent", "remaining_calibration_budget_flops",
              "extra_percent_of_native_whole_query", "extra_percent_of_native_append",
              "native_history_attention_flops", "native_inclusive_self_attention_flops",
              "native_history_and_self_attention_flops", "extra_percent_of_native_history_attention",
              "extra_percent_of_native_history_and_self_attention",
              "common_native_incremental_append_flops", "common_native_whole_query_flops",
              "total_modeled_service_full_flops", "total_modeled_service_relay_flops",
              "total_modeled_service_relay_with_source_reconstruction_flops",
              "net_saved_flops", "net_saved_with_source_reconstruction_flops",
              "net_saved_percent_of_total_full_service", "net_saved_with_source_percent_of_total_full_service",
              "net_kv_rebuild_replacement_saving_percent", "net_kv_rebuild_replacement_with_source_saving_percent",
              "catalogue_table_rows", "shared_state_publication_transfer_bytes")
    fields += ("extra_gate_item_lookup_bytes", "stable_anchor_inverse", "gate_source")
    return {key: result[key] for key in fields}


def audit(names=DEFAULT_BRANCHES, *, stable_anchor_inverse=False):
    ledger_paths = {"original_feedback": BASE / "cost/event_ledger.json"}
    ledger_paths.update({f"stride{s}_candidates{n}": BASE / f"workloads/stride{s}_candidates{n}_ledger.json"
                         for s in (8, 16, 32) for n in (2, 16)})
    ledgers = {name: read(path) for name, path in ledger_paths.items()}
    primary = ledgers["stride16_candidates16"]
    rows = primary["model_config"]["num_items"] + 1
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope="Theoretical FLOPs on actual fixed30k causal event shapes and observed calibration operations; no GPU timing or quality claim.",
        primary_protocol="stride16_candidates16", primary_preparation_policy="four_independent_replica_preparations",
        denominator_scope="Full/Rebuild once per user/release at first recommendation then normal incremental service; every protocol has its own causal first-request denominator. Sum FLOPs before division.",
        service_total_scope="Full=native incremental append+native complete query+first-recommendation KV reconstruction. Relay=the same native append/query+all method-added FLOPs. The20% budget uses only KV reconstruction; total service savings have a separate full-service denominator. Initial V2 cache creation, foundation training, retrieval infrastructure and loaded wall time are outside this modeled phase.",
        native_attention_scope="Ranking history QK+scale/ELU+1+AV charged from actual dense pairs as L*(4D+3H); inclusive self attention is L*(4D+2H) per candidate. Complete native query also includes embeddings, QKV/output/gate projections, norms and scoring.",
        source_reconstruction_scope="Available native source trajectories are primary; required reconstruction is a separately reported sensitivity, never called free.",
        protocols={}, candidates={})
    for name, ledger in ledgers.items():
        report["protocols"][name] = dict(binding=binding(ledger_paths[name]), users=ledger["users"],
            candidate_scores=ledger["requests"], recommendation_rounds=ledger.get("recommendation_rounds"),
            once_rebuild_flops=ledger["once_rebuild_flops"],
            all_resident_eager_rebuild_flops=ledger["all_resident_eager_rebuild_flops"],
            native_whole_query_flops=sum(r["native_query_flops"] for r in ledger["native_by_release"].values()),
            native_incremental_append_flops=sum(r["native_append_flops"] for r in ledger["native_by_release"].values()),
            scope="Original observed feedback candidate rows" if name == "original_feedback" else
                "Pre-registered causal synthetic next-item rounds, each with one positive and uniform negatives; candidate count is not a request-round count.")
    for name in names:
        candidate, publications = gather_candidate(name, rows)
        cfg = candidate["config"]
        kwargs = dict(branch=cfg["mode"], xi_width=1 if cfg["mode"] == "A" else 3,
            head_rank=cfg["read_shared"] + cfg["read_private"], gate_normalization=cfg["gate_normalization"],
            feature_mode=cfg["feature_mode"], publication_metadata=publications,
            gate_source=cfg["gate_source"], stable_anchor_inverse=stable_anchor_inverse,
            calibration_flops=candidate["observed_calibration_flops_including_origin_once"],
            source_reconstruction_flops=candidate["observed_source_reconstruction_sensitivity_flops"])
        candidate["cost_status"] = "all_calibration_stages_observed" if not candidate["missing_calibration_versions"] else "observed_calibration_lower_bound_pending_stages"
        candidate["publication_status"] = "all_full_catalogue_artifacts_observed" if all(
            p["status"] == "completed_full_catalogue_publication" for p in candidate["publication_sources"].values()) else "unpublished_components_use_explicit_compiler_geometry_preview"
        candidate["protocol_prices"] = {}
        for protocol, ledger in ledgers.items():
            result = price(ledger, **kwargs)
            compact = summarize_price(result)
            compact["unspent_calibration_budget_after_observed_stages_flops"] = result["remaining_calibration_budget_flops"] - kwargs["calibration_flops"]
            compact["cost_status"] = candidate["cost_status"]
            compact["missing_calibration_versions"] = candidate["missing_calibration_versions"]
            compact["source_status"] = "Full sensitivity" if not candidate["missing_calibration_versions"] else "Observed sensitivity lower bound; future source captures unpriced"
            candidate["protocol_prices"][protocol] = compact
            if protocol in ("original_feedback", "stride16_candidates16"):
                candidate.setdefault("detailed_primary_and_original", {})[protocol] = result
        broadcast = price(primary, **kwargs, preparation_mode="compile_once_broadcast")
        candidate["centralized_compilation_broadcast_sensitivity"] = summarize_price(broadcast)
        main = candidate["detailed_primary_and_original"]["stride16_candidates16"]
        d_bytes = main["by_release"]["v5"]["shared_item_table_payload_bytes_per_replica"]
        matrices = main["by_release"]["v5"]["shared_composition_matrix_payload_bytes_per_replica"]
        bank = main["immutable_origin_feature_bank"]["payload_bytes_per_replica"]
        inverse = main["by_release"]["v5"]["global_anchor_inverse_payload_bytes_per_replica"]
        final_rows = primary["by_release"]["v5"]["final_history_rows"]
        candidate["persistent_memory"] = dict(code_bytes_per_row=1152, native_kv_bytes_per_row=9216,
            code_fraction_of_native_kv=.125, total_population_final_rows=final_rows,
            total_population_code_bytes=final_rows * 1152, total_population_native_kv_bytes=final_rows * 9216,
            active_compiled_item_table_bytes_per_replica=d_bytes,
            active_dense_P_Q_bytes_per_replica=matrices,
            active_actual_block_inverse_bytes_per_replica=inverse,
            immutable_origin_bank_bytes_per_replica=bank,
            shared_state_bytes_per_replica=d_bytes + matrices + bank + inverse,
            shared_state_bytes_four_replicas=4 * (d_bytes + matrices + bank + inverse),
            scope="One current D/P/Q and one immutable feature bank per replica; row-state population is partitioned across replicas, not multiplied by four. Publication peak includes old and new D. Retained disk artifacts are evidence, not concurrently active serving tables.",
            publication_peak_shared_state_bytes_per_producer=2 * (d_bytes + matrices + inverse) + bank)
        if candidate["missing_calibration_versions"]:
            first_stage = candidate["stages"]["v3"]["stage_calibration_flops"]
            projection = price(ledgers["original_feedback"], **{**kwargs,
                "calibration_flops": candidate["initial_origin"]["calibration_flops"] + 3 * first_stage})
            candidate["same_V3_stage_three_release_projection_original_feedback"] = dict(
                status="Planning sensitivity only; missing stages are not completed", V3_stage_flops=first_stage,
                projected_complete_percent=projection["complete_percent_of_once_rebuild"],
                remaining_actual_stage_budget_flops=candidate["protocol_prices"]["original_feedback"]["unspent_calibration_budget_after_observed_stages_flops"],
                missing_stage_count=len(candidate["missing_calibration_versions"]))
        if cfg["mode"] == "A":
            # Existing r6 fits cannot establish r18 calibration cost or quality.
            capacity = price(primary, **{**kwargs, "head_rank": 18, "calibration_flops": 0,
                "source_reconstruction_flops": 0, "publication_metadata": None})
            candidate["full18_capacity_sensitivity"] = dict(status="Unfitted capacity estimate; no quality claim",
                noncalibration_percent=capacity["complete_percent_of_once_rebuild"],
                remaining_calibration_budget_flops=capacity["remaining_calibration_budget_flops"],
                note="All 18 read coordinates exceed20% before calibration on the registered16/16 workload. r6 calibration cannot be reused as an r18 complete-cost claim.")
        report["candidates"][name] = candidate
    return report


def brief(report):
    lines = ["# Query relay development budget audit", "", report["scope"], "",
        "Primary: 16行为触发一次推荐、每轮16候选；4副本独立准备。原feedback和已注册频率敏感性保留各自的Full/Rebuild分母。", "",
        "| Candidate | observed calibration TF | 16/16 added % | original feedback added % | 16/16 incl. source % | status |",
        "| --- | ---: | ---: | ---: | ---: | --- |"]
    for name, c in report["candidates"].items():
        main, old = (c["protocol_prices"][p] for p in ("stride16_candidates16", "original_feedback"))
        status = "校准三版完成" if not c["missing_calibration_versions"] else "待校准" + ",".join(c["missing_calibration_versions"])
        if c["publication_status"] != "all_full_catalogue_artifacts_observed":
            status += "；未发布项按显式编译几何预测"
        lines.append(f"| {name} | {c['observed_calibration_flops_including_origin_once']/1e12:.6f} | {main['complete_percent_of_once_rebuild']:.5f} | {old['complete_percent_of_once_rebuild']:.5f} | {main['complete_with_source_reconstruction_percent']:.5f} | {status} |")
    lines += ["", "缺少校准版时，上表仅为已观察校准加完整服务开销的下界，不能称完整候选结果。重建源K/V的敏感性已含原始origin一次，不重复叠加。", ""]
    for name, c in report["candidates"].items():
        main = c["protocol_prices"]["stride16_candidates16"]
        mem = c["persistent_memory"]
        lines.append(f"- {name}：主口径额外读开销占native完整query {main['extra_percent_of_native_whole_query']:.3f}%；全局共享状态每副本 {mem['shared_state_bytes_per_replica']/1e9:.3f} GB，4副本 {mem['shared_state_bytes_four_replicas']/1e9:.3f} GB，另加逐行48维码（native K/V的12.5%）。")
        if main["stable_anchor_inverse"]:
            lines.append("  当前发布的实际小块逆矩阵另存8,640B/副本；FP64发布级求逆按LU算法估计48,384 FLOPs/发布/准备副本，四副本三版合计580,608 FLOPs。每行状态和矩阵乘计算量不增加。")
        if main["extra_gate_item_lookup_bytes"]:
            lines.append(f"  Current物品门控另有 {main['extra_gate_item_lookup_bytes']/1e9:.6f} GB瞬时重复lookup载荷：每候选/追加行为768B，包含空纠正前缀的实际lookup，六层共用一次输入。仅披露字节，没有新增投影FLOPs或持久特征；不能推断实际延迟免费。")
        lines.append(f"  native历史＋self注意力的额外query开销为 {main['extra_percent_of_native_history_and_self_attention']:.3f}%。整个建模服务 Full={main['total_modeled_service_full_flops']/1e12:.6f} TF，Relay={main['total_modeled_service_relay_flops']/1e12:.6f} TF；净省 {main['net_saved_flops']/1e12:.6f} TF，即Full服务的 {main['net_saved_percent_of_total_full_service']:.3f}%。若源状态需重建，Relay={main['total_modeled_service_relay_with_source_reconstruction_flops']/1e12:.6f} TF，净省 {main['net_saved_with_source_reconstruction_flops']/1e12:.6f} TF（{main['net_saved_with_source_percent_of_total_full_service']:.3f}%）。")
        if "same_V3_stage_three_release_projection_original_feedback" in c:
            p = c["same_V3_stage_three_release_projection_original_feedback"]
            lines.append(f"  未完成版如均复用V3实际校准成本，原feedback预测 {p['projected_complete_percent']:.5f}%；其剩余{p['missing_stage_count']}版实际校准总预算 {p['remaining_actual_stage_budget_flops']/1e12:.6f} TF。这是规划敏感性。")
    lines += ["", "20%口径是全部新增成本/KV重建，不能解释成整个模型服务的FLOPs比例。整个建模服务包含三版期间真实历史追加、完整query及首次推荐KV重建；两条路径共同继承初始V2缓存，不含基础模型训练、检索基础设施或实际运行时间。", "", "所有数值均为理论FLOPs；query增幅与总重建分母分别报告。全局D表和immutable bank不属于逐行12.5%码占比。20%是成本边界，恢复率和多版本有效性仍需独立质量结果。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branches", nargs="+", default=DEFAULT_BRANCHES)
    parser.add_argument("--output", type=Path, default=BASE / "cost/current_candidate_budget.json")
    parser.add_argument("--stable-anchor-inverse", action="store_true")
    args = parser.parse_args()
    report = audit(args.branches, stable_anchor_inverse=args.stable_anchor_inverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(brief(report))
    print(brief(report))


if __name__ == "__main__":
    main()
