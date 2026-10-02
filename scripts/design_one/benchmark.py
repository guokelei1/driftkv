#!/usr/bin/env python3
"""Score and aggregate fixed Q-v5 C256 / H-v4 C128 Design 1 controls.

Prepare once with benchmark_data.py. Run one ``score`` process per UID shard,
then ``aggregate`` on the common output directory. --max-users uses only the
already frozen 128-user pilot; zero evaluates the entire 10000-user panel.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from design_one.benchmark_data import PANEL, checked, check_history, load_panel, select_uids
from evaluate_yambda500m_foundation_raw import load_histories, load_model
from hstu_kvcache.evaluation.binary_metrics import binary_metrics
from read_correction_2026_09.cost import normalize_metrics
from read_correction_v3.evaluate import cost_record
from read_correction_v4.evaluate_full import HISTOGRAMS, signature, validate_rows, verify_saved_unit
from read_correction_v5.common import configuration, sha256, sources, write_json
from read_correction_v5.scoring import load_policies, score_unit
from selective_recompute_2026_09.scheduling import ordered_uids


def design_mode(args):
    return args.design_calibration is not None or args.design_calibration_list is not None


def design_policy(folder, tag, panel, groups, *, producer_scope=None, execution_mode='mapped_kv'):
    from design_one.scoring import CONTEXT_KV_KINDS, ITEM_KV_KINDS, ITEM_RESPONSE_KIND, KV_KINDS, POLICY_METHODS
    folder = folder.resolve()
    adjudication = folder / 'adjudication.json'
    if adjudication.exists() and not json.loads(adjudication.read_text()).get('eligible_for_benchmark', True):
        raise RuntimeError(f'calibration is retained for diagnosis only: {adjudication}')
    path, weights = folder / "calibration.json", folder / "calibration.pt"
    record = json.loads(path.read_text())
    if (record["kind"] not in POLICY_METHODS
            or record["configuration"]["sha256"] != panel["configuration"]["sha256"]
            or record["users_file_sha256"] != panel["users_file"]["sha256"]
            or record["weights_sha256"] != sha256(weights)):
        raise RuntimeError("Design 1 calibration has different inputs or weights")
    fitting = record["fit_uids"] + record["validation_uids"]
    if (len(set(fitting)) != len(fitting) or set(fitting).intersection(groups["design1_evaluation"])
            or not set(record["fit_uids"]).issubset(groups["design1_fit"])
            or not set(record["validation_uids"]).issubset(groups["design1_validation"])):
        raise RuntimeError("Design 1 calibration violates fixed fitting/validation roles")
    if (record["cutover"] != panel["cutover"] or record["history_length"] != 1024
            or record["checkpoint_hashes"] != {"v4": panel["sources"]["parent"]["sha256"],
                                               "v5": panel["sources"]["current"]["sha256"]}):
        raise RuntimeError("Design 1 calibration release or endpoints differ")
    if producer_scope is not None and record['kind'] not in KV_KINDS:
        raise ValueError('producer_scope applies only to KV read-view policies')
    result = {"tag": tag, "method": POLICY_METHODS[record['kind']], "variant": record["kind"],
        "budget": len(record["fit_uids"]), "calibration_dir": str(folder),
        "calibration": {"path": str(path), "sha256": sha256(path), "weights_sha256": sha256(weights)},
        "calibration_flops": record["cost"]["calibration_flops"]}
    if record['kind'] in KV_KINDS:
        scope = 'all' if producer_scope is None else producer_scope
        if scope not in ('all', 'parent_only'):
            raise ValueError('producer_scope must be all or parent_only')
        if record['kind'] in (*CONTEXT_KV_KINDS, *ITEM_KV_KINDS) and scope != 'all':
            raise ValueError('context/item KV policies require producer_scope=all')
        result['producer_scope'] = scope
    if record['kind'] == ITEM_RESPONSE_KIND:
        result['initial_item_calibration'] = record['initial_item_calibration']
    if execution_mode not in ('mapped_kv', 'compact_hidden'):
        raise ValueError('execution_mode must be mapped_kv or compact_hidden')
    if execution_mode == 'compact_hidden' and record['kind'] != ITEM_RESPONSE_KIND:
        raise ValueError('compact_hidden execution requires the Item+Response artifact')
    result['execution_mode'] = execution_mode
    return result


def policies_for(args, panel, groups):
    if not design_mode(args):
        return panel["policies"]
    if args.design_calibration is not None:
        if args.design_calibration_list is not None:
            raise ValueError("choose a single calibration or a calibration list")
        return [design_policy(args.design_calibration, "Design1", panel, groups)]
    entries = json.loads(args.design_calibration_list.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("Design 1 calibration list must be a nonempty JSON list")
    policies, tags = [], set()
    for entry in entries:
        tag = entry["tag"]
        if not isinstance(tag, str) or not tag or Path(tag).name != tag or tag in tags:
            raise ValueError("calibration tags must be unique file-name components")
        tags.add(tag)
        policies.append(design_policy(ROOT / entry["calibration_dir"], tag, panel, groups,
                                      producer_scope=entry.get('producer_scope'),
                                      execution_mode=entry.get('execution_mode', 'mapped_kv')))
    return policies


def calibration_list_binding(args):
    path = args.design_calibration_list
    return {"path": str(path.resolve()), "sha256": sha256(path)} if path is not None else None


def pending_setup_flops(policies, units):
    """Fold once per worker; completed unit ledgers retain the existing charge."""
    return {policy['tag']: policy['adapter'].setup_flops() for policy in policies
            if policy.get('execution_mode') == 'compact_hidden'
            and not any(unit and unit.get('execution_setup_flops', {}).get(policy['tag'], 0)
                        for unit in units)}


def inputs_for(args, panel, groups, by_user, entries):
    uids = select_uids(groups, args.max_users, args.shard_index, args.num_shards)
    if not uids:
        raise ValueError("empty UID shard")
    cfg = configuration()
    cfg["evaluation_users"] = len(groups["design1_evaluation"])
    cfg["cohort_sizes"]["medium"] = args.cohort_size
    cfg["query_batches"]["medium"] = args.query_batch
    uids = ordered_uids(by_user, max_length=cfg["history_length"], uids=uids)
    execution = sources()
    for filename in ("benchmark.py", "benchmark_data.py"):
        path = Path(__file__).with_name(filename)
        execution[str(path.relative_to(ROOT))] = sha256(path)
    history_inputs = None
    if args.history_packs is not None:
        from design_one.history import history_binding
        history_inputs = history_binding(args.history_packs, panel, groups)
        path = Path(__file__).with_name("history.py")
        execution[str(path.relative_to(ROOT))] = sha256(path)
    if design_mode(args):
        path = Path(__file__).with_name("scoring.py")
        execution[str(path.relative_to(ROOT))] = sha256(path)
        for path in sorted((ROOT / "src/hstu_kvcache/design_one").rglob("*.py")):
            execution[str(path.relative_to(ROOT))] = sha256(path)
        for name in ('reader.py', 'summary.py'):
            path = ROOT / 'src/hstu_kvcache/adaptation' / name
            execution[str(path.relative_to(ROOT))] = sha256(path)
    return {"panel_sha256": sha256(args.panel / "binding.json"),
        "execution_sources": execution, "settings": cfg,
        "max_users": args.max_users, "num_shards": args.num_shards,
        "shard_index": args.shard_index, "uids": uids, "unit_users": args.unit_users,
        "policies": entries, "history_packs": history_inputs,
        "design_calibration_list": calibration_list_binding(args),
        "verify": "all pilot units; first full-panel unit per shard"}


def score(args):
    started = time.perf_counter()
    panel, groups, by_user = load_panel(args.panel)
    entries = policies_for(args, panel, groups)
    inputs = inputs_for(args, panel, groups, by_user, entries)
    binding, uids = signature(inputs), inputs["uids"]
    cfg = json.loads(json.dumps(inputs["settings"]))
    load_method, score_method = load_policies, score_unit
    if design_mode(args):
        from design_one.scoring import load_policies as load_method, score_unit as score_method
    for entry in entries:
        path = checked(entry["calibration"])
        record = json.loads(path.read_text())
        if sha256(path.with_suffix(".pt")) != entry["calibration"]["weights_sha256"]:
            raise RuntimeError("historical calibration weights changed")
        fitting = record["fit_uids"] if design_mode(args) else record["uids"]
        if set(fitting + record["validation_uids"]).intersection(groups["design1_evaluation"]):
            raise RuntimeError("calibration/validation users overlap evaluation")
    tags = [entry["tag"] for entry in entries]
    output = args.output.resolve() / f"shard_{args.shard_index:02d}"
    shard_path = output / "shard.json"
    if shard_path.exists():
        old = json.loads(shard_path.read_text())
        if old["input_signature"] != binding:
            raise RuntimeError("completed shard has different inputs")
        for record in old["scores"].values():
            checked(record)
        print(json.dumps({"status": "already_complete", "output": str(shard_path)}), flush=True)
        return old
    write_json(output / "inputs.json", {**inputs, "input_signature": binding})
    units = []
    for index, start in enumerate(range(0, len(uids), args.unit_users)):
        path = output / "units" / f"unit_{index:05d}.json"
        record = json.loads(path.read_text()) if path.exists() else None
        if record is not None:
            verify_saved_unit(record, binding, uids[start:start + args.unit_users], tags)
            cfg["cohort_sizes"]["medium"] = min(cfg["cohort_sizes"]["medium"], record["cohort_size"])
            cfg["query_batches"]["medium"] = min(cfg["query_batches"]["medium"], record["query_batch"])
        units.append(record)
    model_seconds = history_seconds = 0.
    if any(unit is None for unit in units):
        os.environ["EVOKV_ATTENTION_BACKEND"] = cfg["attention_backend"]
        torch.set_num_threads(cfg["torch_threads"])
        pa.set_cpu_count(cfg["history_threads"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_per_process_memory_fraction(cfg["memory_fraction"], device)
        torch.cuda.reset_peak_memory_stats(device)
        beginning = time.perf_counter()
        # These are the fixed, admitted V4/V5 endpoints, checked before GPU work.
        parent, parent_payload = load_model(checked(panel["sources"]["parent"]), device)
        current, current_payload = load_model(checked(panel["sources"]["current"]), device)
        if parent_payload["config"] != current_payload["config"]:
            raise RuntimeError("parent/current architectures differ")
        dataset_path = checked(panel["sources"]["dataset"])
        dataset = json.loads(dataset_path.read_text())
        known = int(current_payload.get("known_vocab_size", dataset["foundation_items"]))
        parent.requires_grad_(False)
        current.requires_grad_(False)
        policies = load_method(entries, device)
        # Each independent scoring worker folds its decoder once. Resuming
        # completed units retains their charge; cohort/unit boundaries add none.
        setup_pending = pending_setup_flops(policies, units)
        if not design_mode(args) and any(len(policy["modules"]) != len(current.blocks) for policy in policies):
            raise RuntimeError("correction layer count differs from model")
        if design_mode(args):
            for policy in policies:
                adapter = policy['adapter']
                layers = adapter.parameters if policy['variant'] == 'design_one_shared_read_v1' else adapter.layers
                if len(layers) != len(current.blocks) or adapter.max_length != cfg['history_length']:
                    raise RuntimeError('Design 1 adapter layers or history length differ from model')
        del parent_payload, current_payload
        model_seconds = time.perf_counter() - beginning
        remaining = [uid for index, start in enumerate(range(0, len(uids), args.unit_users))
                     if units[index] is None for uid in uids[start:start + args.unit_users]]
        beginning = time.perf_counter()
        if args.history_packs is not None:
            from design_one.history import load_pack
            history = load_pack(inputs["history_packs"], remaining, known_vocab_size=known,
                                oov_buckets=current.cfg.num_items - known)
        else:
            history = load_histories(remaining, dataset_path=dataset_path, known_vocab_size=known,
                oov_buckets=current.cfg.num_items - known, start_timestamp=panel["cutover"],
                end_timestamp=panel["days"][1] * 86400, max_history=cfg["history_length"],
                threads=cfg["history_threads"])
        history_seconds = time.perf_counter() - beginning
        for index, start in enumerate(range(0, len(uids), args.unit_users)):
            if units[index] is not None:
                continue
            selected = uids[start:start + args.unit_users]
            causal = check_history(history, by_user, selected, panel["cutover"], cfg["history_length"])
            reference = sorted([row for uid in selected for row in by_user[uid]], key=lambda row: row["request_id"])
            beginning, reductions = time.perf_counter(), []
            while True:
                try:
                    scores, histograms, controls = score_method(current, parent, history, by_user, selected,
                        panel["cutover"], policies, cfg, "medium", device, verify=bool(args.max_users) or index == 0)
                    break
                except torch.cuda.OutOfMemoryError:
                    cohort, queries = cfg["cohort_sizes"]["medium"], cfg["query_batches"]["medium"]
                    if cohort == queries == 1:
                        raise
                    reductions.append({"cohort_size": cohort, "query_batch": queries})
                    cfg["cohort_sizes"]["medium"], cfg["query_batches"]["medium"] = max(1, cohort // 2), max(1, queries // 2)
                    gc.collect()
                    torch.cuda.empty_cache()
            outputs = {}
            for tag, value in setup_pending.items():
                scores[tag][0]['execution_setup_flops'] = value
                scores[tag][0]['correction_flops'] += value
            for tag in tags:
                rows = validate_rows(scores[tag], reference)
                path = output / "units" / f"unit_{index:05d}.{tag}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
                outputs[tag] = {"path": str(path), "sha256": sha256(path), "rows": len(rows)}
            unit = {"input_signature": binding, "uids": selected, "requests": len(reference),
                "outputs": outputs, "histograms": histograms, "controls": controls,
                "causal_check": causal, "seconds": time.perf_counter() - beginning,
                "batch_reductions": reductions, "cohort_size": cfg["cohort_sizes"]["medium"],
                "query_batch": cfg["query_batches"]["medium"],
                "execution_setup_flops": dict(setup_pending),
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30}
            write_json(output / "units" / f"unit_{index:05d}.json", unit)
            units[index] = unit
            setup_pending.clear()
            print(json.dumps({"status": "scoring", "shard": args.shard_index, "users": start + len(selected),
                "total_users": len(uids), "seconds": unit["seconds"]}), flush=True)
    histograms = {key: Counter() for key in HISTOGRAMS}
    for unit in units:
        for key in HISTOGRAMS:
            histograms[key].update(unit["histograms"][key])
    reference = sorted([row for uid in uids for row in by_user[uid]], key=lambda row: row["request_id"])
    outputs = {}
    for tag in tags:
        table = pa.concat_tables([pq.read_table(unit["outputs"][tag]["path"]) for unit in units]).sort_by([("request_id", "ascending")])
        validate_rows(table.to_pylist(), reference)
        path = output / f"{tag}.parquet"
        pq.write_table(table, path, compression="zstd")
        outputs[tag] = {"path": str(path), "sha256": sha256(path), "rows": len(table)}
    result = {"status": "complete", "input_signature": binding, "inputs": inputs,
        "users": len(uids), "requests": len(reference), "scores": outputs,
        "histograms": {key: dict(value) for key, value in histograms.items()},
        "controls": {key: (sum if key == "requests" else max)(unit["controls"][key] for unit in units)
                     for key in units[0]["controls"]},
        "causal_checked_requests": sum(unit["causal_check"]["requests"] for unit in units),
        "model_load_seconds": model_seconds, "history_load_seconds": history_seconds,
        "scoring_seconds": sum(unit["seconds"] for unit in units),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_gib": max(unit["peak_allocated_gib"] for unit in units),
        "execution_setup_flops": {tag: sum(unit.get('execution_setup_flops', {}).get(tag, 0)
                                            for unit in units) for tag in tags},
        "cost_scope": "shard inference only; aggregate charges historical calibration once per policy"}
    write_json(shard_path, result)
    print(json.dumps({"status": "complete", "output": str(shard_path)}), flush=True)
    return result


def aggregate(args):
    panel, groups, by_user = load_panel(args.panel)
    entries = policies_for(args, panel, groups)
    history_inputs = None
    if args.history_packs is not None:
        from design_one.history import history_binding
        history_inputs = history_binding(args.history_packs, panel, groups)
    selected = select_uids(groups, args.max_users)
    expected = sorted([row for uid in selected for row in by_user[uid]], key=lambda row: row["request_id"])
    histograms = {key: Counter() for key in HISTOGRAMS}
    shards, seen = [], set()
    for rank in range(args.num_shards):
        path = args.output / f"shard_{rank:02d}" / "shard.json"
        shard = json.loads(path.read_text())
        inputs = shard["inputs"]
        if (shard["status"] != "complete" or inputs["panel_sha256"] != sha256(args.panel / "binding.json")
                or inputs["max_users"] != args.max_users or inputs["num_shards"] != args.num_shards
                or inputs["shard_index"] != rank or inputs["policies"] != entries
                or inputs.get("history_packs") != history_inputs
                or inputs.get("design_calibration_list") != calibration_list_binding(args)
                or signature(inputs) != shard["input_signature"]):
            raise RuntimeError("inconsistent or incomplete benchmark shards")
        if shards and inputs["execution_sources"] != shards[0]["inputs"]["execution_sources"]:
            raise RuntimeError("benchmark shards used different scoring sources")
        shard_uids = set(inputs["uids"])
        if shard_uids != set(select_uids(groups, args.max_users, rank, args.num_shards)) or seen.intersection(shard_uids):
            raise RuntimeError("overlapping or missing shard users")
        seen.update(shard_uids)
        for key in HISTOGRAMS:
            histograms[key].update(shard["histograms"][key])
        shards.append(shard)
    if seen != set(selected) or sum(s["causal_checked_requests"] for s in shards) != len(expected):
        raise RuntimeError("not all frozen requests passed causal checks")
    histograms = {key: dict(value) for key, value in histograms.items()}
    labels = np.asarray([row["label"] for row in expected])
    full = binary_metrics(labels, np.asarray([row["full_logit"] for row in expected]))
    reuse = binary_metrics(labels, np.asarray([row["reuse_logit"] for row in expected]))
    points, outputs = [], {}
    for entry in entries:
        tag = entry["tag"]
        table = pa.concat_tables([pq.read_table(checked(s["scores"][tag])) for s in shards]).sort_by([("request_id", "ascending")])
        rows = validate_rows(table.to_pylist(), expected)
        measured = binary_metrics(labels, table["hstu_logit"].to_numpy())
        calibration = json.loads(checked(entry["calibration"]).read_text())
        ledger = cost_record("medium", histograms, sum(row["correction_flops"] for row in rows),
                             calibration["cost"]["calibration_flops"])
        # Include inherited cache initialization in the separately reported lifecycle total.
        ledger["method_with_initial_cache_including_calibration_flops"] = (
            ledger["method_steady_including_calibration_flops"] + ledger["initial_parent_cache_flops"])
        for key in ("summary_initial_flops", "summary_update_flops", "view_flops", "read_flops",
                    "view_initial_flops", "view_update_flops", "context_update_flops", "execution_setup_flops"):
            if key in table.column_names:
                ledger[key] = sum(int(row[key]) for row in rows)
        for key in ('extra_history_read_flops', 'response_flops'):
            if key in table.column_names:
                ledger[key] = sum(int(row[key]) for row in rows)
        if 'item_lookup_bytes' in table.column_names:
            ledger['item_lookup_bytes'] = sum(int(row['item_lookup_bytes']) for row in rows)
        for key in ('persistent_extra_peak_bytes', 'native_kv_peak_bytes',
                    'persistent_extra_to_native_ratio_peak', 'shared_projection_bytes'):
            if key in table.column_names:
                ledger[key] = max(row[key] for row in rows)
        normalized = normalize_metrics(full_auc=full["ROC_AUC"], reuse_auc=reuse["ROC_AUC"],
            baseline_auc=measured["ROC_AUC"], extra_flops=ledger["extra_flops"],
            full_minus_reuse_flops=ledger["full_minus_reuse_flops"])
        path = args.output.resolve() / f"{tag}.parquet"
        pq.write_table(table, path, compression="zstd")
        outputs[tag] = {"path": str(path), "sha256": sha256(path), "rows": len(rows)}
        points.append({"tag": tag, "method": entry["method"], "variant": entry["variant"],
            "execution_mode": entry.get('execution_mode', 'mapped_kv'),
            "calibration_users": entry["budget"], "users": len(selected), "requests": len(rows),
            "full_auc": full["ROC_AUC"], "reuse_auc": reuse["ROC_AUC"], "baseline_auc": measured["ROC_AUC"],
            "auc_difference_from_reuse": measured["ROC_AUC"] - reuse["ROC_AUC"] if measured["ROC_AUC"] is not None else None,
            "metrics": measured, **ledger, **normalized})
    result = {"status": "complete", "scale": "medium", "edge": "v4_to_v5",
        "evaluation_role": "development_exploration", "probe_only": bool(args.max_users),
        "users": len(selected), "requests": len(expected), "num_shards": args.num_shards,
        "panel_sha256": sha256(args.panel / "binding.json"), "points": points, "scores": outputs,
        "current_full": full, "current_reuse": reuse, "histograms": histograms,
        "controls": {key: (sum if key == "requests" else max)(s["controls"][key] for s in shards)
                     for key in shards[0]["controls"]},
        "cost_scope": "Actual complete selected workload; full original calibration charged once per policy, including teacher access. Common read, append and initial parent cache costs explicit. Extra correction includes read-view initialization, every executed appended-token transformation and write-context update arithmetic; parent_only KV views retain native current-producer appends without a learned map. KV views pay the additional mapped history read while the native read still executes. Verification Full/identity forwards are timing overhead, not method FLOPs.",
        "timing_scope": "Methods within a run share snapshot traversal; scoring seconds include numerical verification. Per-policy GPU latency is not separately measured.",
        "persistent_memory_scope": "Per-policy lifecycle peaks are measured from stored cohort state tensors and corresponding native K/V bytes. The ratio is evaluated at each initialization/append then maximized, not computed from unmatched peaks. Compact state contains only SiLU hidden activations; mapped_kv contains the full mapped K/V. Query-expanded tensors, temporary item features and workspace are excluded. shared_projection_bytes is a separate per-worker folded-decoder allocation; base model/adapter weights remain shared. execution_setup_flops includes one actual decoder fold per scoring shard, retained on resume, and is already included in correction_flops.",
        "item_response_cost_scope": "mapped_kv materializes the Item view and reads it before Response. compact_hidden stores only Item hidden activations and computes the algebraically equivalent read from native K/V plus low-rank contractions, without materializing mapped K/V. In either execution mode, the complete extra history read and the Response network are disjoint components included once in read_flops; the common native read still executes. There is no additional score pass or cross-candidate response reuse. The same artifact's full calibration ledger, including inherited Item calibration once, is charged unchanged; decoder folding is separately charged once per worker.",
        "item_feature_memory_scope": "Item-aware views read the already resident Current model item_emb table; its bytes are base-model memory, not additional adapter storage. item_lookup_bytes counts logical embedding values fetched at initial state and real appends, once per entry rather than once per layer. Lookup has zero arithmetic FLOPs. mapped_kv charges the full dense map; compact_hidden charges only its hidden encoder at writes and the factored decoder contractions at reads. Temporary item features are discarded after encoding; no duplicate item table is retained. Producer-scaled item views leave Parent residuals unchanged and multiply each new Current residual by its stored layer scalar; these multiplications are included once in view_update_flops, with no persistent producer mask.",
        "scoring_worker_seconds": sum(s["scoring_seconds"] for s in shards),
        "max_shard_scoring_seconds": max(s["scoring_seconds"] for s in shards),
        "shard_elapsed_seconds": [s["elapsed_seconds"] for s in shards],
        "peak_allocated_gib": max(s["peak_allocated_gib"] for s in shards),
        "calibrations": entries, "history_packs": history_inputs,
        "design_calibration_list": calibration_list_binding(args)}
    write_json(args.output / "summary.json", result)
    print(json.dumps({"status": "complete", "users": len(selected), "requests": len(expected),
        "points": [{k: p[k] for k in ("tag", "baseline_auc", "recovery_percent", "relative_flops_percent")} for p in points]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("score", "aggregate"))
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--unit-users", type=int, default=256)
    parser.add_argument("--cohort-size", type=int, default=128)
    parser.add_argument("--query-batch", type=int, default=128)
    design = parser.add_mutually_exclusive_group()
    design.add_argument("--design-calibration", type=Path,
        help="Score one Design 1 artifact instead of the fixed Q/H controls; also pass to aggregate")
    design.add_argument("--design-calibration-list", type=Path,
        help="JSON [{tag,calibration_dir}] for Design 1 policies sharing one native replay")
    parser.add_argument("--history-packs", type=Path,
        help="Optional checked bounded-history packs; also pass to aggregate")
    args = parser.parse_args()
    if min(args.unit_users, args.cohort_size, args.query_batch, args.num_shards) < 1:
        parser.error("batch, unit and shard counts must be positive")
    (score if args.command == "score" else aggregate)(args)
