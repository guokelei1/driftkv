#!/usr/bin/env python3
"""Compare serial and batched handoff of distinct retained real user states.

The source contains V4 current codes. V4 inverse conversion produces anchors;
V5 forward conversion then produces V5 current codes. One persistent compact
allocation supplies both B=1 user views and B=8. Input reset, model/table load,
reference checks and profiler overhead are outside measured conversion calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import torch
from hstu_kvcache.design_one.lightweight import LightweightValueState
from hstu_kvcache.models import HSTUKVCache
from implementation_2026_10.prepare import fixed_features_from_bank
from implementation_2026_10.state import InplaceReleasePrefix
from query_relay_2026_10.adapter import UnifiedRelayAdapter
from query_relay_2026_10.fixed_features import load_bank
from query_relay_2026_10.precision import load_stable_prefix
from unified_reuse_2026_09.common import sha256, write_json


@torch.no_grad()
def run(args):
    config = json.loads(args.config.read_text())
    if args.output.exists():
        raise FileExistsError("use a fresh attempt directory")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    branch = ROOT / config["branch"]
    source_path = branch / config["source"]
    target_meta = json.loads((branch / "calibration/v5/calibration.json").read_text())
    pack = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
    eligible = sorted(int(uid) for uid, row in pack["rows"].items()
                      if uid in target_meta["fit_uids"] and row["parent"].seq_len == config["history_length"])
    uids = eligible[:config["users"]]
    if len(uids) != config["users"]:
        raise ValueError(f"only {len(uids)} real full-history fit users, no repeated-row substitute")
    source_hash = sha256(source_path)
    if source_hash != target_meta["source_pack_sha256"]:
        raise ValueError("source differs from retained V5 calibration binding")
    bindings = {str(source_path.relative_to(ROOT)): source_hash,
                str(args.config.relative_to(ROOT)): sha256(args.config),
                str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__))}
    prefixes, adapters, metadata = {}, {}, {}
    for version in (config["source_current_version"], config["target_version"]):
        checkpoint = branch / "calibration" / version / "calibration.pt"
        publication = branch / "publication" / f"{version}.pt"
        digest = sha256(checkpoint)
        if version == config["source_current_version"] and digest != pack["prior_checkpoints"][version]["sha256"]:
            raise ValueError("V4 inverse prefix does not match source previous_codes")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        adapter = UnifiedRelayAdapter(**payload["config"]).to(device).eval().requires_grad_(False)
        adapter.load_state_dict(payload["state_dict"])
        prefix, record = load_stable_prefix(publication, adapter, device=device)
        if record["bindings"]["calibration_sha256"] != digest:
            raise ValueError("publication and calibration differ")
        prefixes[version], adapters[version], metadata[version] = prefix, adapter, record
        bindings[str(checkpoint.relative_to(ROOT))] = digest
        bindings[str(publication.relative_to(ROOT))] = sha256(publication)
    before_version, after_version = config["source_current_version"], config["target_version"]
    old, new = prefixes[before_version], prefixes[after_version]
    if adapters[before_version].feature_bank_sha256 != adapters[after_version].feature_bank_sha256:
        raise ValueError("handoff changed the immutable coordinate encoder")
    adapter = adapters[after_version]
    bank = load_bank(adapter.feature_bank_path, adapter.feature_bank_sha256,
                     device=device, dtype=torch.float32)
    features = partial(fixed_features_from_bank, bank=bank)
    backend = config.get("conversion_backend", "torch")
    if backend == "triton":
        from system_study_2026_10.fused_conversion import FusedReleasePrefix
        old_fast, new_fast = (FusedReleasePrefix.wrap(prefix, bank=bank) for prefix in (old, new))
        fused_source = ROOT / "scripts/system_study_2026_10/fused_conversion.py"
        bindings[str(fused_source.relative_to(ROOT))] = sha256(fused_source)
    else:
        old_fast = InplaceReleasePrefix.wrap(old, row_chunk=config["row_chunk"], feature_fn=features)
        new_fast = InplaceReleasePrefix.wrap(new, row_chunk=config["row_chunk"], feature_fn=features)
    rows = [pack["rows"][uid] for uid in uids]
    length = config["history_length"]
    cache = HSTUKVCache(torch.cat([r["parent"].k for r in rows], 1).to(device),
                       torch.cat([r["parent"].v for r in rows], 1).to(device), length)
    ids = torch.cat([r["item_ids"] for r in rows], 0).to(device)
    previous = torch.cat([r["previous_codes"] for r in rows], 1).to(device)
    native_copy = (cache.k.clone(), cache.v.clone())
    state = LightweightValueState(previous.clone(), length)
    pointer = state.codes.data_ptr()
    references = {"anchorize_v4": [], "materialize_v5": []}
    for user in range(len(uids)):
        one = HSTUKVCache(cache.k[:, user:user+1], cache.v[:, user:user+1], length)
        anchor = old.anchorize(one, LightweightValueState(previous[:, user:user+1], length), ids[user:user+1])
        current = new.materialize(one, anchor, ids[user:user+1])
        references["anchorize_v4"].append(anchor.codes)
        references["materialize_v5"].append(current.codes)
    references = {key: torch.cat(value, 1) for key, value in references.items()}
    references["handoff_v4_to_v5"] = references["materialize_v5"]
    actions = ("anchorize_v4", "materialize_v5", "handoff_v4_to_v5")
    inputs = {action: references["anchorize_v4"] if action == "materialize_v5" else previous
              for action in actions}
    contexts = {}
    for batch in config["batch_sizes"]:
        contexts[batch] = [(HSTUKVCache(cache.k[:, start:start+batch], cache.v[:, start:start+batch], length),
                            LightweightValueState(state.codes[:, start:start+batch], length),
                            ids[start:start+batch]) for start in range(0, len(uids), batch)]

    def invoke(action, batch):
        for one, codes, items in contexts[batch]:
            if action != "materialize_v5":
                assert old_fast.anchorize(one, codes, items) is codes
            if action != "anchorize_v4":
                assert new_fast.materialize(one, codes, items) is codes

    def reset(action):
        state.codes.copy_(inputs[action])
        torch.cuda.synchronize(device)

    args.output.mkdir(parents=True)
    write_json(args.output / "configuration.json", dict(config=config, uids=uids,
        eligible_full_history_fit_users=len(eligible), created_utc=datetime.now(timezone.utc).isoformat(),
        command=sys.argv, bindings=bindings, feature_bank_sha256=adapter.feature_bank_sha256,
        semantics="source previous_codes are V4 current; V4 anchorize -> immutable anchors -> V5 materialize",
        persistent_code_layout=list(state.codes.shape), no_training=True, no_new_users_or_labels=True))
    print(json.dumps(dict(status="ready", uids=uids, batch_sizes=config["batch_sizes"],
        actions=actions, row_chunk=config["row_chunk"])), flush=True)
    records = {}
    for action in actions:
        records[action] = {batch: dict(samples=[], validation={}, memory={}) for batch in config["batch_sizes"]}
        for batch in config["batch_sizes"]:
            for _ in range(config["warmup"]):
                reset(action); invoke(action, batch)
            torch.cuda.synchronize(device)
            errors = []
            for user, uid in enumerate(uids):
                actual = state.codes[:, user:user+1]
                expected = references[action][:, user:user+1]
                torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
                errors.append(dict(uid=uid, max_abs=float((actual-expected).abs().max())))
            records[action][batch]["validation"] = dict(per_user=errors,
                max_abs=max(r["max_abs"] for r in errors), atol=3e-5, rtol=3e-5)
        # Alternate which batch shape is timed first; resetting the same input
        # and all correctness/profiling work stay outside the timed region.
        for repeat in range(config["repetitions"]):
            order = config["batch_sizes"][::1 if repeat % 2 == 0 else -1]
            for batch in order:
                reset(action)
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin = time.perf_counter()
                start.record()
                invoke(action, batch)
                stop.record()
                stop.synchronize()
                wall_ms = (time.perf_counter()-begin)*1000
                records[action][batch]["samples"].append(dict(repeat=repeat,
                    wall_ms=wall_ms, cuda_event_ms=start.elapsed_time(stop)))
        for batch in config["batch_sizes"]:
            reset(action)
            torch.cuda.reset_peak_memory_stats(device)
            allocated = torch.cuda.memory_allocated(device)
            invoke(action, batch)
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)-allocated
            records[action][batch]["memory"] = dict(peak_extra_allocated_bytes=peak,
                extra_persistent_code_bytes=0, same_persistent_storage=state.codes.data_ptr() == pointer)
            reset(action)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as profiler:
                invoke(action, batch)
                torch.cuda.synchronize(device)
            gpu_events = [event for event in profiler.events() if event.device_type == torch.autograd.DeviceType.CUDA]
            records[action][batch]["launch_profile"] = dict(cuda_device_events=len(gpu_events),
                event_names=dict(Counter(event.name for event in gpu_events)),
                scope="One separate untimed profile; CUDA device events may include copies/memsets")
            samples = records[action][batch]["samples"]
            records[action][batch].update(median_wall_ms=statistics.median(x["wall_ms"] for x in samples),
                median_cuda_event_ms=statistics.median(x["cuda_event_ms"] for x in samples),
                api_conversion_calls=(len(contexts[batch])*(2 if action.startswith("handoff") else 1)))
            records[action][batch]["median_wall_ms_per_user"] = records[action][batch]["median_wall_ms"]/len(uids)
            torch.testing.assert_close(state.codes, references[action], rtol=3e-5, atol=3e-5)
    assert state.codes.data_ptr() == pointer
    torch.testing.assert_close(cache.k, native_copy[0], rtol=0, atol=0)
    torch.testing.assert_close(cache.v, native_copy[1], rtol=0, atol=0)
    ratios = {action: dict(serial8_over_batch8_wall=records[action][1]["median_wall_ms"]/records[action][8]["median_wall_ms"],
        batch8_over_serial8_wall=records[action][8]["median_wall_ms"]/records[action][1]["median_wall_ms"],
        batch8_over_serial8_cuda_events=records[action][8]["launch_profile"]["cuda_device_events"]/records[action][1]["launch_profile"]["cuda_device_events"])
        for action in actions}
    result = dict(status="complete", uids=uids, history_length=length, records=records, ratios=ratios,
        conversion_backend=backend,
        conversion_precision="FP32 state; TF32x3 native/compact dots" if backend == "triton" else "FP32; Torch TF32 disabled",
        kernel_geometry=new_fast.geometry(batch=len(uids), rows=length) if backend == "triton" else None,
        native_kv_unchanged=True, same_persistent_code_storage=True,
        persistent_native_kv_bytes=2*cache.k.numel()*cache.k.element_size(),
        persistent_codes_bytes=state.codes.numel()*state.codes.element_size(),
        resident_shared_publication_bytes=sum(prefix.storage_bytes() for prefix in prefixes.values()),
        immutable_feature_bank_bytes=bank.storage_bytes(),
        device=dict(name=torch.cuda.get_device_name(device), torch=torch.__version__, cuda=torch.version.cuda),
        interpretation="Eight distinct retained users, identical arithmetic and single code allocation. Batching amortizes host/API/kernel launches but is not full lifecycle C or a loaded throughput/p99 result. Warm tables, no input reset/preparation/profiler cost in timing; no row-feature cache.")
    write_json(args.output / "summary.json", result)
    lines = ["# 八个真实用户的码交接批处理", "",
        "源为 V5 发布前的 V4 current codes：先以 V4 prefix 反锚，再以 V5 prefix 物化。8人均为已有 fit 源、真实1024行，不平铺用户。",
        "B=1 将同8人逐个处理；B=8 合批处理。均使用 prepared feature bank、3次暖机及5次交错重复。",
        f"后端：{backend}；精度：{result['conversion_precision']}。", "",
        "| 算子 | B1处理8人中位ms | B8处理8人中位ms | B8每用户ms | serial/batch | 临时空间B1/B8 MiB | CUDA事件B1/B8 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for action in actions:
        a,b=records[action][1],records[action][8]
        lines.append(f"| {action} | {a['median_wall_ms']:.3f} | {b['median_wall_ms']:.3f} | {b['median_wall_ms_per_user']:.3f} | {ratios[action]['serial8_over_batch8_wall']:.3f} | {a['memory']['peak_extra_allocated_bytes']/2**20:.3f}/{b['memory']['peak_extra_allocated_bytes']/2**20:.3f} | {a['launch_profile']['cuda_device_events']}/{b['launch_profile']['cuda_device_events']} |")
    lines.extend(["", "逐用户与未改 stable reference 比较通过；原生K/V逐元素不变，所有转换使用同一份持久码存储。临时空间不含常驻表、原生状态与测量用输入/参考副本。",
        "这只测量 C 中的码转换子项：不包含共享准备S、历史追赶、推荐、排队或数据搬入。不能据此声称完整S+N×C已验证或生产吞吐提升。CUDA事件数来自另一次不计时的profile，包含其列明的设备事件。", ""])
    lines.extend(["## 五次测量的波动", ""])
    for action in actions:
        ranges = []
        for batch in config["batch_sizes"]:
            values = [x["wall_ms"] for x in records[action][batch]["samples"]]
            ranges.append(f"B{batch} {min(values):.2f}–{max(values):.2f} ms")
        lines.append(f"- {action}：{'；'.join(ranges)}。")
    lines.extend(["", "两阶段交接按每用户先反锚再物化，独立阶段则分别遍历8人；两个独立中位数之和不等于组合测量的中位数。只测5次，不将单个比值视为稳定服务加速。",
        "API计时包含Python调度；CUDA event区间也可能包含主机提交之间的设备空闲，不能当成纯GPU忙时。", ""])
    if backend == "triton":
        lines.extend(["融合核要求上游已验证物品ID范围，避免每层GPU标量同步。持久行码不增，但另有发布级打包权重及矩阵；其大小和tile补齐dot工作见JSON。", ""])
    (args.output / "summary.md").write_text("\n".join(lines))
    print(json.dumps(dict(status="complete", output=str(args.output), ratios=ratios)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/system_study_2026_10/batched_conversion.json")
    parser.add_argument("--output", type=Path, required=True)
    arguments=parser.parse_args()
    arguments.config=arguments.config.resolve()
    run(arguments)
