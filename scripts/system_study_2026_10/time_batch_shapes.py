#!/usr/bin/env python3
"""CPU-only shape execution of the actual distinct-user frontier scheduler."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np

from system_study_2026_10.time_model import read, write, binding, DEFAULT_SHAPES
from query_relay_2026_10.workload import trigger_indices
from query_relay_2026_10.cost import read_flops


def groups(users, key, maximum):
    buckets = defaultdict(list)
    for uid in users:
        buckets[key(uid)].append(uid)
    for members in buckets.values():
        for start in range(0, len(members), maximum):
            yield members[start:start + maximum]


def simulate(histories, users, releases, end, known_vocab_size, *, maximum=8,
             cap=1024, band_cap=32, progress=False, window_users=None):
    """No weights: preserve run_batched's order, readiness and exact shape keys.

    The global trace's first-request order is (time, input-user order). Its
    frontier then takes one pending round per user in that order. All callbacks
    within each wave preserve this order, including the iterative append bands.
    """
    cursor = {u: int(np.searchsorted(histories[u][0], releases[0][1], side="left")) for u in users}
    length = {u: min(cap, cursor[u]) for u in users}
    coded = dict(length)
    current = {u: False for u in users}
    pending_rebuild = {u: True for u in users}
    selected = {u: trigger_indices(histories[u], stride=16, known_vocab_size=known_vocab_size,
                                  releases=releases, end=end) for u in users}
    # Per-user timestamp -> strict-past stop is fixed by the real event stream.
    targets = {u: {v: np.searchsorted(histories[u][0], histories[u][0][ix], side="left")
                   for v, ix in selected[u].items()} for u in users}
    batches, logical, transfers, window_transfers = Counter(), Counter(), Counter(), Counter()
    zero = Counter()
    version = releases[0][0]

    def record(operation, members, width=0):
        n, b = length[members[0]], len(members)
        codes = [coded[u] for u in members]
        maxcode = max(codes) if operation in ("score", "append") else n
        methods = ("rebuild",) if operation == "rebuild" else ("relay",) if operation in ("materialize", "anchorize") else ("relay", "rebuild")
        for method in methods:
            batches[(method, version, operation, n, width, b, maxcode if method == "relay" else 0)] += 1
        for u in members:
            count = coded[u] if operation in ("score", "append") else 0 if operation == "rebuild" else length[u]
            logical[(version, operation, length[u], width, count)] += 1
        if operation in ("score", "append"):
            extra = b * maxcode - sum(codes)
            zero["additional_zero_code_pairs"] += width * extra
            zero["additional_active_code_queries"] += width * sum(c == 0 for c in codes) if maxcode else 0
        if operation in ("anchorize", "materialize"):
            zero["conversion_valid_rows"] += b*n
            zero["conversion_padded_rows"] += ((b*n+15)//16)*16

    def prepare(members):
        pending = [u for u in members if not current[u]]
        for group in groups(pending, lambda u: length[u], maximum):
            record("materialize", group)
            for u in group:
                current[u], coded[u] = True, length[u]

    def advance(stops):
        while True:
            pending = [u for u, stop in stops.items() if cursor[u] < stop]
            if not pending:
                break
            prepare(pending)
            def key(u):
                width = 1 << (min(band_cap, int(stops[u]) - cursor[u]).bit_length() - 1)
                return length[u], width
            for group in groups(pending, key, maximum):
                n, width = key(group[0])
                record("append", group, width)
                for u in group:
                    coded[u] = max(0, coded[u] - max(0, n + width - cap))
                    length[u], cursor[u] = min(cap, n + width), cursor[u] + width

    def traffic(members, before, after, *, recommendation):
        # State paging is a conditional host policy, separate from the actual
        # resident compute scheduler. Directional groups never mix tensor N or
        # field masks. Appends+prepare+score reuse one page per consumer.
        for method in ("relay", "rebuild"):
            rows = {}
            for u in members:
                n, oldcursor, had_code, had_rebuild = before[u]
                changed = cursor[u] > oldcursor
                rebuilt = method == "rebuild" and recommendation and had_rebuild
                active = recommendation or changed or (method == "relay" and after and current[u])
                if not active:
                    continue
                load_native = not (rebuilt and not changed)
                load_codes = method == "relay"
                write_native = changed or rebuilt
                write_codes = method == "relay" and (changed or (recommendation and not had_code) or after)
                rows[u] = (n, length[u], load_native, load_codes, write_native, write_codes)
            for direction, ni, fi in (("h2d", 0, 2), ("d2h", 1, 4)):
                active = [u for u, r in rows.items() if r[fi] or r[fi+1]]
                for group in groups(active, lambda u: (rows[u][ni], rows[u][fi], rows[u][fi+1]), maximum):
                    n, native, code = rows[group[0]][ni], rows[group[0]][fi], rows[group[0]][fi+1]
                    transfers[(method, version, direction, n, native, code, len(group))] += 1

    for i, (version, at) in enumerate(releases):
        current = {u: False for u in users}
        pending_rebuild = {u: True for u in users}
        coded = dict(length)
        until = releases[i+1][1] if i+1 < len(releases) else end
        has_next = i+1 < len(releases)
        window_size = window_users or len(users)
        waves = 0
        for start in range(0,len(users),window_size):
            window = users[start:start+window_size]
            def window_pages(direction):
                for method in ("relay", "rebuild"):
                    for group in groups(window,lambda u:length[u],maximum):
                        window_transfers[(method,version,direction,length[group[0]],True,method=="relay",len(group))] += 1
            window_pages("h2d")
            active = [u for u in window if len(selected[u][version])]
            # Python sort is stable: tie order equals trace insertion/user order.
            active.sort(key=lambda u: int(histories[u][0][selected[u][version][0]]))
            wave = 0
            while active:
                before = {u: (length[u], cursor[u], current[u], pending_rebuild[u]) for u in active}
                advance({u: targets[u][version][wave] for u in active})
                prepare(active)
                for group in groups([u for u in active if pending_rebuild[u]], lambda u: length[u], maximum):
                    record("rebuild", group)
                    for u in group:
                        pending_rebuild[u] = False
                for group in groups(active, lambda u: length[u], maximum):
                    record("score", group, 16)
                traffic(active, before, False, recommendation=True)
                wave += 1
                active = [u for u in active if len(selected[u][version]) > wave]
            waves += wave
            before = {u: (length[u], cursor[u], current[u], pending_rebuild[u]) for u in window}
            advance({u: int(np.searchsorted(histories[u][0], until, side="left")) for u in window})
            traffic(window, before, has_next, recommendation=False)
            if has_next:
                for group in groups([u for u in window if current[u]], lambda u: length[u], maximum):
                    record("anchorize", group)
            window_pages("d2h")
            zero["max_window_native_and_code_bytes"] = max(zero["max_window_native_and_code_bytes"],sum(length[u] for u in window)*10368)
        if progress:
            print({"status": "CPU_frontier_counted", "version": version, "waves": waves,
                   "aggregate_batch_shapes": len(batches)}, flush=True)
    work = read_flops(pairs=zero["additional_zero_code_pairs"], queries=zero["additional_active_code_queries"],
        head_rank=6, layers=6, heads=6, head_dim=32, gates=True, gate_normalization="rms")
    zero["additional_coalescing_flops"] = work["total"] - 5*6*(4*32+2)*zero["additional_active_code_queries"]
    zero["conversion_padding_extra_dot_flops"] = 6*(31744*zero["conversion_padded_rows"]-15264*zero["conversion_valid_rows"])
    return batches, logical, transfers, zero, window_transfers


def canonical_individual(shapes):
    result = Counter()
    for r in shapes:
        result[(r.get("parameter_version", r["version"]), r["operation"], r["history_length"], r["width"], r["coded_rows"])] += r["calls"]
    return result


def first_query_redundancy(histories, users, releases, end, known_vocab_size):
    """Only append work immediately followed by an existing first-query Rebuild."""
    from query_relay_2026_10.workload import _native_totals
    counts = {v:Counter() for v,_ in releases}
    for uid in users:
        events = histories[uid]
        selected = trigger_indices(events,stride=16,known_vocab_size=known_vocab_size,releases=releases,end=end)
        for version,at in releases:
            if not len(selected[version]):
                counts[version]["users_without_query_excluded"] += 1
                continue
            cursor = int(np.searchsorted(events[0],at,side="left"))
            first_query_at = int(events[0][selected[version][0]])
            stop = int(np.searchsorted(events[0],first_query_at,side="left"))
            counts[version]["users_with_first_query"] += 1
            if stop == cursor:
                continue
            c = counts[version]
            c["users_with_overwritten_append"] += 1
            n = min(cursor,1024)
            while cursor < stop:
                width = 1 << (min(32,stop-cursor).bit_length()-1)
                c["tokens"] += width
                c["append_user_operations"] += 1
                c[f"native_append_shape_{n}:{width}"] += 1
                cursor += width
                n = min(1024,n+width)
    result = {}
    for version,c in counts.items():
        result[version] = dict(counts=dict(c),native_append_flops=_native_totals(c)["native_append_flops"])
    total = sum(r["native_append_flops"] for r in result.values())
    old = read(ROOT/"results/system_study_2026_10/cost/scaling/attempt1/summary.json")["branches"]["current"]["decomposition"]
    return dict(status="complete_CPU_first_query_overwrite_sensitivity",users=len(users),by_release=result,
        removable_native_append_mathematical_flops=total,
        percent_of_recorded_common_native_service=100*total/old["common_native_at_reference_flops"],
        percent_of_recorded_full_rebuild_service=100*total/(old["common_native_at_reference_flops"]+old["rebuild_at_reference_flops"]),
        common_native_after_skip_flops=old["common_native_at_reference_flops"]-total,
        original_rebuild_flops_unchanged=old["rebuild_at_reference_flops"],
        scope="Strictly causal first-query-only sensitivity: Rebuild can advance item/event metadata to its existing first query and reconstruct once, without performing the immediately overwritten incremental K/V writes. Users without a query excluded. Actual fair128 scheduling is unchanged; no GPU or timing benefit measured. Native service decreases by delta; method work is unchanged, so its added cost over common native increases by delta.")


def run(output, *, threads=4, maximum=8, selection=None, session=None, window_users=None,redundancy_only=False):
    import pyarrow.parquet as pq
    from design_one.prepare_history import load_histories_semijoin
    from design_two.common import RELEASES, END
    from query_relay_2026_10.cost import CONFIG, PANEL
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    config, panel = read(CONFIG), read(PANEL)
    dataset_path = ROOT / config["data"]["dataset"]["path"]
    population_path = ROOT / config["data"]["population"]["path"]
    dataset = read(dataset_path)
    users = read(selection)["selected_uids"] if selection else sorted(pq.read_table(population_path, columns=["uid"])["uid"].to_pylist())
    print({"status": "loading_CPU_histories", "users": len(users)}, flush=True)
    histories = load_histories_semijoin(users, dataset_path=dataset_path,
        known_vocab_size=dataset["foundation_items"], oov_buckets=panel["model_config"]["num_items"]-dataset["foundation_items"],
        start_timestamp=RELEASES[0][1], end_timestamp=END, max_history=1024, threads=threads).rows
    histories = {u: histories.get(u, (np.array([], dtype=np.int64), np.array([], dtype=np.int64))) for u in users}
    if redundancy_only:
        result = first_query_redundancy(histories,users,RELEASES,END,dataset["foundation_items"])
        result["source_bindings"] = [binding(p) for p in (CONFIG,PANEL,dataset_path,population_path,Path(__file__))]
        result["elapsed_CPU_seconds"] = time.perf_counter()-started
        write(output/"summary.json",result)
        (output/"summary.md").write_text(f"# 首次Rebuild覆盖的追加工作敏感性\n\n仅数有真实首请求的用户，从发布到该query前的严格历史追加；不把无query用户或所有first-writer视为可省。\n\n可被一次重建覆盖的原生追加={result['removable_native_append_mathematical_flops']/1e12:.6f}TF，占原完整Rebuild服务{result['percent_of_recorded_full_rebuild_service']:.6f}%。这是CPU数学账敏感性，未改当前fair128执行或计时。跳过后原生共同服务减少此量，方法工作不变，所以相对共同服务的额外方法开销相应增加。\n")
        print({k:v for k,v in result.items() if k not in ("by_release","source_bindings")},flush=True)
        return
    batches, logical, transfers, extra, window_transfers = simulate(histories, users, RELEASES, END, dataset["foundation_items"], maximum=maximum, progress=True,window_users=window_users)
    checks = {}
    if selection is None:
        expected = canonical_individual(read(DEFAULT_SHAPES)["shapes"])
        if expected != logical:
            bad = [(k, expected[k], logical[k]) for k in expected.keys() | logical.keys() if expected[k] != logical[k]]
            raise AssertionError(("individual ledger mismatch", bad[:10], len(bad)))
        checks["all_individual_joint_shapes"] = "exact_match_30k_retained_ledger"
    if session:
        measured = Counter()
        for method in ("relay", "rebuild"):
            for r in read(session / f"{method}_calls.json"):
                coded = r["coded_rows"] if method == "relay" and r["operation"] in ("append", "score") else r["history_length"] if method == "relay" else 0
                measured[(method, r["version"], r["operation"], r["history_length"], r["query_width"], r["batch"], coded)] += 1
        if measured != batches:
            bad = [(k, measured[k], batches[k]) for k in measured.keys() | batches.keys() if measured[k] != batches[k]]
            raise AssertionError(("actual batch scheduler mismatch", bad[:10], len(bad)))
        checks["actual_distinct_user_session_batch_shapes"] = "exact_match"
    rows = [dict(method=m, version=v, operation=o, history_length=n, width=w, batch=b, coded_rows=c, calls=k)
            for (m,v,o,n,w,b,c),k in sorted(batches.items())]
    traffic = [dict(method=m,version=v,direction=d,history_length=n,native=na,code=co,batch=b,calls=k,
                    payload_bytes=k*b*n*(9216*na+1152*co))
               for (m,v,d,n,na,co,b),k in sorted(transfers.items())]
    window_traffic = [dict(method=m,version=v,direction=d,history_length=n,native=na,code=co,batch=b,calls=k,
                    payload_bytes=k*b*n*(9216*na+1152*co))
               for (m,v,d,n,na,co,b),k in sorted(window_transfers.items())]
    batch_mix = Counter()
    for r in rows:
        batch_mix[(r["method"],r["operation"],r["batch"])] += r["calls"]
    result = dict(status="complete_CPU_frontier_batch_shapes", users=len(users), max_batch=maximum,
        batches=rows, host_directional_batches=traffic, window_directional_batches=window_traffic,
        window_users=window_users, execution_extra=dict(extra), checks=checks,
        batch_mix=[dict(method=m,operation=o,batch=b,calls=c,user_operations=b*c) for (m,o,b),c in sorted(batch_mix.items())],
        elapsed_CPU_seconds=time.perf_counter()-started,
        scope="Exact CPU shape replay of same per-user frontier/append-band/N-grouped coalescing scheduler; no model execution or arrival queue. Batch membership and padding come from the real population, not scaled small-sample means.",
        host_scope="window_directional_batches conservatively loads and writes complete native+code state for every UID on every release, including idle users, initial input and final drain output. Max transfer batch8; resident window bound explicit. Other host_directional_batches describe an unimplemented request-local potential. No overlap or online arrival latency claim.",
        source_bindings=[binding(p) for p in (CONFIG,PANEL,dataset_path,population_path,Path(__file__),
            ROOT/"scripts/system_study_2026_10/batched_session.py",ROOT/"scripts/system_study_2026_10/batched_session_probe.py")])
    if selection:
        result["source_bindings"].append(binding(selection))
    write(output/"summary.json",result)
    (output/"summary.md").write_text(f"# CPU真实frontier批次计数\n\n{len(users):,}用户；驻留UID窗口{window_users or '全人口'}；最大批次{maximum}；{len(rows):,}种批联合形状。检查：{checks}。\n\n零尾合批额外{extra['additional_coalescing_flops']/1e12:.6f} TF；转换tile补齐额外{extra['conversion_padding_extra_dot_flops']/1e12:.6f} TF。均为显式逻辑工作，不是机器指令计数。\n\n调度器只重排独立用户，不跨发布；每轮每用户一个最早待处理请求。每release按固定UID窗口完成请求、尾部drain、码交接后卸载，含空闲用户与初末全状态迁移。峰值窗口原生KV+code={extra['max_window_native_and_code_bytes']/1e9:.6f}GB（模型、目录、工作区另计）。此为离线/饱和bulk调度形状，并非按真实到达提供在线时延保证。\n")
    print({"status": result["status"], "output": str(output), "checks": checks, "extra": dict(extra)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--maximum", type=int, default=8)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--window-users",type=int)
    parser.add_argument("--first-query-redundancy",action="store_true")
    args = parser.parse_args()
    run(args.output, threads=args.threads, maximum=args.maximum, selection=args.selection, session=args.session,window_users=args.window_users,redundancy_only=args.first_query_redundancy)
