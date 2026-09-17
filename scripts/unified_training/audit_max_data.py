#!/usr/bin/env python3
"""Validate the prepared Max population, IDs, requests and real histories."""

import json, hashlib, time
from pathlib import Path
import duckdb, pyarrow.parquet as pq, numpy as np
from hstu_kvcache.data.scale_population import uid_selector_digest
from hstu_kvcache.data.yambda_history import load_yambda_histories
from hstu_kvcache.data.oov import apply_stable_oov_buckets

root = Path(__file__).resolve().parents[2]
base = root / "data/processed/yambda5b_max_200k_v1"
scale = base / "scales/max"
out = root / "results/unified_training_2026_09/max/preparation"
c = duckdb.connect()
c.execute("SET threads=32")
c.execute("SET memory_limit='128GB'")


def path(p):
    return str(p)


def query(q):
    return c.execute(q).fetchone()[0]


u = scale / "users.parquet"
m = scale / "item_mapping.parquet"
raw = root / "data/raw/yambda/flat/5b"
gl = base / "shared/listens/**/*.parquet"
fb = base / "shared/feedback/**/*.parquet"
cut = 217 * 86400
selected = pq.read_table(u).to_pandas()
eligible = c.execute(
    f"SELECT DISTINCT uid FROM read_parquet('{raw / 'listens.parquet'}') WHERE timestamp<{cut}"
).fetchnumpy()["uid"]
ranked = sorted(map(int, eligible), key=lambda x: (uid_selector_digest(x), x))[:200000]
assert selected.sort_values("selector_rank").uid.tolist() == ranked
checks = {
    "eligible_users": len(eligible),
    "selected_users": len(selected),
    "population_exact_hash_order": True,
}
print("population checked", flush=True)
maptable = pq.read_table(m).to_pandas()
assert np.array_equal(maptable.item_idx, np.arange(1, len(maptable) + 1))
assert (np.diff(maptable.raw_item_id.astype(np.int64)) > 0).all()
c.execute(
    f"CREATE TEMP TABLE ref_items AS SELECT DISTINCT e.item_id::UBIGINT AS raw_item_id FROM read_parquet('{raw / 'listens.parquet'}') e JOIN read_parquet('{u}') u USING(uid) WHERE e.timestamp<{cut}"
)
assert (
    query(
        f"SELECT count(*) FROM (SELECT raw_item_id FROM ref_items EXCEPT SELECT raw_item_id FROM read_parquet('{m}'))"
    )
    == 0
)
assert (
    query(
        f"SELECT count(*) FROM (SELECT raw_item_id FROM read_parquet('{m}') EXCEPT SELECT raw_item_id FROM ref_items)"
    )
    == 0
)
checks["mapping_exact_raw_prefix_match"] = True
checks["known_items"] = len(maptable)
print("mapping checked", flush=True)
counts = {}
for name in ["listens", "likes", "dislikes"]:
    n = query(
        f"SELECT count(*) FROM read_parquet('{raw / (name + '.parquet')}') e JOIN read_parquet('{u}') u USING(uid)"
    )
    counts[name] = n
actual_listens = query(f"SELECT count(*) FROM read_parquet('{gl}',hive_partitioning=true)")
assert actual_listens == counts["listens"]
actual_feedback = c.execute(
    f"SELECT label,count(*) FROM read_parquet('{fb}',hive_partitioning=true) GROUP BY label"
).fetchall()
assert dict(actual_feedback) == {1: counts["likes"], 0: counts["dislikes"]}
checks["complete_selected_raw_rows"] = counts
checks["population_overlap_with_Large"] = len(
    set(selected.uid)
    & set(
        pq.read_table(
            root / "data/processed/yambda500m_unified_v1/scales/large/users.parquet",
            columns=["uid"],
        )["uid"].to_pylist()
    )
)
q = root / "data/manifests/yambda5b_max_200k_hstu_native_v1/requests_quality.parquet"
f = q.with_name("requests_fidelity.parquet")
assert "label" not in pq.read_schema(f).names
assert pq.read_metadata(q).num_rows == pq.read_metadata(f).num_rows
assert query(f"SELECT count(*)-count(DISTINCT request_id) FROM read_parquet('{q}')") == 0
assert (
    query(
        f"SELECT count(*) FROM read_parquet('{q}') q LEFT JOIN read_parquet('{u}') u USING(uid) LEFT JOIN read_parquet('{m}') m USING(raw_item_id) WHERE u.uid IS NULL OR q.query_timestamp<=u.first_timestamp OR q.item_idx!=coalesce(m.item_idx,0) OR q.target_known!=(m.item_idx IS NOT NULL) OR q.label NOT IN(0,1)"
    )
    == 0
)
checks["request_unique_causal_ID_and_label_audit"] = True
windows = {
    "v0": [0, 217],
    "v1": [217, 231],
    "v2": [231, 245],
    "v3": [245, 259],
    "v4": [259, 273],
    "v5": [273, 287],
    "E14_v5": [287, 301],
}
stats = {}
for name, (a, b) in windows.items():
    r = c.execute(
        f"SELECT count(*),count(*) FILTER(WHERE target_known),count(DISTINCT uid) FILTER(WHERE target_known),count(*) FILTER(WHERE NOT target_known) FROM read_parquet('{q}') WHERE query_timestamp>={a * 86400} AND query_timestamp<{b * 86400}"
    ).fetchone()
    stats[name] = {
        "days": [a, b],
        "requests": r[0],
        "known_requests": r[1],
        "known_users": r[2],
        "oov_target_requests": r[3],
        "oov_target_fraction": r[3] / max(1, r[0]),
    }
checks["windows"] = stats
print("requests checked", flush=True)
# A small real-data reference checks strict-prefix truncation and disjoint OOV IDs.
qs = c.execute(
    f"SELECT uid,query_timestamp FROM read_parquet('{q}') WHERE target_known AND query_timestamp>={287 * 86400} ORDER BY query_timestamp,uid LIMIT 8"
).fetchall()
uids = sorted(set(r[0] for r in qs))
ds = scale / "dataset.json"
k = len(maptable)
h = load_yambda_histories(
    ds,
    uids,
    known_vocab_size=k,
    oov_buckets=256,
    start_timestamp=287 * 86400,
    end_timestamp=max(r[1] for r in qs) + 1,
    max_pre_events=1024,
    threads=14,
)
for uid, ts in qs:
    ref = c.execute(
        f"SELECT timestamp,l.raw_item_id,coalesce(m.item_idx,0),behavior FROM read_parquet('{gl}',hive_partitioning=true) l LEFT JOIN read_parquet('{m}') m USING(raw_item_id) WHERE uid={uid} AND timestamp<{ts} ORDER BY timestamp DESC,l.raw_item_id DESC,behavior DESC LIMIT 1024"
    ).fetchall()[::-1]
    expected = apply_stable_oov_buckets(
        np.array([x[1] for x in ref]),
        np.array([x[2] for x in ref]),
        known_vocab_size=k,
        buckets=256,
        bucket_start=k + 1,
    )
    items, behaviors, timestamps = h.prefix(uid, ts, max_history=1024)
    assert (
        np.array_equal(items, expected)
        and np.array_equal(timestamps, [x[0] for x in ref])
        and np.array_equal(behaviors, [x[3] for x in ref])
    )
checks["real_history_reference_requests"] = len(qs)
checks["status"] = "passed"
(out / "data_audit.json").write_text(json.dumps(checks, indent=2) + "\n")
print(json.dumps({"eligible": len(eligible), "counts": counts, "windows": stats}, indent=2))
