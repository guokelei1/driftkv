#!/usr/bin/env python3
"""Count causal cache lengths for the already fixed 30,000-user cost population.

This reads UID/timestamp metadata only, including reserved users' counts.
It does not read feedback, model outputs, or confirmation quality.
"""

import hashlib
import json
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT/"data/processed/yambda500m_unified_v1/scales/medium/dataset.json"
DAYS = (231,245,259,273,287)


def sql_path(path):
    return str(path.resolve()).replace("'","''")


def main():
    start = time.perf_counter()
    dataset = json.loads(DATASET.read_text())
    users = DATASET.parent/"users.parquet"
    listens = DATASET.parent/dataset["shared_listens_glob"]
    counts = ",".join(f"count(l.timestamp) FILTER (WHERE l.timestamp < {day*86400}) AS n{i}"
                      for i,day in enumerate(DAYS,1))
    con = duckdb.connect()
    con.execute("SET threads=8")
    frame = con.execute(f"""
        SELECT u.uid,{counts}
        FROM read_parquet('{sql_path(users)}') u
        LEFT JOIN read_parquet('{sql_path(listens)}',hive_partitioning=true) l
          ON u.uid=l.uid AND l.timestamp < {DAYS[-1]*86400}
        GROUP BY u.uid
    """).fetchdf()
    con.close()
    assert len(frame)==30000
    rows=[]
    for target,day in enumerate(DAYS,1):
        lengths=frame[f"n{target}"].clip(upper=1024)
        rows.append(dict(target=target,cutover_day=day,users=len(frame),
            total_retained_events=int(lengths.sum()),mean_cache_length=float(lengths.mean()),
            short_cache_users=int((lengths<1024).sum()),empty_cache_users=int((lengths==0).sum()),
            exact_length_counts={str(int(k)):int(v) for k,v in lengths.value_counts().sort_index().items()}))
    result=dict(scope="label-free UID/timestamp metadata for fixed population; includes reserved UID counts, no model/quality access",
        confirmation_target_outputs_read=False,estimated_seconds_range=[30,120],
        inputs_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (DATASET,users)},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows,
        elapsed_seconds=time.perf_counter()-start,
        limitation="history-weighted denominator inputs only; no compute or 20-percent qualification")
    out=ROOT/"results/design/analysis/population_history_cost_probe.json"
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(elapsed_seconds=result["elapsed_seconds"],rows=[{k:v for k,v in r.items() if k!="exact_length_counts"} for r in rows])))


if __name__=="__main__":
    main()
