#!/usr/bin/env python3
"""Fast parallel request manifest builder for the HSTU-native recipe matrix.

Unlike the retired Base+residual builder, this never reconstructs popularity,
recency or Base features.  DuckDB performs the request collapse, known-item
join and strict-prior-listen qualification in parallel.  HSTU histories remain
loaded from the canonical chronological listen source at training/evaluation.
"""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

import duckdb
import pyarrow.parquet as pq
import yaml

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/"configs/contracts/yambda500m_small_hstu_native_rolling_recipe_matrix_v2.yaml"
OUTPUT=ROOT/"data/manifests/yambda500m_small_hstu_native_rolling_matrix_fast_v2"

def digest(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--contract",type=Path,default=CONTRACT);p.add_argument("--output",type=Path,default=OUTPUT);p.add_argument("--threads",type=int,default=24);a=p.parse_args()
 a.contract=a.contract.resolve(); a.output=a.output.resolve()
 c=yaml.safe_load(a.contract.read_text()); f=c["frozen_inputs"]
 for k in ("dataset_manifest","item_mapping"):
  if digest(ROOT/f[k])!=f[k+"_sha256"]:raise RuntimeError(f"matrix input hash mismatch: {k}")
 if a.output.exists():raise FileExistsError(f"refusing to overwrite {a.output}")
 d=json.loads((ROOT/f["dataset_manifest"]).read_text());root=(ROOT/f["dataset_manifest"]).parent
 windows=c["windows_days_half_open"]
 start_day=int(windows["foundation_end_day"]); end_day=int(windows["maximum_timestamp_exclusive_day"])
 feedback=(root/d["shared_feedback_glob"]).resolve(); listens=(root/d["shared_listens_glob"]).resolve(); mapping=(ROOT/f["item_mapping"]).resolve();start=start_day*86400;end=end_day*86400
 a.output.mkdir(parents=True); quality=a.output/"requests_quality.parquet"; fidelity=a.output/"requests_fidelity.parquet"
 con=duckdb.connect();con.execute(f"PRAGMA threads={a.threads}")
 # Group same-timestamp duplicates atomically; conflicting labels are excluded.
 query="""
 COPY (
  WITH grouped AS (
   SELECT uid,timestamp,raw_item_id,min(is_organic)::UTINYINT AS is_organic,
          min(label)::UTINYINT AS label,count(DISTINCT label) AS label_count
   FROM read_parquet(?) WHERE selector_rank<=10000 AND timestamp>=? AND timestamp<?
   GROUP BY uid,timestamp,raw_item_id
  )
  SELECT concat('__REQUEST_PREFIX__:',uid,':',timestamp,':',raw_item_id) AS request_id,
         g.uid::UBIGINT AS uid,g.timestamp::UBIGINT AS query_timestamp,'matrix_horizon' AS time_block,
         g.raw_item_id::UBIGINT AS raw_item_id,coalesce(m.item_idx,0)::UBIGINT AS item_idx,
         (coalesce(m.item_idx,0)<>0) AS target_known,g.is_organic,g.label
  FROM grouped g LEFT JOIN read_parquet(?) m USING(raw_item_id)
  WHERE label_count=1 AND EXISTS (
   SELECT 1 FROM read_parquet(?) l WHERE l.uid=g.uid AND l.timestamp<g.timestamp LIMIT 1
  )
  ORDER BY query_timestamp,uid,request_id
 ) TO '__QUALITY__' (FORMAT PARQUET, COMPRESSION ZSTD)
 """.replace("__QUALITY__",str(quality.resolve()).replace("'","''")).replace("__REQUEST_PREFIX__",str(c["contract"]).replace("'","''"))
 con.execute(query,[str(feedback),start,end,str(mapping),str(listens)])
 con.execute("COPY (SELECT request_id,uid,query_timestamp,time_block,raw_item_id,item_idx,target_known,is_organic FROM read_parquet(?)) TO '"+str(fidelity.resolve()).replace("'","''")+"' (FORMAT PARQUET, COMPRESSION ZSTD)",[str(quality)])
 con.close()
 payload={"status":"hstu_native_parallel_matrix_manifest","contract":str(a.contract.relative_to(ROOT)),"contract_sha256":digest(a.contract),"threads":a.threads,"window_days_half_open":[start_day,end_day],"base_features_materialized":False,"artifacts":{x.name:{"rows":pq.read_metadata(x).num_rows,"sha256":digest(x)} for x in (quality,fidelity)}}
 (a.output/"manifest.json").write_text(json.dumps(payload,indent=2)+"\n");print(json.dumps(payload,indent=2))
if __name__=="__main__":main()
