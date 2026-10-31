"""Local inputs, minimal rolling checkpoints and the declared operation ledger."""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
import duckdb
import numpy as np
import pyarrow as pa
from hstu_kvcache.evaluation.history import load_histories_semijoin
from hstu_kvcache.evaluation.baselines import event_shapes
from hstu_kvcache.evaluation.cost import CostModel, eager_read
from hstu_kvcache.methods.kv_map.cost_large import deployment_flops

DAY = 86400

def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def configuration():
    return json.loads((HERE / "config.json").read_text())

def policies(cfg):
    rows = [dict(tag="reuse", method="reuse"), dict(tag="rebuild", method="rebuild")]
    rows += [dict(tag=f"layer_{a}_{b}", method="layer", interval=[a-1,b-1])
             for a,b in cfg["layer_intervals_one_based"]]
    rows += [dict(tag=f"{kind}_{n}", method=kind, tokens=n)
             for kind in ("tail", "deviation") for n in cfg["tokens"]]
    rows += [dict(tag=f"kv_map_u{p['fit_users']}_w{p['width']}", method="kv_map", **p)
             for p in cfg["kv_map"]]
    return rows

def edge_config(cfg, edge):
    return dict(cfg, edge=edge, release_day=cfg["release_days"][edge-1], end_day=cfg["end_days"][edge-1])

def load_inputs(cfg, edge, rank=0, world=1, limit=None, include_fit=True):
    """Reuse frozen UID membership, rederive the identical causal raw inputs.

    Every rank owns disjoint UIDs, selected before inspecting any model score.
    No input packs or request parquet duplicates are left on disk.
    """
    started = time.perf_counter()
    roles = json.loads((ROOT / cfg["role_source"]).read_text())["roles"]
    quality = roles["quality"][rank::world]
    if limit is not None:
        quality = quality[:limit]
    fit = roles["fit"][:max(p["fit_users"] for p in cfg["kv_map"])] if include_fit else []
    uids = fit + quality
    path = ROOT / cfg["dataset"]
    ds = json.loads(path.read_text())
    start,end = cfg["release_days"][edge-1]*DAY,cfg["end_days"][edge-1]*DAY
    con = duckdb.connect()
    con.execute(f"SET threads={cfg['history_threads']}")
    con.execute("SET memory_limit='16GB'")
    con.register("selected", pa.table({"uid":pa.array(quality,type=pa.uint64())}))
    quoted = lambda p: str(p.resolve()).replace("'", "''")
    feedback = quoted(path.parent / ds["shared_feedback_glob"])
    mapping = quoted(path.parent / ds["item_mapping_path"])
    table = con.execute(f"""
        WITH grouped AS (
          SELECT uid,timestamp,raw_item_id,min(label) AS label
          FROM read_parquet('{feedback}',hive_partitioning=true)
          SEMI JOIN selected USING(uid) WHERE timestamp>=? AND timestamp<?
          GROUP BY uid,timestamp,raw_item_id HAVING min(label)=max(label)
        ) SELECT g.uid,g.timestamp AS query_timestamp,g.raw_item_id,m.item_idx,g.label
          FROM grouped g JOIN read_parquet('{mapping}') m USING(raw_item_id)
          ORDER BY uid,query_timestamp,raw_item_id
    """, [start,end]).fetch_arrow_table()
    con.close()
    by_user = defaultdict(lambda:defaultdict(list))
    for row in table.to_pylist():
        by_user[int(row["uid"])][int(row["query_timestamp"])].append(row)
    histories = load_histories_semijoin(uids, dataset_path=path, known_vocab_size=ds["foundation_items"],
        oov_buckets=256,start_timestamp=start,end_timestamp=end,max_history=1024,threads=cfg["history_threads"])
    h = hashlib.sha256()
    for uid in uids:
        h.update(np.asarray([uid],dtype=np.int64).tobytes())
        for column in histories.rows[uid]:
            h.update(column.tobytes())
    for column in table.columns:
        h.update(column.to_numpy().tobytes())
    return dict(histories=histories,by_user=by_user,quality=quality,fit=fit,
        total_population=len(roles["quality"]),requests=table.num_rows,
        fingerprint=h.hexdigest(),load_seconds=time.perf_counter()-started)

class Rolling:
    """Two atomic generations containing only cursor, labels, UID and scores.

    Exact pooled AUC cannot be recovered by averaging per-block AUC. These
    scalar arrays are the sufficient retained evidence for exact AUC and paired
    UID resampling; no K/V, hidden trajectory or per-step log is serialized.
    """
    def __init__(self, folder, identity, interval=1800):
        self.folder=Path(folder)
        self.folder.mkdir(parents=True,exist_ok=True)
        self.identity=identity
        self.interval=interval
        self.last=time.monotonic()
        self.cursor=0
        self.labels=[]
        self.uids=[]
        self.scores=[]
        self.generation=0
        self.save_count=0
        self.save_seconds=0.
        for name in ("checkpoint.npz", "previous.npz"):
            path=self.folder/name
            if not path.exists():
                continue
            with np.load(path,allow_pickle=False) as a:
                if str(a["identity"]) != identity:
                    raise RuntimeError("checkpoint input/code/configuration identity changed")
                self.cursor=int(a["cursor"])
                self.generation=int(a["generation"])
                self.labels=[a["labels"].copy()]
                self.uids=[a["uids"].copy()]
                self.scores=[a["scores"].copy()]
            break

    def append(self, uid, labels, scores):
        if len(labels):
            self.labels.append(np.asarray(labels,dtype=np.uint8))
            self.uids.append(np.full(len(labels),uid,dtype=np.int64))
            self.scores.append(np.asarray(scores,dtype=np.float32))

    def arrays(self, width):
        return (np.concatenate(self.labels) if self.labels else np.empty(0,np.uint8),
            np.concatenate(self.uids) if self.uids else np.empty(0,np.int64),
            np.concatenate(self.scores) if self.scores else np.empty((0,width),np.float32))

    def save(self, cursor, width, force=False):
        self.cursor=cursor
        if not force and time.monotonic()-self.last < self.interval:
            return False
        started=time.monotonic()
        labels,uids,scores=self.arrays(width)
        tmp=self.folder/"checkpoint.writing.npz"
        self.generation+=1
        np.savez_compressed(tmp,identity=self.identity,cursor=cursor,generation=self.generation,
            labels=labels,uids=uids,scores=scores)
        with tmp.open("rb") as f:
            os.fsync(f.fileno())
        current,previous=self.folder/"checkpoint.npz",self.folder/"previous.npz"
        if current.exists():
            os.replace(current,previous)
        os.replace(tmp,current)
        self.labels=[labels];self.uids=[uids];self.scores=[scores]
        self.save_seconds+=time.monotonic()-started
        self.save_count+=1
        self.last=time.monotonic()
        return True

def ledger(inputs,cfg,rows,fit_costs):
    cm=CostModel(320,10,10,"triton")
    shapes=event_shapes(inputs["histories"],inputs["by_user"],inputs["quality"],cfg)
    denominator=0
    added={p["tag"]:0 for p in rows}
    comparisons={p["tag"]:0. for p in rows}
    boundary={p["tag"]:0 for p in rows}
    for (n,_,_),count in shapes["first"].items():
        cost=cm if n==1024 else CostModel(320,10,10,"torch")
        denominator+=count*cost.full_cache(n)
        for p in rows:
            method=p["method"]
            if method=="reuse":
                value=0
            elif method=="rebuild":
                value=cost.full_cache(n)
            elif method=="kv_map":
                value=deployment_flops(n,p["width"])
            else:
                work=cost.operation_cost(method,n,selected_tokens=p.get("tokens",0),interval=p.get("interval"))
                value=work["total_flops"]
                comparisons[p["tag"]]+=count*work["selection_sort_comparisons_estimate"]
            added[p["tag"]]+=int(count*value)
    for p in rows:
        if p["method"]=="layer" and p["interval"][0]>0:
            boundary[p["tag"]]=sum(n*count for n,count in shapes["initial"].items())*320*4
    native_append=sum(count*(cm if full else CostModel(320,10,10,"torch")).band_append(n,w,window_size=1024)
        for (n,w,full),count in shapes["appends"].items())
    native_read=sum(count*eager_read(cm,n,queries=q) for (n,q),count in shapes["queries"].items())
    initial=sum(count*(cm if n==1024 else CostModel(320,10,10,"torch")).full_cache(n)
        for n,count in shapes["initial"].items())
    return dict(once_rebuild_flops=denominator,repair_flops=added,calibration_flops=fit_costs,
        native_append_flops=native_append,native_read_flops=native_read,common_old_cache_flops=initial,
        old_boundary_bytes=boundary,selection_sort_comparisons=comparisons,
        scope="All frozen eligible UIDs and all events, including no-feedback users and post-last-query maintenance; calibration charged once per full population, not once per GPU shard.",
        boundary_source="Actual old-model input, retained by non-prefix Layer only. One FP32 [N,320] boundary per non-prefix point; copies/retention are bytes, not floating-point operations.")
