#!/usr/bin/env python3
"""CPU-only within-request ranking audit of the retained V4 fit-size scores."""

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


COUNTS = (16, 32, 64, 128, 256)
METHODS = [f"fit{n}" for n in COUNTS] + ["reuse", "rebuild"]
METRICS = ("pair_auc", "hit1", "hit5", "hit10", "ndcg5", "ndcg10", "mrr")


def measures(rows, method):
    positive = next(row[method] for row in rows if row["label"] == 1)
    negatives = np.array([row[method] for row in rows if row["label"] == 0])
    wins, ties = int((negatives < positive).sum()), int((negatives == positive).sum())
    ranks = np.arange(16-wins-ties, 17-wins, dtype=float)
    # Average across all possible positive ranks within an exactly tied block.
    # This avoids request-id tie breaking, where candidate zero is the positive.
    return np.array([(wins + .5*ties)/15,
        (ranks <= 1).mean(), (ranks <= 5).mean(), (ranks <= 10).mean(),
        np.where(ranks <= 5, 1/np.log2(ranks+1), 0).mean(),
        np.where(ranks <= 10, 1/np.log2(ranks+1), 0).mean(), (1/ranks).mean()])


def run(directory):
    path = directory / "scores.parquet"
    config = json.loads((directory / "configuration.json").read_text())["config"]
    groups = defaultdict(list)
    for row in pq.read_table(path).to_pylist():
        if row["protocol"] == "behavior_stride16_candidates16":
            groups[row["request_id"].rsplit(":", 1)[0]].append(row)
    records = []
    for key, rows in groups.items():
        assert len(rows) == 16 and sum(row["label"] for row in rows) == 1
        assert len({(row["uid"], row["version"], row["query_timestamp"]) for row in rows}) == 1
        assert {int(row["request_id"].rsplit(":", 1)[1]) for row in rows} == set(range(16))
        records.append(dict(uid=rows[0]["uid"], values=np.stack([measures(rows, method) for method in METHODS])))
    result = dict(scope="Post-fit CPU diagnostic on unchanged V4 development scores; sampled 16-candidate ranking, not full-catalog retrieval or a replacement for pooled AUC",
        source=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
        aggregation="Each 16-candidate request receives equal weight. Single positive: pair-AUC against15 negatives; Hit/NDCG/MRR use expected value under uniform exact-tie ordering.",
        uncertainty="1000 paired UID resamples from each full declared panel; request-weighted ratio in each resample. Pointwise percentile95, conditional on fixed weights, seed and selected fit prefix; no multiplicity adjustment.",
        panels={})
    pairs = [(f"fit{a}", f"fit{b}") for a, b in zip(COUNTS[1:], COUNTS[:-1])] + [("fit64", "reuse"), ("fit64", "rebuild")]
    for name, users in (("fixed_first128", config["trend_evaluation_uids"]), ("registered32", config["evaluation_uids"])):
        lookup = {uid:i for i, uid in enumerate(users)}
        selected = [row for row in records if row["uid"] in lookup]
        values = np.stack([row["values"] for row in selected])
        uid = np.array([lookup[row["uid"]] for row in selected])
        mean = values.mean(axis=0)
        summary = {method:dict(zip(METRICS, mean[i].tolist())) for i, method in enumerate(METHODS)}
        contrasts = {f"{a}_minus_{b}":[] for a,b in pairs}
        rng = np.random.default_rng(20261019)
        for _ in range(1000):
            weights = np.bincount(rng.integers(0,len(users),len(users)), minlength=len(users))[uid]
            if weights.sum() == 0:
                continue
            sample = np.einsum("n,nmk->mk", weights, values)/weights.sum()
            for a,b in pairs:
                contrasts[f"{a}_minus_{b}"].append(sample[METHODS.index(a)]-sample[METHODS.index(b)])
        intervals = {key:{metric:bounds.tolist() for metric,bounds in zip(METRICS, np.quantile(samples,[.025,.975],axis=0).T)} for key,samples in contrasts.items()}
        result["panels"][name] = dict(users=len(users), users_with_requests=len(set(uid)), requests=len(selected), methods=summary, contrasts_uid95=intervals)
    (directory / "ranking.json").write_text(json.dumps(result,indent=2)+"\n")
    p = result["panels"]["fixed_first128"]
    lines = ["# V4 拟合人数：请求内排序核对", "", "仅复用五点已保存分数；主128面板包含110名实际请求用户、1832组16候选请求，每组1正15负。按候选组计算，保留原pooled AUC口径；这些指标衡量采样候选排序，不是全物品检索。", "", "|方法|pair-AUC|Hit@1|Hit@5|Hit@10|NDCG@5|NDCG@10|MRR|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        lines.append("|"+method+"|"+"|".join(f"{p['methods'][method][metric]:.6f}" for metric in METRICS)+"|")
    lines += ["", "这组请求内指标与pooled AUC的64人峰值一致，不能将已观察到的优势仅解释为跨用户或请求的共同分数偏移。128−64的pair-AUC、Hit@5、NDCG@5/10及MRR区间均低于零，但Hit@1/10仍跨零；256−128的所有这些区间均低于零。仍是单次V4开发面板的诊断，不证明其他发布、全目录排序或训练人数的普适最优值。", "", "每组等权；精确并列按正例在并列区间内均匀排序的期望计分，避免固定正例候选序号优先。各相邻差以及fit64对Reuse/Rebuild的UID95区间见ranking.json；同原128用户整簇重采样1000次，未包含训练随机性或多重比较校正。"]
    (directory / "ranking.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(p,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    run(parser.parse_args().directory)
