#!/usr/bin/env python3
"""Direct paired-UID AUC/recovery increments for one complete E5 edge.

Load committed probe/anchor shards, verify their frozen identities and hashes,
then jointly resample ALL eligible UIDs for every stage and correction seed.
An increment interval is bootstrapped from the paired increment itself; no
interval endpoints are subtracted. Training-seed ranges remain separate from
conditional UID intervals, and missing seeds remain explicitly incomplete.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys


import numpy as np
import pyarrow.parquet as pq

from experiments.ch3_motivation.code.common import ROOT, write_json
from hstu_kvcache.evaluation.statistics import weighted_auc
from hstu_kvcache.common.artifacts import sha256

METHODS=("f","fg","fgh","f_extended","fg_extended")
CONTRASTS=dict(g=("fg","f"),h=("fgh","fg"),
    f_extended_minus_f=("f_extended","f"),fg_extended_minus_fg=("fg_extended","fg"))


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def _context(identity):
    return {key:identity.get(key) for key in
        ("edge","panel","users","models","input_binding_sha256","scenario","sources","backend")}


def _score_key(row):
    return int(row["uid"]),int(row["query_timestamp"]),int(row["raw_item_id"])


def _aligned_rows(rows,users):
    rows=sorted(rows,key=_score_key)
    keys=[_score_key(row) for row in rows]
    if len(set(keys))!=len(keys):
        raise ValueError("duplicate score event keys")
    allowed=set(users)
    if any(int(row["uid"]) not in allowed or row["label"] not in (0,1) or
           not np.isfinite(float(row["score"])) for row in rows):
        raise ValueError("invalid UID, label or nonfinite score")
    return rows


def _same_events(a,b):
    return [(*_score_key(row),int(row["item_idx"]),int(row["label"])) for row in a]==[
        (*_score_key(row),int(row["item_idx"]),int(row["label"])) for row in b]


class ShardLoader:
    """Only the two committed on-disk formats actually written by E5."""
    def __init__(self):
        self.hashes={}
        self.anchors={}

    def verify(self,path,expected=None):
        path=Path(path)
        name=str(path.resolve())
        digest=self.hashes.get(name)
        if digest is None:
            digest=sha256(path)
            self.hashes[name]=digest
        if expected is not None and digest!=expected:
            raise ValueError(f"artifact hash changed: {path}")
        return digest

    def json(self,path):
        self.verify(path)
        return json.loads(Path(path).read_text())

    def load_anchors(self,summary):
        identity=summary["identity"]
        folder=Path(summary["anchor_source"]["path"])
        key=str(folder.resolve())
        current_key=_digest({name:identity[name] for name in
            ("edge","models","users","input_binding_sha256","sources")})
        if summary["anchor_source"]["key"]!=current_key:
            raise ValueError("anchor source key changed")
        if key in self.anchors:
            context,groups=self.anchors[key]
            if context!=_context(identity):
                raise ValueError("shared anchor context differs from probe")
            return groups
        anchor_identity=self.json(folder.parent/"configuration.json")
        if _context(anchor_identity)!=_context(identity):
            raise ValueError("anchor edge/population/models/input/source/backend differ from probe")
        anchor_key=_digest({name:anchor_identity[name] for name in
            ("edge","models","users","input_binding_sha256","sources")})
        if summary["anchor_source"]["key"]!=anchor_key:
            raise ValueError("anchor source key changed")
        users=anchor_identity["users"]
        fingerprint=_digest(anchor_identity)
        rows=[]
        covered=[]
        for marker_path in sorted((folder/"quality_shards").glob("part_*.json")):
            marker=self.json(marker_path)
            offset=int(marker_path.stem.split("_")[-1])
            n=marker["users"]
            block=users[offset:offset+n]
            uid_hash=hashlib.sha256(json.dumps(block).encode()).hexdigest()
            if (offset!=len(covered) or marker["offset"]!=offset or marker["fingerprint"]!=fingerprint or
                marker["uids_sha256"]!=uid_hash or len(block)!=n):
                raise ValueError("anchor shard identity/UID coverage changed")
            path=marker_path.with_suffix(".parquet")
            self.verify(path,marker["scores_sha256"])
            values=pq.read_table(path).to_pylist()
            allowed=set(block)
            if len(values)!=marker["scores"] or any(row["uid"] not in allowed for row in values):
                raise ValueError("anchor shard rows differ from committed block")
            rows.extend(values);covered.extend(block)
        if covered!=users:
            raise ValueError("incomplete anchor UID coverage")
        groups={method:_aligned_rows([row for row in rows if row["method"]==method],users)
                for method in ("reuse","rebuild")}
        if len(groups["reuse"])+len(groups["rebuild"])!=len(rows) or not _same_events(groups["reuse"],groups["rebuild"]):
            raise ValueError("anchor methods/events differ")
        self.anchors[key]=_context(identity),groups
        return groups

    def load(self,path):
        path=Path(path)
        summary=self.json(path)
        if summary.get("status")!="complete" or summary.get("experiment")!="E5":
            raise ValueError("expected a completed E5 quality summary")
        identity=summary["identity"]
        if self.json(path.parent/"configuration.json")!=identity:
            raise ValueError("summary/configuration identity differs")
        users=identity["users"]
        if len(users)<512 or len(set(users))!=len(users) or summary["quality_users"]!=len(users):
            raise ValueError("quality analysis needs at least 512 distinct eligible UIDs")
        if identity["edge"]!=summary["edge"]:
            raise ValueError("summary edge differs from identity")
        self.verify(identity["checkpoint"],identity["checkpoint_sha256"])
        for name,digest in identity["models"].items():
            self.verify(Path(name) if Path(name).is_absolute() else ROOT/name,digest)
        method=summary["method_label"]
        if method not in METHODS or summary["stage"]!={"f_extended":"f","fg_extended":"fg"}.get(method,method):
            raise ValueError("unknown or inconsistent E5 ladder method")
        rows=[];covered=[]
        fingerprint=_digest(identity)
        for marker_path in sorted((path.parent/"probe_blocks").glob("part_*.json")):
            marker=self.json(marker_path)
            offset=int(marker_path.stem.split("_")[-1])
            block=marker["users"]
            if (offset!=len(covered) or block!=users[offset:offset+len(block)] or marker["fingerprint"]!=fingerprint):
                raise ValueError("probe shard identity/UID coverage changed")
            score_path=marker_path.with_suffix(".parquet")
            self.verify(score_path,marker["sha256"])
            values=pq.read_table(score_path).to_pylist()
            allowed=set(block)
            if any(row["method"]!=method or row["uid"] not in allowed for row in values):
                raise ValueError("probe shard method/UID changed")
            rows.extend(values);covered.extend(block)
        if covered!=users or len(rows)!=summary["feedback_events"]:
            raise ValueError("incomplete probe UID/event coverage")
        rows=_aligned_rows(rows,users)
        anchors=self.load_anchors(summary)
        if not _same_events(rows,anchors["reuse"]):
            raise ValueError("probe/anchor score keys, item IDs or labels differ")
        return dict(summary=summary,path=str(path.resolve()),rows=rows,anchors=anchors)


def _interval(values):
    values=np.asarray(values)
    finite=values[np.isfinite(values)]
    return np.percentile(finite,[2.5,97.5]).tolist() if len(finite) else None


def _metric(point,draws,gap_point,gap_draws):
    recovery=100*point/gap_point if gap_point!=0 else None
    valid=np.isfinite(gap_draws)&(gap_draws>0)
    with np.errstate(invalid="ignore",divide="ignore"):
        recovery_draws=100*draws/gap_draws
    return dict(auc_difference=float(point),auc_difference_ci95=_interval(draws),
        auc_percentage_point_difference=float(100*point),
        recovery_percentage_point_difference=float(recovery) if recovery is not None else None,
        recovery_difference_ci95=_interval(recovery_draws) if valid.all() else None,
        nonpositive_or_invalid_reference_gap_replicates=int((~valid).sum()),
        valid_auc_replicates=int(np.isfinite(draws).sum()))


def _seed_status(seeds,expected_seeds):
    seeds=sorted(seeds)
    expected=sorted(expected_seeds) if expected_seeds is not None else None
    complete=len(seeds)==3 and (expected is None or seeds==expected)
    return dict(seeds=seeds,seed_count=len(seeds),expected_seed_count=3,
        expected_seeds=expected,missing_expected_seeds=sorted(set(expected)-set(seeds)) if expected is not None else None,
        missing_seed_count=max(0,3-len(seeds)),three_seed_complete=complete,
        status="three_seed_complete" if complete else "incomplete_or_different_seed_set")


def analyze(summary_paths,*,replicates=1000,seed=17051,expected_seeds=None):
    if replicates<1 or (expected_seeds is not None and len(set(expected_seeds))!=3):
        raise ValueError("positive replicates and three distinct expected seeds required")
    loader=ShardLoader()
    loaded=[loader.load(path) for path in summary_paths]
    if not loaded:
        raise ValueError("provide completed stage summaries")
    context=_context(loaded[0]["summary"]["identity"])
    anchors=loaded[0]["anchors"]
    for point in loaded:
        if _context(point["summary"]["identity"])!=context:
            raise ValueError("one-edge paired analysis requires identical population/models/input/source/backend")
        for method in anchors:
            if not _same_events(point["anchors"][method],anchors[method]) or [r["score"] for r in point["anchors"][method]]!=[r["score"] for r in anchors[method]]:
                raise ValueError("matched anchor scores differ between stage/seed runs")
    population=np.asarray(sorted(context["users"]),dtype=np.int64)
    base=anchors["reuse"]
    labels=np.asarray([row["label"] for row in base],dtype=np.float64)
    indices=np.searchsorted(population,np.asarray([row["uid"] for row in base],dtype=np.int64))
    rng=np.random.default_rng(seed)
    weights=np.asarray([np.bincount(rng.integers(len(population),size=len(population)),minlength=len(population))
                        for _ in range(replicates)],dtype=np.uint16)
    all_weights=np.r_[np.ones((1,len(population)),dtype=np.uint16),weights]
    def auc(rows):
        return weighted_auc(np.asarray([row["score"] for row in rows]),labels,indices,all_weights)
    baseline={method:auc(rows) for method,rows in anchors.items()}
    gap=baseline["rebuild"]-baseline["reuse"]
    if not np.isfinite(gap[0]):
        raise ValueError("point population requires both feedback labels")
    groups=defaultdict(dict)
    for point in loaded:
        summary=point["summary"]
        key=(summary["variant"],summary["fit_users"],summary["width"],summary["depth"])
        method=summary["method_label"]
        seed_group=groups[key].setdefault(summary["seed"],{})
        if method in seed_group:
            raise ValueError("duplicate variant/fit/width/depth/seed/method; select one fitting configuration")
        values=auc(point["rows"])
        if (abs(values[0]-summary["auc"])>1e-12 or abs(baseline["reuse"][0]-summary["reuse_auc"])>1e-12 or
            abs(baseline["rebuild"][0]-summary["rebuild_auc"])>1e-12):
            raise ValueError("summary AUC differs from verified event scores")
        seed_group[method]=dict(auc=values,summary=summary,path=point["path"])
    reports=[]
    for key,seeds in sorted(groups.items()):
        report=dict(zip(("variant","fit_users","width","depth"),key),per_seed=[],across_seed={})
        contrasts=defaultdict(dict)
        methods=defaultdict(dict)
        for correction_seed,points in sorted(seeds.items()):
            per_seed=dict(seed=correction_seed,methods={},increments={},missing_methods=[m for m in METHODS if m not in points])
            for method,point in points.items():
                values=point["auc"]
                gain=values-baseline["reuse"]
                per_seed["methods"][method]=dict(auc=float(values[0]),auc_ci95=_interval(values[1:]),
                    recovery_percent=float(100*gain[0]/gap[0]) if gap[0]!=0 else None,
                    recovery_percent_ci95=_interval(100*gain[1:]/gap[1:]) if np.all(np.isfinite(gap[1:])&(gap[1:]>0)) else None,
                    source_summary=point["path"])
                methods[method][correction_seed]=values
            for name,(after,before) in CONTRASTS.items():
                if after not in points or before not in points:
                    continue
                delta=points[after]["auc"]-points[before]["auc"]
                per_seed["increments"][name]=dict(after=after,before=before,
                    **_metric(delta[0],delta[1:],gap[0],gap[1:]))
                contrasts[name][correction_seed]=delta
            report["per_seed"].append(per_seed)
        for family,values in (("methods",methods),("increments",contrasts)):
            report["across_seed"][family]={}
            for name,seed_values in values.items():
                matrix=np.stack(list(seed_values.values()))
                point_values=matrix[:,0]
                mean_draws=matrix[:,1:].mean(0)
                mean_point=float(point_values.mean())
                value=dict(**_seed_status(seed_values,expected_seeds),
                    auc_mean=mean_point,auc_range=[float(point_values.min()),float(point_values.max())],
                    mean_paired_uid_ci95=_interval(mean_draws),
                    recovery_mean=float(100*mean_point/gap[0]) if gap[0]!=0 else None,
                    recovery_range=sorted((100*point_values/gap[0]).tolist()) if gap[0]!=0 else None)
                if family=="methods":
                    gains=point_values-baseline["reuse"][0]
                    value["recovery_mean"]=float(100*gains.mean()/gap[0]) if gap[0]!=0 else None
                    value["recovery_range"]=[float(np.min(100*gains/gap[0])),float(np.max(100*gains/gap[0]))] if gap[0]!=0 else None
                    recovery_numerator=mean_draws-baseline["reuse"][1:]
                elif value["recovery_range"] is not None:
                    value["recovery_range"]=[value["recovery_range"][0],value["recovery_range"][-1]]
                    recovery_numerator=mean_draws
                else:
                    recovery_numerator=mean_draws
                value["mean_paired_uid_recovery_ci95"]=(_interval(100*recovery_numerator/gap[1:])
                    if np.all(np.isfinite(gap[1:])&(gap[1:]>0)) else None)
                if family=="increments":
                    value["auc_difference_mean"]=value.pop("auc_mean")
                    value["auc_difference_range"]=value.pop("auc_range")
                    value["recovery_percentage_point_difference_mean"]=value.pop("recovery_mean")
                    value["recovery_percentage_point_difference_range"]=value.pop("recovery_range")
                report["across_seed"][family][name]=value
        report.update(_seed_status(seeds,expected_seeds))
        report["included_methods"]=sorted(methods)
        report["missing_methods_global"]=[method for method in METHODS if method not in methods]
        if not all(value["three_seed_complete"] for value in report["across_seed"]["methods"].values()):
            report["three_seed_complete"]=False
            report["status"]="incomplete_method_seed_coverage"
        reports.append(report)
    return dict(status="complete",experiment="E5 direct paired increments",edge=context["edge"],
        population=context["users"],quality_users=len(population),users_with_feedback=len({row["uid"] for row in base}),
        feedback_events=len(base),models=context["models"],groups=reports,
        anchors=dict(reuse_auc=float(baseline["reuse"][0]),rebuild_auc=float(baseline["rebuild"][0]),
            reference_gap=float(gap[0]),reference_gap_ci95=_interval(gap[1:]),
            nonpositive_or_invalid_gap_replicates=int((~(np.isfinite(gap[1:])&(gap[1:]>0))).sum())),
        confidence=dict(method="joint paired UID-cluster percentile bootstrap",replicates=replicates,seed=seed,
            scope="all eligible UIDs including users without feedback; conditional on fixed fitted weights; mean UID intervals do not estimate training-seed uncertainty",
            increments="bootstrap after-minus-before directly with identical UID multiplicities; recovery denominator is the same paired Rebuild-minus-Reuse draw; no clipping or independent-CI subtraction",
            recovery_rule="no bounded recovery CI when any bootstrap reference gap is nonpositive/invalid; retain signed point ratios"),
        verification=dict(artifact_sha256=loader.hashes,context=context,
            analysis_sources={str(path.relative_to(ROOT)):sha256(path) for path in
                (Path(__file__),ROOT/"src/hstu_kvcache/evaluation/statistics.py")}))


def run(args):
    result=analyze(args.summary,replicates=args.replicates,seed=args.bootstrap_seed,expected_seeds=args.expected_seeds)
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/"summary.json",result)
    rows=[]
    for group in result["groups"]:
        common={key:group[key] for key in ("variant","fit_users","width","depth")}
        for per_seed in group["per_seed"]:
            for contrast,value in per_seed["increments"].items():
                ci=value["auc_difference_ci95"] or [None,None]
                rows.append(dict(common,seed=per_seed["seed"],contrast=contrast,
                    auc_difference=value["auc_difference"],auc_ci95_low=ci[0],auc_ci95_high=ci[1],
                    recovery_percentage_point_difference=value["recovery_percentage_point_difference"]))
    if rows:
        with (args.output/"increments.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(status="complete",output=str(args.output),groups=len(result["groups"]),edge=result["edge"])),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--replicates",type=int,default=1000)
    p.add_argument("--bootstrap-seed",type=int,default=17051)
    p.add_argument("--expected-seeds",type=int,nargs=3)
    run(p.parse_args())
