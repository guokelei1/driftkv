#!/usr/bin/env python3
"""Timestamp E6 with every randomized matrix screened before rank replay.

Use this entry in a dedicated process. The legacy module's local aliases are
temporarily replaced only inside guarded_execution(), never on disk; every
alias is restored even on failure. Concurrent legacy calls in that process
are unsupported. Release-joint E6 remains a separate protocol/entry.
"""
from __future__ import annotations

# Allow direct execution from a source checkout as well as python -m.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _REPO = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(_REPO / "src"), str(_REPO)]

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys
from threading import Lock

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import run_lowrank as legacy
from experiments.ch3_motivation.code.common import write_json
from hstu_kvcache.common.artifacts import sha256

_BASE_SOURCE_PATHS=legacy.source_paths
_EXECUTION_LOCK=Lock()
GUARD=dict(enabled=True,ranks=[2,6,16],threshold=.98,relative_energy_tolerance=1e-5,
    fallback="same centered/normalized matrix exact SVD for failed heads only",
    scope="all randomized interaction matrices, including the scalar-head branch; numerical screen, not a formal floating-point proof")


def _extended_sources(paths):
    return list(dict.fromkeys(Path(path).resolve() for path in [*paths,Path(__file__),
        ROOT/"src/hstu_kvcache/evaluation/history.py",ROOT/"src/hstu_kvcache/evaluation/yambda.py",
        ROOT/"src/hstu_kvcache/common/artifacts.py"]))


def source_paths():
    """Read-only inventory; importing this module does not patch or allocate GPU."""
    return _extended_sources(_BASE_SOURCE_PATHS())


def sources():
    return {str(path.relative_to(ROOT)):sha256(path) for path in source_paths()}


def summarize_guard(records,expected_matrices,*,guard_seconds=0.,fallback_seconds=0.):
    if len(records)!=expected_matrices:
        raise ValueError("timestamp energy guard did not cover every decomposed matrix")
    return dict(matrices=len(records),expected_matrices=expected_matrices,
        passed_without_exact=sum(not row["exact_fallback"] for row in records),
        exact_fallback_matrices=sum(row["exact_fallback"] for row in records),
        worst_ratio_lower_bound=min((min(row["captured_energy_ratio_lower_bound"]) for row in records),default=None),
        guard_seconds=guard_seconds,exact_fallback_seconds=fallback_seconds,
        all_matrix_scope=True,protocol=GUARD)


def check_guard_summary(summary,expected):
    if (summary["matrices"]!=expected or summary["expected_matrices"]!=expected or
            summary["passed_without_exact"]+summary["exact_fallback_matrices"]!=expected or
            min(summary["passed_without_exact"],summary["exact_fallback_matrices"])<0 or
            summary["all_matrix_scope"] is not True or summary["protocol"]!=GUARD or
            any(not math.isfinite(summary[key]) or summary[key]<0 for key in ("guard_seconds","exact_fallback_seconds")) or
            (expected and (summary["worst_ratio_lower_bound"] is None or not math.isfinite(summary["worst_ratio_lower_bound"])))):
        raise ValueError("timestamp energy guard summary does not cover every decomposed matrix")


def merge_guard(total,checked):
    for key in ("matrices","expected_matrices","passed_without_exact","exact_fallback_matrices","guard_seconds","exact_fallback_seconds"):
        total[key]+=checked[key]
    ratio=checked["worst_ratio_lower_bound"]
    if ratio is not None:
        total["worst_ratio_lower_bound"]=min(ratio,total["worst_ratio_lower_bound"]) if total["worst_ratio_lower_bound"] is not None else ratio


@contextmanager
def guarded_execution(*,diagnostic=False):
    """Single-process, non-reentrant injection; unrelated modules stay intact."""
    if not _EXECUTION_LOCK.acquire(blocking=False):
        raise RuntimeError("timestamp guard execution cannot overlap within one process")
    names=("decompose_interaction_batch","decompose_interaction","compress_snapshot","source_paths")
    originals={name:getattr(legacy,name) for name in names}
    active=None

    def batch(h,**options):
        nonlocal active
        method=options.get("method","exact")
        options.update(energy_guard=True,energy_threshold=GUARD["threshold"],
            energy_tolerance=GUARD["relative_energy_tolerance"])
        result=originals["decompose_interaction_batch"](h,**options)
        if method=="randomized":
            records=[]
            for obj in result:
                record=obj.metadata.get("energy_guard")
                if record is None:
                    raise ValueError("randomized timestamp matrix lacks its energy certificate")
                records.append(record)
                if active is not None and diagnostic and record["exact_fallback"] and len(active["failed_matrix_examples"])<2:
                    active["failed_matrix_examples"].append(dict(matrix_shape=obj.metadata["matrix_shape"],only=obj.metadata["only"],
                        normalize_kv=obj.metadata["normalize_kv"],seed=obj.metadata["seed"],**record))
            if result and active is not None:
                checked=summarize_guard(records,len(result),guard_seconds=result[0].metadata["energy_guard_batch_seconds"],
                    fallback_seconds=result[0].metadata["exact_fallback_batch_seconds"])
                merge_guard(active,checked)
        return result

    def scalar(h,**options):
        # The old scalar implementation has no guard parameter. Exact oracle
        # checks remain exact; scalar randomized uses the same guarded matrix.
        if options.get("method","exact")=="randomized":return batch(h[None],**options)[0]
        return originals["decompose_interaction"](h,**options)

    def snapshot(*args,**options):
        nonlocal active
        if options.get("method","randomized")!="randomized":
            raise ValueError("guarded timestamp entry measures randomized decomposition only")
        if active is not None:raise RuntimeError("timestamp snapshot guard cannot nest")
        active=summarize_guard([],0)
        if diagnostic:active["failed_matrix_examples"]=[]
        try:
            values,report=originals["compress_snapshot"](*args,**options)
            expected=sum(row["count"] for row in report["matrix_shapes"])
            check_guard_summary(active,expected)
            report["energy_guard"]=active
            return values,report
        finally:active=None

    try:
        legacy.decompose_interaction_batch=batch
        legacy.decompose_interaction=scalar
        legacy.compress_snapshot=snapshot
        legacy.source_paths=lambda:_extended_sources(originals["source_paths"]())
        yield
    finally:
        for name,value in originals.items():setattr(legacy,name,value)
        _EXECUTION_LOCK.release()


def aggregate_blocks(output,configuration):
    """Read certificates from committed blocks, including resumed snapshots."""
    identity=hashlib.sha256(json.dumps(configuration,sort_keys=True).encode()).hexdigest()
    total=summarize_guard([],0);maximum=0.;snapshot_count=0
    for path in sorted((Path(output)/"blocks").glob("part_*.json")):
        marker=json.loads(path.read_text())
        if marker["identity"]!=identity:raise ValueError("timestamp guarded block identity differs")
        for snapshot in marker["report"]["snapshots"]:
            snapshot_count+=1
            count=sum(row["count"] for row in snapshot["matrix_shapes"])
            guard=snapshot["energy_guard"]
            check_guard_summary(guard,count)
            merge_guard(total,guard)
            errors=[snapshot[key] for key in ("full_rank_logit_max_error","full_rank_read_max_error",
                "full_rank_real_candidate_shape_logit_max_error")]
            if any(not math.isfinite(error) or error<0 for error in errors):
                raise ValueError("timestamp full-rank numerical errors must be finite and nonnegative")
            maximum=max(maximum,*errors)
    return dict(energy_guard=total,
        full_rank_max_error=maximum,guarded_snapshots=snapshot_count)


def run(args):
    if args.method!="randomized":raise ValueError("guarded timestamp entry requires randomized decomposition")
    if args.diagnostic_users is not None and not 1<=args.diagnostic_users<=16:
        raise ValueError("timestamp readiness uses1..16 diagnostic users")
    with guarded_execution(diagnostic=args.diagnostic_users is not None):result=legacy.run(args)
    configuration=json.loads((Path(args.output)/"configuration.json").read_text())
    for name,digest in configuration["source_sha256"].items():
        if sha256(ROOT/name)!=digest:raise ValueError("timestamp guarded source changed during execution: "+name)
    aggregate=aggregate_blocks(args.output,configuration)
    if aggregate["guarded_snapshots"]!=result["snapshots"]:
        raise ValueError("timestamp guarded snapshot count differs from committed blocks")
    diagnostic=args.diagnostic_users is not None
    if diagnostic and any(key in result for key in ("points","confidence","same_state_h_disabled_diagnostic")):
        raise ValueError("timestamp numerical diagnostics cannot contain quality AUC results")
    result.update(aggregate,quality_auc_computed=not diagnostic,seconds=result["execution_seconds"],
        numerical_entry=Path(__file__).name,numerical_entry_sha256=sha256(Path(__file__)))
    write_json(Path(args.output)/("diagnostic.json" if diagnostic else "summary.json"),result)
    return result


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument("--edge",type=int,choices=range(1,6),required=True)
    result.add_argument("--input-root",type=Path,default=ROOT/"experiments/ch3_motivation/inputs")
    result.add_argument("--population-file",type=Path)
    result.add_argument("--probe",type=Path,required=True)
    result.add_argument("--fg-probe",type=Path,required=True)
    result.add_argument("--fg-method-label",choices=("fg","fg_extended"),default="fg")
    result.add_argument("--output",type=Path,required=True)
    result.add_argument("--gpu",type=int,default=1)
    result.add_argument("--diagnostic-users",type=int)
    result.add_argument("--sensitivity-users",type=int)
    result.add_argument("--candidates",type=int,default=230)
    result.add_argument("--candidate-chunk",type=int,default=230)
    result.add_argument("--row-chunk",type=int,default=1024)
    result.add_argument("--checkpoint-users",type=int,default=4)
    result.add_argument("--variants",default="both_norm,both_raw,key_norm,value_norm")
    result.add_argument("--method",choices=("randomized",),default="randomized")
    result.add_argument("--oversample",type=int,default=16)
    result.add_argument("--power-iterations",type=int,default=2)
    result.add_argument("--verify-svd-heads",type=int,default=2)
    result.add_argument("--decomposition-batch-heads",type=int,default=25)
    result.add_argument("--batch-rank-scores",action="store_true")
    result.add_argument("--execution",choices=("dense","projected"),default="projected")
    return result


if __name__=="__main__":run(parser().parse_args())
