"""All-matrix timestamp guard, scalar routing and bounded resume evidence."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code import run_lowrank_timestamp_guarded as entry, lowrank_batch
from experiments.ch3_motivation.code.probe_replay import ProbeTrajectory
from experiments.ch3_motivation.code.probe_variants import build_variant
from hstu_kvcache.models import HSTU,HSTUConfig
from hstu_kvcache.common.artifacts import sha256


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND","torch")
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_injection_is_local_nonreentrant_and_restored_after_failure():
    names=("decompose_interaction","decompose_interaction_batch","compress_snapshot","source_paths")
    original={name:getattr(entry.legacy,name) for name in names}
    independent=lowrank_batch.decompose_interaction_batch
    with pytest.raises(RuntimeError,match="fixture interruption"):
        with entry.guarded_execution():
            assert all(getattr(entry.legacy,name) is not value for name,value in original.items())
            assert lowrank_batch.decompose_interaction_batch is independent
            with pytest.raises(RuntimeError,match="cannot overlap"):
                with entry.guarded_execution():pass
            raise RuntimeError("fixture interruption")
    assert all(getattr(entry.legacy,name) is value for name,value in original.items())
    records=entry.sources()
    assert set(str(path.resolve().relative_to(ROOT)) for path in original["source_paths"]())<=set(records)
    assert records["experiments/ch3_motivation/code/run_lowrank_timestamp_guarded.py"]==sha256(Path(entry.__file__))
    assert "src/hstu_kvcache/evaluation/history.py" in records
    assert not torch.cuda.is_initialized()


def test_scalar_randomized_branch_is_screened_and_failed_matrix_gets_exact_fallback():
    generator=torch.Generator().manual_seed(123)
    low=torch.randn(24,4,dtype=torch.float64,generator=generator)[:,None]*torch.randn(1,30,1,dtype=torch.float64,generator=generator)
    flat=torch.randn(24,30,4,dtype=torch.float64,generator=generator)
    options=dict(method="randomized",max_rank=6,oversample=2,power_iterations=0)
    with entry.guarded_execution():
        batch=entry.legacy.decompose_interaction_batch(torch.stack((low,flat)),**options)
        scalar=entry.legacy.decompose_interaction(flat,**options)
        exact=entry.legacy.decompose_interaction(flat,method="exact",max_rank=6)
    assert not batch[0].metadata["energy_guard"]["exact_fallback"]
    assert batch[1].metadata["energy_guard"]["exact_fallback"] and scalar.metadata["energy_guard"]["exact_fallback"]
    assert len(scalar.metadata["energy_guard"]["all_sketch_singular_values"])==8
    assert "energy_guard" not in exact.metadata
    torch.testing.assert_close(scalar.reconstruct(6),exact.reconstruct(6),atol=1e-11,rtol=1e-11)
    torch.testing.assert_close(batch[1].reconstruct(6),exact.reconstruct(6),atol=1e-11,rtol=1e-11)


def tiny_snapshot(*,zero_h=False,cleared=False):
    torch.manual_seed(83)
    cfg=HSTUConfig(num_items=32,num_behaviors=2,hidden_size=8,num_heads=2,num_layers=2,
        max_seq_len=8,input_dropout=0.,attn_dropout=0.)
    parent=HSTU(cfg).eval().requires_grad_(False);current=copy.deepcopy(parent)
    with torch.no_grad():current.item_emb.weight.add_(torch.randn_like(current.item_emb.weight)*.01)
    probe=build_variant("full_item",current,stage="fgh",hidden_width=8,depth=1,query_chunk=3,row_chunk=2).eval().requires_grad_(False)
    with torch.no_grad():
        for parameter in probe.parameters():parameter.normal_(std=.02)
        if zero_h:
            for network in probe.h:network.weights[-1].zero_();network.biases[-1].zero_()
    times=np.r_[[100,200,300],np.arange(86401,86410)] if cleared else np.array([100,200,300,86401,86403])
    events=(times,np.arange(1,len(times)+1),np.zeros(len(times),dtype=np.int64))
    settings=dict(edge=1,release_day=1,end_day=2,history_length=8,append_band_size=4)
    trajectory=ProbeTrajectory(parent,current,probe,events,settings)
    trajectory.advance(86410 if cleared else 86402);trajectory._prepare()
    return current,probe,trajectory


@pytest.mark.parametrize("batch_heads",[1,2,25])
@pytest.mark.parametrize("zero_h",[False,True])
def test_real_complete_model_snapshot_covers_every_variant_and_preserves_state(batch_heads,zero_h):
    model,probe,trajectory=tiny_snapshot(zero_h=zero_h)
    before=trajectory.cache.k.clone(),trajectory.cache.v.clone(),trajectory.native.cursor
    options=dict(ranks=(0,1,2),variants=tuple(entry.legacy.VARIANTS),method="randomized",oversample=16,
        power_iterations=2,decomposition_batch_heads=batch_heads,verify_svd_heads=2,batch_rank_scores=True)
    original,_=entry.legacy.compress_snapshot(model,probe,trajectory,[9,10,11,12,13,14],[1,4],torch.tensor([1.]),**options)
    with entry.guarded_execution():
        result,report=entry.legacy.compress_snapshot(model,probe,trajectory,[9,10,11,12,13,14],[1,4],torch.tensor([1.]),**options)
    for method in original:torch.testing.assert_close(result[method],original[method],atol=1e-6,rtol=1e-6)
    guard=report["energy_guard"]
    assert guard["matrices"]==guard["expected_matrices"]==16
    assert guard["passed_without_exact"]+guard["exact_fallback_matrices"]==16
    assert guard["all_matrix_scope"] and "records" not in guard and "failed_matrix_examples" not in guard
    if zero_h:assert guard["exact_fallback_matrices"]==0
    assert report["full_rank_logit_max_error"]<1e-5 and report["full_rank_read_max_error"]<1e-5
    assert torch.equal(trajectory.cache.k,before[0]) and torch.equal(trajectory.cache.v,before[1])
    assert trajectory.native.cursor==before[2]


def test_cleared_inherited_state_retains_feedback_scores_without_fabricated_matrices():
    model,probe,trajectory=tiny_snapshot(cleared=True)
    assert trajectory.source.seq_len==0
    with entry.guarded_execution():
        result,report=entry.legacy.compress_snapshot(model,probe,trajectory,[15,16,17,18],[0,2],torch.tensor([1.]),
            ranks=(0,1,2),variants=tuple(entry.legacy.VARIANTS),method="randomized",oversample=16,power_iterations=2,
            decomposition_batch_heads=1,batch_rank_scores=True)
    assert len(result)==14 and all(value.shape==(2,) for value in result.values())
    assert report["energy_guard"]["matrices"]==report["energy_guard"]["expected_matrices"]==0


def test_only_diagnostic_keeps_bounded_failed_matrix_examples(monkeypatch):
    generator=torch.Generator().manual_seed(123)
    matrices=torch.randn(3,24,30,4,dtype=torch.float64,generator=generator)
    def snapshot(*args,**options):
        entry.legacy.decompose_interaction_batch(matrices,method="randomized",max_rank=6,oversample=2,power_iterations=0)
        return {},dict(matrix_shapes=[dict(count=3)])
    monkeypatch.setattr(entry.legacy,"compress_snapshot",snapshot)
    with entry.guarded_execution(diagnostic=True):_,report=entry.legacy.compress_snapshot(method="randomized")
    guard=report["energy_guard"]
    assert guard["matrices"]==guard["exact_fallback_matrices"]==3
    assert len(guard["failed_matrix_examples"])==2 and "records" not in guard
    assert all(row["all_sketch_singular_values"] for row in guard["failed_matrix_examples"])


def snapshot_record(error=1e-6):
    records=[dict(exact_fallback=False,captured_energy_ratio_lower_bound=[.99,1.,1.],all_sketch_singular_values=[2.,1.]),
        dict(exact_fallback=True,captured_energy_ratio_lower_bound=[.8,.8,.8],all_sketch_singular_values=[1.,.5])]
    guard=entry.summarize_guard(records,2,guard_seconds=.1,fallback_seconds=.2)
    return dict(matrix_shapes=[dict(count=2)],energy_guard=guard,full_rank_logit_max_error=error,
        full_rank_read_max_error=error,full_rank_real_candidate_shape_logit_max_error=error)


def write_block(output,index,configuration,snapshot):
    blocks=output/"blocks";blocks.mkdir(parents=True,exist_ok=True)
    identity=hashlib.sha256(json.dumps(configuration,sort_keys=True).encode()).hexdigest()
    path=blocks/f"part_{index:06d}.json"
    path.write_text(json.dumps(dict(identity=identity,report=dict(snapshots=[snapshot]))))
    return path


def test_aggregation_covers_resumed_blocks_and_rejects_incomplete_scope_or_nan(tmp_path):
    configuration=dict(source_sha256={"original.py":"sha"})
    write_block(tmp_path,0,configuration,snapshot_record())
    path=write_block(tmp_path,4,configuration,snapshot_record(2e-6))
    result=entry.aggregate_blocks(tmp_path,configuration)
    assert result["energy_guard"]["matrices"]==result["energy_guard"]["expected_matrices"]==4
    assert result["energy_guard"]["passed_without_exact"]==result["energy_guard"]["exact_fallback_matrices"]==2
    assert result["full_rank_max_error"]==2e-6 and result["guarded_snapshots"]==2
    marker=json.loads(path.read_text());marker["report"]["snapshots"][0]["energy_guard"]["matrices"]-=1
    path.write_text(json.dumps(marker))
    with pytest.raises(ValueError,match="every decomposed matrix"):entry.aggregate_blocks(tmp_path,configuration)
    marker["report"]["snapshots"][0]=snapshot_record(float("nan"));path.write_text(json.dumps(marker))
    with pytest.raises(ValueError,match="finite and nonnegative"):entry.aggregate_blocks(tmp_path,configuration)


def test_actual_cli_and_result_plumbing_keep_hash_and_diagnostic_no_auc(tmp_path,monkeypatch):
    args=entry.parser().parse_args(["--edge","2","--probe","fgh.pt","--fg-probe","fg.pt","--output",str(tmp_path),
        "--diagnostic-users","8","--oversample","16","--power-iterations","2","--decomposition-batch-heads","25",
        "--batch-rank-scores","--execution","projected"])
    original=entry.legacy.source_paths
    def fake_run(args):
        configuration=dict(source_sha256={str(path.relative_to(ROOT)):sha256(path) for path in entry.legacy.source_paths()},
            checkpoint_sha256="trained-h",trained_fg_control=dict(checkpoint_sha256="trained-fg"),models={"v1":"old","v2":"current"},
            population=dict(manifest="frozen"),variants=args.variants.split(","),decomposition=dict(method=args.method,
                oversample=args.oversample,power_iterations=args.power_iterations,batch_heads=args.decomposition_batch_heads))
        (args.output/"configuration.json").write_text(json.dumps(configuration))
        write_block(args.output,0,configuration,snapshot_record())
        identity=hashlib.sha256(json.dumps(configuration,sort_keys=True).encode()).hexdigest()
        return dict(status="complete",users=8,snapshots=1,identity_hash=identity,execution_seconds=3.,peak_reserved_bytes=1234)
    monkeypatch.setattr(entry.legacy,"run",fake_run)
    result=entry.run(args)
    assert entry.legacy.source_paths is original
    assert result["quality_auc_computed"] is False and result["seconds"]==3.
    assert result["full_rank_max_error"]==1e-6 and result["energy_guard"]["matrices"]==2
    assert result["peak_reserved_bytes"]==1234 and "points" not in result
    configuration=json.loads((tmp_path/"configuration.json").read_text())
    assert result["identity_hash"]==hashlib.sha256(json.dumps(configuration,sort_keys=True).encode()).hexdigest()
    assert "experiments/ch3_motivation/code/run_lowrank_timestamp_guarded.py" in configuration["source_sha256"]
    assert (tmp_path/"diagnostic.json").exists() and not torch.cuda.is_initialized()

