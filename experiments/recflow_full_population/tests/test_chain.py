import json
import math
from pathlib import Path

import pytest

from experiments.recflow_full_population.code.chain import Chain, FOUNDATION_SECONDS, REPO
from experiments.recflow_full_population.code.configuration import DEFAULT_CONFIG
from experiments.recflow_full_population.code.data_common import json_save, sha256


def test_dispatch_full_epochs_actual_parent_and_unchanged_resume_schedule(tmp_path):
    base=Path(__file__).resolve().parents[1]
    chain=Chain(tmp_path/"run",base/"data",tmp_path/"readiness.json")
    warm=chain.train_args(tmp_path/"warm","warmup",list(range(1,24)),1,16384,stop=4096)
    assert warm[warm.index("--micro-batch")+1] == 16
    assert "--max-steps" not in warm and "--request-limit" not in warm
    resumed=chain.train_args(tmp_path/"warm","warmup",list(range(1,24)),1,16384,resume=tmp_path/"first.pt")
    assert "--stop-after-steps" not in resumed and "--parent" not in resumed
    updated=chain.train_args(tmp_path/"update","update",[28],1,100000,parent=tmp_path/"m0.pt")
    assert updated[updated.index("--parent")+1] == tmp_path/"m0.pt"
    assert updated[updated.index("--wall-hours")+1] == 2
    assert updated[updated.index("--budget-spent-seconds")+1] == 0
    assert FOUNDATION_SECONDS == 54000 and (REPO/"src/hstu_kvcache").exists()
    manifest=json.loads((base/"data/selection_manifest.json").read_text())["datasets"]
    assert math.ceil(manifest["0:24,25,26,27"]["requests"]/64) == 10116
    assert math.ceil(manifest["0:"+",".join(map(str,range(1,24)))]["requests"]/64) == 64431
    assert DEFAULT_CONFIG.is_file()


def test_inventory_retains_negative_cache_gap_and_rejects_wrong_parent(tmp_path):
    chain=Chain(tmp_path,tmp_path,tmp_path/"readiness.json")
    parent=tmp_path/"m0.pt"
    parent.write_bytes(b"previous model")
    json_save(parent.with_suffix(".json"),dict(checkpoint_sha256=sha256(parent)))
    current=tmp_path/"m1.pt"
    current.write_bytes(b"current model")
    json_save(current.with_suffix(".json"),dict(checkpoint_sha256=sha256(current),
        metadata=dict(quality_use_permitted=True,parent_checkpoint_sha256=sha256(parent),
                      arguments=dict(parent=str(parent)))))
    summary=dict(checkpoint_sha256=sha256(current),days=[29],metrics={
        "current":{"ndcg@10":.03,"recall@10":.04},"reuse":{"ndcg@10":.04,"recall@10":.05}},
        paired_current_minus=dict(reuse=dict(ci95=[-.02,-.001])),requests=100,users=40,positive_requests=80,positive_users=35)
    chain.append_model(1,current,parent,tmp_path/"evaluation",summary,[28])
    assert len(chain.versions)==1
    text=(tmp_path/"version_metrics.csv").read_text()
    assert "-0.010000" in text and "-33.333" in text
    other=tmp_path/"wrong.pt"
    other.write_bytes(b"wrong model")
    json_save(other.with_suffix(".json"),dict(checkpoint_sha256=sha256(other)))
    with pytest.raises(ValueError,match="parent changed"):
        chain.append_model(2,current,other,tmp_path/"eval",summary,[28])
