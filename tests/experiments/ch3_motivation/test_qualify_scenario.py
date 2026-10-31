"""The scenario screen compares new matched anchors, not per-UID gains."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.qualify_scenario import (
    bootstrap_anchors, frozen_panels, old_native_reference,
)
from experiments.ch3_motivation.code.selective import evaluate_reference
from hstu_kvcache.models import HSTU,HSTUConfig


def test_three_anchor_screen_uses_matched_global_auc_and_paired_users():
    users=list(range(512))
    rows=[]
    for uid in users:
        label=uid%2
        for method,value in (("old_full",.4),("reuse",.4), ("rebuild",float(label))):
            rows.append(dict(uid=uid,query_timestamp=1,raw_item_id=uid,label=label,method=method,score=value))
    summary=bootstrap_anchors(rows,users,replicates=32,seed=17)
    assert summary["auc"]==dict(old_full=.5,reuse=.5,rebuild=1.)
    assert summary["point_qualifies"] and summary["uid95_qualifies"]
    assert summary["new_rebuild_minus_old_full"]==.5
    assert summary["new_rebuild_minus_reuse"]==.5
    assert summary["release_gain_erased_percent"]==100
    assert summary["confidence"]["all_selected_users_including_no_feedback"]
    with pytest.raises(ValueError,match="512"):
        bootstrap_anchors(rows,[0],replicates=32)


def test_screen_rejects_anchor_event_misalignment():
    users=list(range(512))
    rows=[dict(uid=uid,query_timestamp=1,raw_item_id=uid,label=uid%2,method=method,score=float(uid%2))
          for uid in users for method in ("old_full","reuse","rebuild")]
    rows[-1]["raw_item_id"]+=1
    with pytest.raises(ValueError,match="coverage"):
        bootstrap_anchors(rows,users,replicates=8)


@torch.no_grad()
def test_old_full_writes_and_scores_with_old_weights(monkeypatch):
    monkeypatch.setenv("EVOKV_ATTENTION_BACKEND","torch")
    torch.manual_seed(151)
    old=HSTU(HSTUConfig(num_items=32,num_behaviors=2,hidden_size=32,num_heads=1,
        num_layers=2,max_seq_len=4,input_dropout=0.,attn_dropout=0.)).eval().requires_grad_(False)
    new=copy.deepcopy(old)
    new.item_emb.weight.add_(torch.randn_like(new.item_emb.weight)*.3)
    events=(np.arange(10)*10,np.arange(1,11),np.zeros(10,dtype=np.int64))
    histories=SimpleNamespace(rows={7:events})
    by_user={7:{65:[dict(uid=7,query_timestamp=65,raw_item_id=21,item_idx=21,label=1)],
                85:[dict(uid=7,query_timestamp=85,raw_item_id=22,item_idx=22,label=0)]}}
    cfg=dict(edge=1,release_day=60/86400,end_day=100/86400,history_length=4,
        append_band_size=2,query_batch=16)
    actual=old_native_reference(old,histories,by_user,[7],cfg,"cpu")
    same_old,_=evaluate_reference(old,old,histories,by_user,[7],
        [dict(tag="old_full",method="reuse")],cfg,"cpu")
    current_reuse,_=evaluate_reference(old,new,histories,by_user,[7],
        [dict(tag="old_full",method="reuse")],cfg,"cpu")
    assert actual==same_old
    assert any(a["score"]!=b["score"] for a,b in zip(actual,current_reuse,strict=True))
