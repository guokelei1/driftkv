"""Graph arithmetic and cold setup must follow physical, complete shapes."""
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.probe_cost import (
    cached_preparation_flops, deployment_graph_setup, padded_read_extra_flops, required_graph_keys,
    scalar_behavior_flops,
)
from experiments.ch3_motivation.code.probes import native_query_flops
from experiments.ch3_motivation.code.probe_variants import build_variant
from hstu_kvcache.models import HSTU,HSTUConfig


@pytest.fixture
def model():
    return HSTU(HSTUConfig(num_items=8,num_behaviors=2,hidden_size=64,num_heads=2,
        num_layers=2,max_seq_len=1024,input_dropout=0.,attn_dropout=0.))


@pytest.mark.parametrize("variant",["full_item","layerwise_kv"])
@pytest.mark.parametrize("rows",[1,1024])
def test_preparation_uses_real_variant_geometry(model,variant,rows):
    probe=build_variant(variant,model,hidden_width=7,depth=1)
    c=probe.cfg
    output=2*c.head_dim if variant=="full_item" else 2*model.cfg.hidden_size
    maps=c.num_heads if variant=="full_item" else 1
    mlp=2*c.row_dim*7+7+2*7*output+output+2*7
    expected=rows*c.num_layers*(2*c.row_dim*c.num_heads+maps*(mlp+2*output))
    assert cached_preparation_flops(probe,rows)==expected
    probe.cfg.normalize=False
    # Base f executes literal-1 output multiplication, layerwise skips it.
    expected=rows*c.num_layers*maps*(mlp+output*(2 if variant=="full_item" else 1))
    assert cached_preparation_flops(probe,rows)==expected


@pytest.mark.parametrize("variant",["full_item","layerwise_kv"])
@pytest.mark.parametrize("rows",[1,1024])
def test_padded_read_is_one_physical_block_independent_of_reference_chunks(model,variant,rows):
    probe=build_variant(variant,model,stage="fgh",hidden_width=7,depth=1,
                        row_chunk=128,query_chunk=2)
    c=probe.cfg
    d=c.head_dim
    def mlp(width):
        return 2*width*7+7+2*7*(2*d)+2*d+2*7
    # Literal executed tensor arithmetic: g includes q/scale (D), q*gK
    # (2D), K/V output scaling (2D), gV product/read addition (2D),
    # two mask multiplies + score-add + weight-reduction (4N).
    g=d+mlp(d)+6*d+4*rows
    h=d+rows*(mlp(d+c.row_dim)+6*d+1)+d
    expected=16*c.num_layers*c.num_heads*(g+h)
    assert padded_read_extra_flops(probe,rows,16)==expected
    probe.cfg.row_chunk=1024
    probe.cfg.query_chunk=16
    assert padded_read_extra_flops(probe,rows,16)==expected
    assert padded_read_extra_flops(probe,rows,0)==0


@pytest.mark.parametrize("variant",["full_item","layerwise_kv"])
def test_setup_uses_full_ledger_final_zero_source_append_and_f_score(model,variant):
    # A quality shortcut could observe no graph at all; the complete ledger
    # still contains a full-cache behavior tail and a final source eviction.
    execution=dict(read_shapes=[
        dict(kind="query",history_length=1,inherited_rows=1,queries=16,calls=1),
        dict(kind="query",history_length=1024,inherited_rows=3,queries=16,calls=4),
        dict(kind="query",history_length=1024,inherited_rows=2,queries=5,calls=2),
        dict(kind="query",history_length=1024,inherited_rows=0,queries=1,calls=2),
        dict(kind="behavior",history_length=1023,inherited_rows=0,queries=1,calls=1)],
        append_shapes=[dict(before_length=1024,width=1,inherited_rows=0,calls=5)])
    probe=build_variant(variant,model,stage="fgh",hidden_width=7,depth=1)
    expected=[("append",1),("score",5),("score",16)]
    assert required_graph_keys(probe,execution,window=1024)==expected
    setup=deployment_graph_setup(probe,execution,model)
    expected_total=0
    for row in setup["per_key"]:
        n,q=row["physical_read_rows"],row["queries"]
        native=(scalar_behavior_flops(model,n) if row["operation"]=="append" else
                native_query_flops(model,history_length=n,queries=q))
        extra=padded_read_extra_flops(probe,n,q)
        assert row["native_forward_flops_per_call"]==native
        assert row["probe_forward_flops_per_call"]==extra
        assert row["setup_flops"]==3*(native+extra)
        expected_total+=3*(native+extra)
    assert setup["shared_graph_setup_flops"]==expected_total
    assert setup["pools"]==1
    f=build_variant(variant,model,stage="f",hidden_width=7,depth=1)
    assert required_graph_keys(f,execution,window=1024)==[("score",5),("score",16)]
    assert deployment_graph_setup(f,execution,model)["shared_graph_setup_probe_flops"]==0


def test_short_window_becoming_full_stays_eager_and_idle_ledger_needs_no_graph(model):
    probe=build_variant("layerwise_kv",model,stage="fg",hidden_width=7,depth=1)
    short=dict(read_shapes=[dict(kind="behavior",history_length=1023,inherited_rows=4,queries=1,calls=1)],
        append_shapes=[dict(before_length=1023,width=1,inherited_rows=4,calls=1)])
    assert required_graph_keys(probe,short,window=1024)==[]
    setup=deployment_graph_setup(probe,dict(read_shapes=[],append_shapes=[]),model)
    assert setup["shared_graph_setup_flops"]==0
