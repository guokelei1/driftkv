"""Executed arithmetic for fixed-window probe graphs and their shared setup.

These formulas describe probe_fast's single physical history block. Masked
native rows still execute h. Graph setup is gross model-plus-probe work and
is charged once for every operation/query shape needed by the COMPLETE event
ledger, even when quality replay skips an idle user or an unscored tail.
"""
from __future__ import annotations

from experiments.ch3_motivation.code.probes import native_query_flops
from experiments.ch3_motivation.code.probe_variants import LayerwiseKVProbeLadder


def _head_mlp_flops(cfg,input_width):
    widths=[input_width]+[cfg.hidden_width]*cfg.depth+[2*cfg.head_dim]
    return sum(2*i*o+o for i,o in zip(widths[:-1],widths[1:],strict=True))+2*cfg.depth*cfg.hidden_width


def cached_preparation_flops(probe,rows):
    """f plus one fixed row-feature preparation, using the actual variant.

    Layerwise f runs ONE joint map per layer, rather than one map/head. Its
    forward_flops override accounts for that geometry and for duplicated
    feature normalization. Base/full-item f multiply by a literal 1 even
    with normalization disabled; layerwise f skips that multiplication.
    """
    value=probe.forward_flops(int(rows),0)["f_prepare_flops"]
    if not probe.cfg.normalize and not isinstance(probe,LayerwiseKVProbeLadder):
        c=probe.cfg
        value+=int(rows)*c.num_layers*c.num_heads*2*c.head_dim
    return int(value)


def padded_read_extra_flops(probe,rows,queries):
    """Actual g/h arithmetic in PaddedProbeGraph._read, above native reading.

    There is one full physical block, regardless of cfg.row_chunk/query_chunk.
    Unlike the reference reader, the graph has no zero-result + AV addition.
    Two prefix-mask multiplies execute per physical score, including when the
    final inherited row has just been evicted before a behavior read.
    """
    if probe.stage=="f":
        return 0
    c,n,q=probe.cfg,int(rows),int(queries)
    lh,d=c.num_layers*c.num_heads,c.head_dim
    normalization=d if c.normalize else 0
    # g: normalize, MLP, output scale, q*gK reduction, masked score shift,
    # masked weights/reduction, gV multiply and final read addition.
    g=q*lh*(normalization+_head_mlp_flops(c,d)+6*d+4*n)
    h=0
    if probe.stage=="fgh":
        # h q-normalization is evaluated again; all physical (q,row) pairs
        # pass through the ordinary MLP and K/V output scaling.
        h=q*lh*(normalization+n*(_head_mlp_flops(c,d+c.row_dim)+6*d+1)+d)
    return int(g+h)


def scalar_behavior_flops(model,history_length):
    """One behavior read/write, including final norm and excluding score head."""
    width=model.cfg.hidden_size
    return int(native_query_flops(model,history_length=int(history_length),queries=1)
               -width-(2*width+1))


def required_graph_keys(probe,execution,*,window):
    """Operation/Q keys required by full deployment, independent of GPU replay.

    Query records retain inherited counts even for f, whose full-window reads
    also use a score graph. Corrected writes use a graph only when their cache
    was already full BEFORE pre-eviction; a short window becoming full remains
    eager. The final corrected write with zero surviving inherited rows still
    requires its append graph.
    """
    window=int(window)
    keys=set()
    for row in execution["read_shapes"]:
        if (row["kind"]=="query" and row["calls"] and
                row["history_length"]==window and row["inherited_rows"]>0):
            keys.add(("score",int(row["queries"])))
    if probe.stage!="f":
        corrected={(row["history_length"],row["inherited_rows"])
            for row in execution["read_shapes"] if row["kind"]=="behavior" and row["calls"]}
        if any(row["calls"] and row["before_length"]==window and row["width"]==1 and
               (window-1,row["inherited_rows"]) in corrected for row in execution["append_shapes"]):
            keys.add(("append",1))
    return sorted(keys)


def deployment_graph_setup(probe,execution,model,*,window=None):
    """Bill 2 warmups + 1 capture per required key, shared across ALL users.

    This is the deployment algorithm's one-pool startup charge. Repeated graph
    capture after restarting a quality-only worker is execution provenance, not
    another deployment pool. Allocation/copy bytes and measured setup seconds
    are separate quantities, never substituted for these forward FLOPs.
    """
    window=int(model.cfg.max_seq_len if window is None else window)
    parts=[]
    for operation,queries in required_graph_keys(probe,execution,window=window):
        rows=window if operation=="score" else window-1
        native=(native_query_flops(model,history_length=rows,queries=queries)
                if operation=="score" else scalar_behavior_flops(model,rows))
        extra=padded_read_extra_flops(probe,rows,queries)
        parts.append(dict(operation=operation,queries=queries,physical_read_rows=rows,
            warmup_forward_calls=2,capture_forward_calls=1,
            native_forward_flops_per_call=int(native),probe_forward_flops_per_call=int(extra),
            gross_forward_flops_per_call=int(native+extra),
            setup_native_flops=int(3*native),setup_probe_flops=int(3*extra),
            setup_flops=int(3*(native+extra))))
    return dict(required_graph_keys=[dict(operation=p["operation"],queries=p["queries"]) for p in parts],
        per_key=parts,shared_graph_setup_native_flops=sum(p["setup_native_flops"] for p in parts),
        shared_graph_setup_probe_flops=sum(p["setup_probe_flops"] for p in parts),
        shared_graph_setup_flops=sum(p["setup_flops"] for p in parts),
        pools=1,warmup_forward_calls_per_key=2,capture_forward_calls_per_key=1,
        scope="one frozen model/probe pool for all deployment users; required keys from complete CPU event ledger, including idle users and all behavior tails; gross native-plus-probe startup FLOPs")
