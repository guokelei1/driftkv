"""Complete extra preparation/read/startup accounting for projected execution.

The underlying nonlinear h and its parameters are unchanged. This module
charges the different execution: one row-side first-affine projection when a
user first needs a full-window graph, plus the entire remaining MLP on every
physical query/row pair. It never uses the quality shortcut's smaller ledger.
"""
from collections import Counter

from experiments.ch3_motivation.code.probe_cost import deployment_graph_setup, required_graph_keys
from experiments.ch3_motivation.code.probe_ledger import full_probe_ledger
from experiments.ch3_motivation.code.probe_projected_fast import (
    projected_read_extra_flops, projected_row_preparation_flops,
)


def deployment_projected_graph_setup(probe,execution,model,*,window=None):
    """Gross native+projected forward work, 2 warmups+1 capture per full key."""
    raw=deployment_graph_setup(probe,execution,model,window=window)
    parts=[]
    for row in raw["per_key"]:
        extra=projected_read_extra_flops(probe,row["physical_read_rows"],row["queries"])
        native=row["native_forward_flops_per_call"]
        parts.append(dict(row,probe_forward_flops_per_call=extra,
            gross_forward_flops_per_call=native+extra,setup_probe_flops=3*extra,
            setup_flops=3*(native+extra)))
    return dict(raw,per_key=parts,
        shared_graph_setup_probe_flops=sum(row["setup_probe_flops"] for row in parts),
        shared_graph_setup_flops=sum(row["setup_flops"] for row in parts),
        scope="one frozen model/probe projected pool; complete ledger keys; gross native+projected forward warmup/capture; user row projection prepared separately")


def full_projected_preparation(probe,histories,by_user,uids,cfg):
    """Bill each user's row map once if its COMPLETE trajectory needs a graph.

    A short user may become full. Before its first full graph no inherited
    row can have been evicted: the first projection therefore covers its
    initial inherited row count. A user ending exactly on its first full
    state, without a full-state read/append, does not need the projected table.
    """
    histogram=Counter()
    if probe.h is not None:
        for uid in uids:
            ledger=full_probe_ledger(histories,by_user,[uid],cfg,stage=probe.stage,padded=True)
            if required_graph_keys(probe,ledger,window=cfg["history_length"]):
                histogram[ledger["initial_shapes"][0]["history_length"]]+=1
    c=probe.cfg
    element=next(probe.parameters()).element_size()
    calls=sum(histogram.values())
    window=cfg["history_length"]
    return dict(extra_row_preparation_flops=sum(count*projected_row_preparation_flops(probe,n)
                for n,count in histogram.items()),
        projected_users=calls,projected_rows=sum(n*count for n,count in histogram.items()),
        preparation_shapes=[dict(inherited_rows=n,calls=count) for n,count in sorted(histogram.items())],
        per_full_window_projected_table_bytes=(c.num_layers*c.num_heads*window*c.hidden_width*element if probe.h is not None else 0),
        all_projected_users_full_window_table_bytes=(calls*c.num_layers*c.num_heads*window*c.hidden_width*element),
        projection_table_to_native_kv_ratio=c.hidden_width/(2*c.head_dim) if probe.h is not None else 0.,
        scope="complete causal event ledger, including idle users with behavior tails; row projection once at first full graph; hypothetical all-active table bytes, separate from measured one-user GPU execution and raw-feature/source/cache state")


def projected_padded_read_delta(probe,execution,raw_padded_read_extra_flops):
    """Signed replacement of already billed raw padded reads, not another add."""
    delta=0
    physical=0
    for row in execution.get("padded_read_shapes",[]):
        n,q=row["physical_read_rows"],row["queries"]
        new=projected_read_extra_flops(probe,n,q)
        old=raw_padded_read_extra_flops(probe,n,q)
        delta+=row["calls"]*(new-old)
        physical+=row["calls"]*new
    return dict(projected_minus_raw_padded_read_flops=int(delta),
        projected_physical_padded_read_flops=int(physical),
        scope="replace each complete physical graph-read correction cost; exact short-window reference reads remain unchanged")
