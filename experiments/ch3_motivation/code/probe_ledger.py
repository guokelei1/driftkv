"""Full E5 deployment shapes without unnecessary post-last-score GPU replay.

Lengths, FIFO survival, batching and first consumers depend only on the frozen
event timeline. This CPU ledger includes idle users and the unscored suffix;
quality execution may stop earlier because edges are independent.
"""
from collections import Counter

import numpy as np


def full_probe_ledger(histories, by_user, uids, cfg, *, stage, padded=False):
    if stage not in ("f","fg","fgh"):
        raise ValueError("unknown probe stage")
    counts,initial,preparation,reads,appends,physical = (Counter() for _ in range(6))
    cap,band=cfg["history_length"],cfg["append_band_size"]
    start,end=cfg["release_day"]*86400,cfg["end_day"]*86400
    for uid in uids:
        times=histories.rows[uid][0]
        cursor=int(np.searchsorted(times,start,side="left"))
        n=min(cursor,cap)
        inherited=n
        initial[n]+=1
        counts["initial_native_rows"]+=n
        prepared=False
        rounds=by_user.get(uid,{})
        for until in sorted(set(rounds)|{end}):
            stop=int(np.searchsorted(times,until,side="left"))
            if cursor<stop and not prepared:
                preparation[n]+=1
                counts["prepare_calls"]+=1
                counts["prepare_rows"]+=n
                prepared=True
            while cursor<stop:
                if stage=="f" or inherited==0:
                    width=1<<(min(band,stop-cursor).bit_length()-1)
                    dropped=max(0,n+width-cap)
                    appends[n,width,0]+=1
                    counts["native_band_append_calls"]+=1
                    counts["appended_rows"]+=width
                    inherited=max(0,inherited-dropped)
                    cursor+=width
                    n=min(cap,n+width)
                else:
                    before=n
                    inherited=max(0,inherited-max(0,n+1-cap))
                    length=min(before,cap-1)
                    reads["behavior",length,inherited,1]+=1
                    appends[before,1,inherited]+=1
                    counts["corrected_append_calls"]+=1
                    counts["appended_rows"]+=1
                    if padded and before==cap:
                        physical["behavior",length,inherited,1]+=1
                    cursor+=1
                    n=min(cap,n+1)
            if until not in rounds:
                continue
            if not prepared:
                preparation[n]+=1
                counts["prepare_calls"]+=1
                counts["prepare_rows"]+=n
                prepared=True
            values=rounds[until]
            for offset in range(0,len(values),cfg["query_batch"]):
                q=min(cfg["query_batch"],len(values)-offset)
                reads["query",n,inherited,q]+=1
                counts["candidate_reads"]+=q
                counts["candidate_batches"]+=1
                if padded and stage!="f" and inherited>0 and n==cap:
                    physical["query",n,inherited,q]+=1
    return dict(counts=dict(counts),
        initial_shapes=[dict(history_length=n,calls=c) for n,c in sorted(initial.items())],
        prepare_shapes=[dict(history_length=n,calls=c) for n,c in sorted(preparation.items())],
        read_shapes=[dict(kind=k,history_length=n,inherited_rows=m,queries=q,calls=c)
            for (k,n,m,q),c in sorted(reads.items())],
        append_shapes=[dict(before_length=n,width=w,inherited_rows=m,calls=c)
            for (n,w,m),c in sorted(appends.items())],
        padded_read_shapes=[dict(kind=k,physical_read_rows=n,inherited_rows=m,queries=q,calls=c)
            for (k,n,m,q),c in sorted(physical.items())],
        scope="complete CPU event ledger, including idle users and all unscored behavior tails; independent-edge quality shortcut does not reduce billed work")
