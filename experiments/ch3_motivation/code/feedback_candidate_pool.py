"""Causal feedback candidate fallback, confined to the fitting-user pool.

Own recent feedback may be empty. Filling the requested feedback quota from
other fitting users' pre-release observed items keeps the configured mixture
explicit. This helper never reads feedback labels or evaluation users.
It is an alternative candidate rule, not a change to any completed attempt.
"""
from __future__ import annotations

import numpy as np


def fitting_feedback_catalog(recent_items,fit_uids,known_items):
    return sorted({int(item) for uid in fit_uids for item in recent_items[str(uid)]
        if 1<=int(item)<=known_items})


def pooled_candidates(uid,own_recent,fit_catalog,known_items,*,count=16,recent_fraction=.5,seed=17):
    if not 0<=recent_fraction<=1 or not 0<count<=known_items:
        raise ValueError("candidate count/fraction outside the known catalog")
    quota=int(count*recent_fraction)
    values=[]
    for item in own_recent:
        item=int(item)
        if len(values)==quota:break
        if 1<=item<=known_items and item not in values:values.append(item)
    own=len(values)
    # Separate generators leave uniform distractors' seed independent of how
    # many fallback draws were required for this user's feedback quota.
    rng=np.random.default_rng(np.random.SeedSequence([seed,int(uid),28451]))
    available=[int(item) for item in fit_catalog if 1<=int(item)<=known_items and int(item) not in values]
    missing=min(quota-own,len(available))
    if missing:
        indices=rng.choice(len(available),size=missing,replace=False)
        values.extend(available[int(index)] for index in indices)
    fit_fallback=len(values)-own
    rng=np.random.default_rng(np.random.SeedSequence([seed,int(uid)]))
    seen=set(values)
    while len(values)<count:
        item=int(rng.integers(1,known_items+1))
        if item not in seen:values.append(item);seen.add(item)
    return values,dict(own_recent=own,other_fit_recent=fit_fallback,uniform=count-own-fit_fallback,
        requested_feedback_quota=quota,actual_feedback_quota=own+fit_fallback)
