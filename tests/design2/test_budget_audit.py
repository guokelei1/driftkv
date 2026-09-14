import numpy as np
import pandas as pd

from design2.audit_budget import check_cost, curve, permutation_groups, permute, selected


def test_cost_prefix_and_uid_count():
    f=pd.DataFrame(dict(uid=[1,1,2,3],target=[1]*4,rebuild_flops=[2.,2.,4.,8.],
        max_abs_error=[.6,0.,.6,0.],mse=[1.,1.,1.,1.]))
    # Eight cost units select three states but only two unique UIDs.
    r=curve(f,-f.rebuild_flops.to_numpy(),budgets=np.array([.5]))[0]
    assert r["selected_states"]==3 and r["selected_uids"]==2
    assert r["recall"][0]==1
    np.testing.assert_equal(selected(np.array([0,1,2]),np.array([3.,5.,4.]),7),[0])


def test_permutation_preserves_target_length_and_uid_profiles():
    f=pd.DataFrame(dict(uid=[1,1,2,2,3,3],target=[1,1,1,1,3,3],count=[1024]*6,
        kind=["a","b"]*3,state_ordinal=[0,1]*3))
    x=np.arange(6.)
    for blocked in (False,True):
        groups=permutation_groups(f,blocked)
        y,donor=permute(x,groups,np.random.default_rng(17))
        np.testing.assert_equal(f.target.to_numpy()[donor],f.target)
        np.testing.assert_equal(np.sort(y),x)
        if blocked:
            assert y[1]-y[0]==y[3]-y[2]==y[5]-y[4]==1


def test_panel_charges_all_checks():
    c=check_cost()
    # Direct count of lower-triangular multiply/subtract plus diagonal divide.
    p=1281
    reference=16*36*sum(2*i+1 for i in range(p))
    assert c["panel_solve_only"]==reference
    assert c["panel_arithmetic"]>reference
