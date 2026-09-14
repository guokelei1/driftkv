import numpy as np
import pandas as pd

from design2.report_detection import auc, weights


def test_weighted_auc_with_ties_matches_all_pairs():
    y=np.array([True,False,True,False,False])
    score=np.array([.5,.5,.9,.2,.8])
    w=np.array([1.,2.,3.,4.,1.])
    numerator=denominator=0.
    for i in np.flatnonzero(y):
        for j in np.flatnonzero(~y):
            numerator += w[i]*w[j]*(float(score[i]>score[j])+.5*float(score[i]==score[j]))
            denominator += w[i]*w[j]
    assert abs(auc(y,score,w)-numerator/denominator)<1e-12


def test_uid_weight_not_inflated_by_more_scenes():
    frame=pd.DataFrame(dict(uid=[1,1,1,2,2],target=[1,1,3,1,3]))
    w=weights(frame)
    np.testing.assert_allclose(w,[.25,.25,.5,.5,.5])
    assert w[frame.uid==1].sum()==w[frame.uid==2].sum()==1
