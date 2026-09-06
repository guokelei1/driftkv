"""The larger cohort bootstrap must retain the original paired UID statistic."""

import numpy as np
import pandas as pd
from design.quality_report import NAMES, user_bootstrap

from hstu_kvcache.evaluation.binary_metrics import binary_metrics


def test_counted_bootstrap_equals_repeated_users_with_ties_and_single_classes():
    rows = pd.DataFrame(dict(uid=[8, 8, 8, 1, 1, 20, 20], label=[1, 1, 1, 0, 1, 0, 0],
        exact=[1000., 100., 1., -1000., .2, .2, .4],
        reuse=[100., 1000., .2, .2, 1., 1., -1000.],
        learned=[1., 1., .4, .2, .2, -1000., 1.]))
    actual = user_bootstrap(rows, 100)
    groups = list(rows.groupby("uid", sort=True).indices.values())
    rng, samples = np.random.default_rng(17), []
    for _ in range(100):
        indices = np.concatenate([groups[i] for i in rng.integers(0, len(groups), size=len(groups))])
        sampled = rows.iloc[indices]
        auc = {name:binary_metrics(sampled.label.to_numpy(), sampled[name].to_numpy())["ROC_AUC"] for name in NAMES}
        if auc["exact"] is not None:
            samples.append([auc["exact"]-auc["reuse"], auc["learned"]-auc["reuse"], auc["learned"]-auc["exact"]])
    assert actual["valid_samples"] == len(samples) < 100
    np.testing.assert_allclose(list(actual["auc_difference_95ci"].values()),
                               np.quantile(samples, [.025, .975], axis=0).T, rtol=0, atol=1e-14)
