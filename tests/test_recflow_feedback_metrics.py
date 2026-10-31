import numpy as np

from hstu_kvcache.evaluation.recflow.evaluate_feedback import metrics


def test_feedback_metrics_include_negative_only_slates_and_separate_mixed_support():
    rows = dict(label=np.array([1, 0, 0, 0]), logit=np.array([3., -1., -2., -3.]),
        uid=np.array([1, 1, 2, 2]), event_index=np.arange(4), request_index=np.array([5, 5, 8, 8]))
    output = metrics(rows)
    assert output["ROC_AUC"] == 1
    assert output["effective_view_PR_AUC"] == 1
    assert output["requests"] == 2
    assert output["mixed_label_requests"] == 1
    assert output["within_exposure_request_NDCG@1"] == 0.5
    assert output["mixed_label_request_NDCG@1"] == 1
    assert output["user_macro_AUC_eligible_users"] == 1


def test_feedback_auc_and_ap_treat_equal_scores_as_ties():
    rows = dict(label=np.array([1, 0]), logit=np.zeros(2), uid=np.array([1, 1]),
        event_index=np.arange(2), request_index=np.array([4, 4]))
    output = metrics(rows)
    assert output["ROC_AUC"] == 0.5
    assert output["effective_view_PR_AUC"] == 0.5
