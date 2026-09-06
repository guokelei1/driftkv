"""The calibration selector must actually exclude complete users."""

import torch
from design.fit_adaptive_ridge import MULTIPLIERS, select_ridge


def test_group_ridge_residuals_match_explicit_user_deletion():
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(17, 7, generator=generator, dtype=torch.float64)
    x = x-x.mean(0)
    y = torch.randn(17, 3, generator=generator, dtype=torch.float64)+2
    uids = [1]*3+[2]*5+[3]*2+[4]*7
    gram = x @ x.T
    _, selected = select_ridge(gram, y, uids)
    for multiplier in MULTIPLIERS:
        penalty = gram.trace()/len(x)*multiplier
        errors = []
        for uid in set(uids):
            keep = torch.tensor([v != uid for v in uids])
            train, response = x[keep], y[keep]
            center, mean = train.mean(0), response.mean(0)
            train = train-center
            weights = torch.linalg.solve(train.T @ train+penalty*torch.eye(x.shape[1], dtype=x.dtype), train.T @ (response-mean))
            predicted = (x[~keep]-center) @ weights+mean
            errors.append((predicted-y[~keep]).square().mean())
        torch.testing.assert_close(torch.tensor(selected["leave_user_out_mse"][str(multiplier)], dtype=x.dtype),
                                   torch.stack(errors).mean(), atol=1e-9, rtol=1e-9)
