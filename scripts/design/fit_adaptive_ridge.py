#!/usr/bin/env python3
"""Choose release regularization by exact leave-user-out ridge residuals.

Selection precedes the existing rank-32 compression. It is a calibration
response proxy, not validation of final logits. The execution pipeline and
inference weight format are unchanged.
"""

import argparse
import json
from collections import defaultdict
from types import SimpleNamespace

import torch
from design import run
from design.data import ROOT

MULTIPLIERS = (.001, .01, .1, 1., 10.)


def select_ridge(gram, y, uids):
    """Delete all scenes and query-time copies of each UID together."""
    groups = defaultdict(list)
    for i, uid in enumerate(uids):
        groups[uid].append(i)
    assert len(groups) > 1 and len(uids) == len(gram)
    sizes = [len(g) for g in groups.values()]
    index = torch.tensor([g+[0]*(max(sizes)-len(g)) for g in groups.values()], device=gram.device)
    valid = torch.arange(max(sizes), device=gram.device)[None, :] < torch.tensor(sizes, device=gram.device)[:, None]
    pair_mask = valid[:, :, None] & valid[:, None, :]
    eigenvalues, vectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.clamp_min(0)
    group_vectors = vectors[index]
    mean = y.mean(0)
    projected_y = vectors.T @ (y-mean)
    base = (gram.trace()/len(gram)).clamp_min(1e-6)
    errors = []
    for multiplier in MULTIPLIERS:
        regularizer = (base*multiplier).clamp_min(1e-6)
        shrink = eigenvalues/(eigenvalues+regularizer)
        predicted = vectors @ (shrink[:, None]*projected_y)+mean
        # The unpenalized intercept contributes 1/n to the full hat matrix.
        hat = ((group_vectors*shrink) @ group_vectors.transpose(1, 2)+1/len(gram))*pair_mask
        residual = (y-predicted)[index]*valid[:, :, None]
        deleted = torch.linalg.solve(torch.eye(max(sizes), device=gram.device, dtype=gram.dtype)-hat, residual)
        errors.append(((deleted.square().mean(-1)*valid).sum(1)/valid.sum(1)).mean())
    errors = torch.stack(errors)
    selected = int(errors.argmin())
    regularizer = (base*MULTIPLIERS[selected]).clamp_min(1e-6)
    alpha = vectors @ (projected_y/(eigenvalues+regularizer)[:, None])
    return alpha, dict(multiplier=MULTIPLIERS[selected], regularizer=float(regularizer),
        leave_user_out_mse={str(k):float(v) for k, v in zip(MULTIPLIERS, errors, strict=True)},
        fitting_users=len(groups), selected_proxy="untruncated ridge; UID-equal response MSE with a free intercept")


@torch.no_grad()
def fit_selected(mapper, features, rates, uids):
    mapper.center.copy_(features.mean(0))
    mapper.input_scale.copy_(features.std(0).clamp_min(1e-3))
    x = (features-mapper.center)/mapper.input_scale
    scale = rates.square().mean((0, 2)).sqrt().clamp_min(1e-7).repeat_interleave(mapper.width)
    y = rates.flatten(1)/scale
    gram = x @ x.T
    alpha, selection = select_ridge(gram, y, uids)
    fitted = gram @ alpha
    _, singular, vh = torch.linalg.svd(fitted, full_matrices=False)
    rank = min(mapper.rank, len(vh))
    basis = vh[:rank]
    mapper.encode.zero_()
    mapper.decode.zero_()
    mapper.encode[:, :rank] = x.T @ alpha @ basis.T
    mapper.decode[:rank] = basis*scale
    mapper.offset.copy_(y.mean(0)*scale)
    if mapper.temporal:
        slopes = (mapper.encode[mapper.base_inputs:]/mapper.input_scale[mapper.base_inputs:, None]) @ mapper.decode
        mapper.time_weights.copy_(slopes.reshape_as(mapper.time_weights))
    stride = mapper.layers*2*mapper.width+1
    mapper.supported_producers = {p for p in range(mapper.producer_count)
                                  if bool((features[:, (p+1)*stride-1] > 0).any())}
    predicted = x @ mapper.encode @ mapper.decode+mapper.offset
    return dict(effective_rank=rank, regularization_selection=selection,
        centered_prediction_energy_retained=float(singular[:rank].square().sum()/singular.square().sum().clamp_min(1e-20)),
        normalized_fit_mse=float((predicted/scale-y).square().mean()),
        supported_producers=sorted(mapper.supported_producers))


def main(cli):
    config = json.loads((ROOT / "results/design/v9_stratified_stable64_01/configuration.json").read_text())
    args = SimpleNamespace(**config)
    args.run_id = cli.run_id
    args.fit_users = cli.users
    args.lifetime_users = cli.users//4
    args.targets = cli.targets
    args.source_confidence = False
    args.second_moments = False
    args.ridge_selection = dict(multipliers=MULTIPLIERS, grouping="UID, all scenes and time copies together",
        rule="minimum leave-user-out normalized response MSE before rank32 compression, recomputed each corrected-query pass",
        scope="fitting-only release adaptation; no development labels, extra teacher users or model-ID branches")
    args.prospective_resources = dict(expected_seconds=[30, 120] if cli.users <= 16 else [60, 240],
        expected_gpu_mib=16000, estimate_basis="stratified64 fixed-ridge44s plus one eigendecomposition and five grouped residual solves per pass")
    original = run.fit_ridge_batched

    def fit_with_uids(current, scenes, mapper, options):
        uids = [s.uid for s in scenes for _ in range(4 if mapper.temporal else 1)]
        # A process-local fitting experiment: reuse all existing data, teacher,
        # continuous replay, canary, serialization and evaluation orchestration.
        mapper.fit = lambda features, rates: fit_selected(mapper, features, rates, uids)
        return original(current, scenes, mapper, options)

    run.fit_ridge_batched = fit_with_uids
    run.main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--users", type=int, choices=(16, 64), default=64)
    parser.add_argument("--targets", type=int, choices=(1, 2, 3, 4, 5), default=5)
    main(parser.parse_args())
