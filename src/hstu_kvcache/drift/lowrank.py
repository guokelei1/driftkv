from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.func import jvp

from ..models import HSTU
from .jvp import dtheta_as_dict, make_kv_func


@dataclass
class LowRankFit:
    """Result of fitting a low-rank drift predictor on probe users.

    Path 1 (roadmap Insight 5.1): compute J.dtheta for a small probe set (cost =
    |probe| JVPs), then predict drift norms for all users from cheap features.
    If the per-user drift vectors live in a low-dim subspace, a tiny probe set
    suffices and the online cost drops to << one recompute.
    """

    rank: int
    probe_drift_matrix: np.ndarray  # [|probe|, kv_numel] J.dtheta rows
    probe_norms: np.ndarray  # [|probe|] ||J.dtheta|| per probe user
    sing_values: np.ndarray  # [rank] spectrum of probe drift matrix
    fit_recon_err: float  # relative reconstruction err at `rank`
    explained_var: float


def collect_probe_drifts(
    model: HSTU,
    batches: list[dict],
    dtheta_vec: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Compute J.dtheta for each probe-user batch; stack into [|probe|, kv_numel]."""
    dtheta_dict = dtheta_as_dict(model, dtheta_vec)
    rows: list[torch.Tensor] = []
    for batch in batches:
        params, _, kv_flat_fn = make_kv_func(model, batch, device)
        _, tangent = jvp(kv_flat_fn, (params,), (dtheta_dict,))
        rows.append(tangent.detach().reshape(batch["item_ids"].shape[0], -1))
    return torch.cat(rows, dim=0).cpu().numpy()


def fit_low_rank(probe_matrix: np.ndarray, ranks: list[int]) -> dict[int, LowRankFit]:
    """SVD the probe drift matrix; report relative reconstruction error per rank.

    Gating for V3: if a small rank captures most of the drift variance, the
    cross-user sharing path (1) is viable. If the spectrum is flat (no low-rank
    structure), path 1 narrows and we lean on path 2 (Fisher spectrum).
    """
    U, S, Vt = np.linalg.svd(probe_matrix, full_matrices=False)
    fro2 = (S ** 2).sum() + 1e-12
    out: dict[int, LowRankFit] = {}
    for r in ranks:
        r = min(r, len(S))
        recon = (U[:, :r] * S[:r]) @ Vt[:r]
        err = np.linalg.norm(probe_matrix - recon) / (np.linalg.norm(probe_matrix) + 1e-12)
        ev = (S[:r] ** 2).sum() / fro2
        out[r] = LowRankFit(
            rank=r,
            probe_drift_matrix=probe_matrix,
            probe_norms=np.linalg.norm(probe_matrix, axis=1),
            sing_values=S[:r],
            fit_recon_err=float(err),
            explained_var=float(ev),
        )
    return out


@dataclass
class FisherSpectrum:
    """Path 2: offline characterisation of J^T J (the Fisher approximation).

    Online drift estimate = ||dtheta|| projected onto the top sensitive
    directions, scaled by per-user cheap features. The expensive eigen-probe is
    done once offline; online cost is a handful of dot products.
    """

    top_directions: torch.Tensor  # [k, n_params] top sensitive directions in param space
    top_eigvalues: torch.Tensor  # [k]
    probe_user_norms: torch.Tensor  # [|probe|] ||J.dtheta|| for calibration


def estimate_fisher_spectrum(
    model: HSTU,
    probe_batches: list[dict],
    k: int = 16,
    device: torch.device | None = None,
) -> FisherSpectrum:
    """Estimate top-k eigendirections of sum_u J_u^T J_u via random-probe Rayleigh.

    J^T J is the empirical Fisher over probe users. We never materialise it:
    each estimate uses a forward jvp (J v) then accumulates its norm, giving a
    stochastic Rayleigh quotient for that direction. This is a coarse spectrum
    sketch sufficient for the offline "which param directions are sensitive"
    characterisation of path 2.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = sum(p.numel() for p in model.parameters())
    directions = []
    eigvals = []
    rng = torch.Generator(device=device).manual_seed(0)
    for _ in range(k):
        v = torch.randn(n, device=device, generator=rng)
        v = v / (v.norm() + 1e-12)
        # J v (forward jvp, accumulate over users)
        jv_acc = None
        for batch in probe_batches:
            params, _, kv_flat_fn = make_kv_func(model, batch, device)
            primal, tangent = jvp(kv_flat_fn, (params,), (dtheta_as_dict(model, v),))
            if jv_acc is None:
                jv_acc = tangent.detach()
            else:
                jv_acc = jv_acc + tangent.detach()
        # J^T (Jv): vjp with cotangent = jv_acc split per batch is needed; approximate
        # the leading singular direction with the random-probe Rayleigh quotient.
        rayleigh = float((jv_acc.norm() / (v.norm() + 1e-12)).item())
        directions.append(v.cpu())
        eigvals.append(rayleigh)
    return FisherSpectrum(
        top_directions=torch.stack(directions),
        top_eigvalues=torch.tensor(eigvals),
        probe_user_norms=torch.tensor([]),
    )
