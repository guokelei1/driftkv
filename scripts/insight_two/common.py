from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT
    / "configs/contracts/yambda500m_medium_hstu_native_insight2_functional_boundary_v1.yaml"
)
INPUT_MANIFEST = ROOT / "data/manifests/yambda500m_medium_insight1_locality_v1"
DATASET = ROOT / "data/processed/yambda500m_unified_v1/scales/medium/dataset.json"
RESULT_ROOT = ROOT / "results/yambda500m_medium_seed17/insight2_functional_boundary_v1"

DAY = 86_400
CUTOVER_DAYS = (231, 245, 259, 273, 287)
EDGES = tuple(f"v{index}_to_v{index + 1}" for index in range(5))
HISTORY = 1024
KNOWN_ITEMS = 1_380_509
OOV_BUCKETS = 256
POPULATION = 3000
CANARY_USERS = 32
CANDIDATES = 64
ANCHOR_INDICES = tuple(range(0, CANDIDATES, 2))
HELDOUT_INDICES = tuple(range(1, CANDIDATES, 2))

STAGE_PRESENTATION = {
    "kv_prefix_contribution": "S3_positionwise_response_summed_for_injection",
    "av_aggregation": "S4_aggregated_context",
    "u_gated_update": "S5_transformed_update",
    "layer_hidden": "S6_post_block_residual",
    "final_readout": "S7_final_representation",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(version: int) -> Path:
    base = RESULT_ROOT.parent / "full_reuse_matrix_v1"
    if version == 0:
        return base / "shared_v0/checkpoint_100.pt"
    if version == 5:
        return base / "D14/v5_extension_v1/checkpoint/checkpoint_100.pt"
    return base / f"D14/checkpoints/v{version}/checkpoint_100.pt"


def verify_contract() -> dict[str, Any]:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    if contract["scope"]["edges"] != list(EDGES):
        raise RuntimeError("contract edge order differs")
    if tuple(contract["candidate_split"]["anchor_indices"]) != ANCHOR_INDICES:
        raise RuntimeError("contract anchor split differs")
    if tuple(contract["candidate_split"]["heldout_indices"]) != HELDOUT_INDICES:
        raise RuntimeError("contract held-out split differs")
    frozen = contract["frozen_inputs"]
    for name, record in frozen.items():
        if name == "checkpoints":
            continue
        path = ROOT / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"frozen input differs from contract: {name}")
    for version in range(6):
        record = frozen["checkpoints"][f"v{version}"]
        path = ROOT / record["path"]
        if path != checkpoint(version):
            raise RuntimeError(f"checkpoint path differs for v{version}")
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"checkpoint v{version} differs from contract")
    return contract


def load_frozen_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    descriptor = json.loads((INPUT_MANIFEST / "manifest.json").read_text(encoding="utf-8"))
    for name, record in descriptor["artifacts"].items():
        path = INPUT_MANIFEST / name
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"input-manifest artifact differs: {name}")
    population = np.load(INPUT_MANIFEST / "population.npz", allow_pickle=False)
    panels = np.load(INPUT_MANIFEST / "candidate_panels.npz", allow_pickle=False)
    uids = population["uids"].astype(np.int64, copy=False)
    candidates = panels["candidates"].astype(np.int64, copy=False)
    modes = panels["modes"].astype(np.uint8, copy=False)
    if uids.shape != (POPULATION,):
        raise RuntimeError("frozen population shape differs")
    if candidates.shape != (len(EDGES), POPULATION, CANDIDATES):
        raise RuntimeError("candidate panel shape differs")
    if modes.shape != candidates.shape or not np.array_equal(panels["uids"], uids):
        raise RuntimeError("candidate panel lineage differs")
    if np.any(candidates[:, :, ANCHOR_INDICES] == candidates[:, :, HELDOUT_INDICES]):
        # Element-wise equality is not sufficient to detect all overlap, but any
        # equality here is already a violation worth stopping on.
        raise RuntimeError("anchor and held-out panels overlap at paired positions")
    return uids, candidates, modes


def verify_model_payload(payload: dict[str, Any]) -> None:
    expected = {
        "hidden_size": 192,
        "num_layers": 6,
        "num_heads": 6,
        "max_seq_len": 1024,
    }
    for name, value in expected.items():
        if int(payload["config"][name]) != value:
            raise RuntimeError(f"checkpoint {name} differs: {payload['config'][name]}")


def _bernoulli_js(left_logits: torch.Tensor, right_logits: torch.Tensor) -> torch.Tensor:
    left = torch.sigmoid(left_logits.float()).clamp(1e-7, 1 - 1e-7)
    right = torch.sigmoid(right_logits.float()).clamp(1e-7, 1 - 1e-7)
    middle = 0.5 * (left + right)

    def kl(probability: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return probability * torch.log(probability / reference) + (
            1 - probability
        ) * torch.log((1 - probability) / (1 - reference))

    return 0.5 * (kl(left, middle) + kl(right, middle)).mean(dim=1)


def _rank_correlation(reference: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    if reference.shape != values.shape or reference.ndim != 2:
        raise ValueError("rank correlation requires equal [batch,candidate] tensors")
    reference_rank = torch.argsort(torch.argsort(reference.float(), dim=1), dim=1).float()
    value_rank = torch.argsort(torch.argsort(values.float(), dim=1), dim=1).float()
    reference_rank -= reference_rank.mean(dim=1, keepdim=True)
    value_rank -= value_rank.mean(dim=1, keepdim=True)
    numerator = (reference_rank * value_rank).sum(dim=1)
    denominator = reference_rank.square().sum(dim=1).sqrt() * value_rank.square().sum(
        dim=1
    ).sqrt()
    return numerator / denominator.clamp_min(1e-20)


def score_metrics(
    exact: torch.Tensor,
    reuse: torch.Tensor,
    observed: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if exact.shape != reuse.shape or exact.shape != observed.shape or exact.ndim != 2:
        raise ValueError("score metrics require equal [batch,candidate] tensors")
    exact = exact.float()
    reuse = reuse.float()
    observed = observed.float()
    exact_probability = torch.sigmoid(exact)
    reuse_logit_gap = torch.abs(reuse - exact).mean(dim=1)
    observed_logit_gap = torch.abs(observed - exact).mean(dim=1)
    reuse_probability_gap = torch.abs(torch.sigmoid(reuse) - exact_probability).mean(dim=1)
    observed_probability_gap = torch.abs(
        torch.sigmoid(observed) - exact_probability
    ).mean(dim=1)
    width = exact.shape[1]
    topk = min(10, width)
    exact_top = torch.topk(exact, topk, dim=1).indices
    observed_top = torch.topk(observed, topk, dim=1).indices
    overlap = (
        exact_top[:, :, None] == observed_top[:, None, :]
    ).any(dim=2).float().mean(dim=1)
    return {
        "reuse_logit_gap": reuse_logit_gap,
        "observed_logit_gap": observed_logit_gap,
        "logit_gap_recovery": 1.0
        - observed_logit_gap / reuse_logit_gap.clamp_min(1e-20),
        "reuse_probability_gap": reuse_probability_gap,
        "observed_probability_gap": observed_probability_gap,
        "probability_gap_recovery": 1.0
        - observed_probability_gap / reuse_probability_gap.clamp_min(1e-20),
        "bernoulli_js_to_exact": _bernoulli_js(observed, exact),
        "top1_agreement": (observed.argmax(dim=1) == exact.argmax(dim=1)).float(),
        "top10_overlap": overlap,
        "rank_correlation": _rank_correlation(exact, observed),
    }


def metrics_row(metrics: dict[str, torch.Tensor], index: int = 0) -> dict[str, float]:
    return {name: float(values[index].detach()) for name, values in metrics.items()}
