from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from hstu_kvcache.data.p7_training import P7Request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_scale_8l_queue as queue
import scale_8l_common as scale


def request(length: int) -> P7Request:
    return P7Request(
        request_id="scale-test", workload="F", uid=1, query_timestamp=10_000,
        history_items=np.ones(length, dtype=np.int64),
        history_behaviors=np.ones(length, dtype=np.int64),
        history_time_deltas=np.zeros(length, dtype=np.float32),
        query_time_delta=1.0,
        candidate_ids=np.asarray([2], dtype=np.int64),
        base_features=np.zeros((1, 7), dtype=np.float32),
        target_index=None, label=1, request_weight=1.0,
    )


def test_scale_request_supports_1024_without_relaxing_beyond_contract() -> None:
    assert len(request(1024).history_items) == 1024
    with pytest.raises(ValueError):
        request(1025)


def test_scale_contract_freezes_architecture_method_and_four_gpus() -> None:
    contract = yaml.safe_load((ROOT / "configs/contracts/scale_8l_v1.yaml").read_text())
    assert contract["architecture"] == {
        "layers": 8, "hidden_size": 256, "heads": 8, "context": 1024,
        "recent_control": 32, "temporal_num_freqs": 16,
        "block_variant": "legacy", "training_history": "Full1024_only",
    }
    assert contract["execution"]["GPU_allowlist"] == [0, 1, 2, 3]
    assert contract["frozen_method"]["actions"] == [
        "noop", "layer0_recent128", "layer0_middle", "layer0_full",
        "hybrid_tail128", "exact_all",
    ]
    assert contract["frozen_method"]["predictor"] == "StandardScaler_plus_Ridge_alpha_1"


def test_scale_queue_contains_all_seeds_and_rejects_other_gpus() -> None:
    theta = queue.model_jobs("theta0")
    assert {(job.model, job.seed) for job in theta} == {
        (model, seed) for model in ("m0_f", "m1") for seed in (17, 37, 71)
    }
    releases = queue.model_jobs("release", "r1_edge1")
    assert len(releases) == 6
    assert "scripts/train_scale_8l_fsdp_release.py" in releases[0].command(3)
    assert releases[0].command(3)[:3] == ["torchrun", "--standalone", "--nproc_per_node=4"]
    with pytest.raises(ValueError):
        releases[0].command(4)


def test_scale_contract_input_hashes_resolve() -> None:
    assert scale.contract()["contract"] == "scale_8l_v1"


def test_scale_queue_stages_pilot_before_replication() -> None:
    source = (ROOT / "scripts/run_scale_8l_queue.py").read_text()
    assert "S3_PILOT_THETA0_M0_F_SEED17" in source
    assert "pilot/s3_m0_f_seed17_h_adjudication.json" in source
    assert "pilot/replication_authorization.json" in source
    assert source.index("S3_PILOT_THETA0_M0_F_SEED17") < source.index("S3_REPLICATION_THETA0_FSDP_ALL_GPUS")
