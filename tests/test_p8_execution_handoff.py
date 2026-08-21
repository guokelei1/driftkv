from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import adjudicate_p8_hs as hs  # noqa: E402
import run_p8_pipeline as pipeline  # noqa: E402
import train_p8_release as training  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_release_parent_lineage_is_frozen() -> None:
    assert training.parent_path("m0_f", 17, "r1_edge1").name == "theta0_selected.pt"
    assert training.parent_path("m1", 37, "r2").name == "theta0_selected.pt"
    edge2 = training.parent_path("m1", 71, "r1_edge2")
    assert str(edge2).endswith("results/p8/release_training/r1_edge1/m1_seed71/selected.pt")


def test_hs_metrics_compare_current_full_to_requested_alternative(tmp_path: Path) -> None:
    path = tmp_path / "F_fidelity.parquet"
    table = pa.table({
        "request_id": ["a", "b"], "uid": [1, 2],
        "current_full512_logit": [1.0, -1.0],
        "current_recent32_logit": [0.0, 0.0],
        "reuse_parent_kv_logit": [1.0, -1.0],
    })
    pq.write_table(table, path)
    h = hs.fidelity_records(path, "F", "current_recent32_logit")
    s = hs.fidelity_records(path, "F", "reuse_parent_kv_logit")
    assert all(row["output_js_divergence"] > 0 for row in h)
    assert all(row["output_js_divergence"] == 0 for row in s)


def test_pipeline_never_skips_r0_or_release_adjudication(tmp_path: Path) -> None:
    job, message = pipeline.next_job(tmp_path)
    assert job is None and message.startswith("BLOCKED: R0")
    write_json(tmp_path / "results/p8/r0_control/adjudication_v1.json", {"status": "R0_blocking_control_passed"})
    job, message = pipeline.next_job(tmp_path)
    assert message is None and (job.kind, job.release, job.model, job.seed) == ("train", "r1_edge1", "m0_f", 17)
    for candidate in pipeline.release_jobs("r1_edge1", tmp_path)[:12]:
        payload = {"admitted": True} if candidate.kind == "train" else {}
        write_json(candidate.artifact, payload)
    job, _ = pipeline.next_job(tmp_path)
    assert job is not None and job.kind == "seal"
    write_json(job.artifact, {})
    job, _ = pipeline.next_job(tmp_path)
    assert job is not None and job.kind == "adjudicate"
    write_json(job.artifact, {})
    job, _ = pipeline.next_job(tmp_path)
    assert job is not None and job.release == "r1_edge2" and job.kind == "train"


def test_pipeline_stops_on_rejected_release(tmp_path: Path) -> None:
    write_json(tmp_path / "results/p8/r0_control/adjudication_v1.json", {"status": "R0_blocking_control_passed"})
    jobs = pipeline.release_jobs("r1_edge1", tmp_path)
    for candidate in jobs[:6]:
        write_json(candidate.artifact, {"admitted": candidate.seed != 37})
    job, message = pipeline.next_job(tmp_path)
    assert job is None
    assert "admission failed" in message


def test_fidelity_schema_contract_excludes_quality_fields() -> None:
    import eval_p8_release_raw as raw

    forbidden = {"label", "target_index", "is_target", "feedback_history_stratum_v2"}
    assert not forbidden & set(raw.common_schema().names)
    assert forbidden <= set(raw.quality_schema().names)


def test_release_job_matrix_has_six_train_six_raw_and_two_gates(tmp_path: Path) -> None:
    jobs = pipeline.release_jobs("r2", tmp_path)
    assert [job.kind for job in jobs].count("train") == 6
    assert [job.kind for job in jobs].count("raw") == 6
    assert [job.kind for job in jobs].count("seal") == 1
    assert [job.kind for job in jobs].count("adjudicate") == 1


def test_next_wave_never_mixes_training_and_raw_jobs(tmp_path: Path) -> None:
    write_json(tmp_path / "results/p8/r0_control/adjudication_v1.json", {"status": "R0_blocking_control_passed"})
    wave, message = pipeline.next_wave(tmp_path)
    assert message is None and len(wave) == 6 and {job.kind for job in wave} == {"train"}
    for job in wave:
        write_json(job.artifact, {"admitted": True})
    wave, message = pipeline.next_wave(tmp_path)
    assert message is None and len(wave) == 6 and {job.kind for job in wave} == {"raw"}
