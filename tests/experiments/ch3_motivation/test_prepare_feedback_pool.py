import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
from experiments.ch3_motivation.code.prepare_feedback_pool import prepare
from experiments.ch3_motivation.code.prepare_candidates import candidates_for
from hstu_kvcache.common.artifacts import sha256


def test_preparation_changes_only_calibration_candidates(tmp_path):
    source=tmp_path/"original";source.mkdir()
    population=dict(fit=list(range(1,65)),validation=[900],main=[901])
    (source/"population.json").write_text(json.dumps(population))
    for edge in range(1,6):
        folder=source/f"edge_{edge}";folder.mkdir()
        for name in ("histories.npz","requests.parquet"):
            (folder/name).write_bytes(f"frozen-{name}-{edge}".encode())
        binding=dict(population_sha256=sha256(source/"population.json"),
            history=dict(path="histories.npz",sha256=sha256(folder/"histories.npz")),
            requests=dict(path="requests.parquet",sha256=sha256(folder/"requests.parquet")))
        (folder/"inputs.json").write_text(json.dumps(binding))
        recent={str(uid):[uid] if uid<=32 else [] for uid in population["fit"]}
        recent.update({"900":[999],"901":[1000]})
        candidates=dict(population_sha256=binding["population_sha256"],known_items=1000,recent_items=recent)
        (folder/"calibration_feedback_candidates.json").write_text(json.dumps(candidates))
    original_hashes={str(p.relative_to(source)):sha256(p) for p in source.rglob("*") if p.is_file()}
    output=tmp_path/"pooled"
    result=prepare(source,output,64)
    assert len(result)==5 and all(x["all_users_have_16_observed_candidates"] for x in result)
    assert (output/"population.json").read_bytes()==(source/"population.json").read_bytes()
    for edge in range(1,6):
        folder=output/f"edge_{edge}"
        assert (folder/"histories.npz").is_symlink()
        assert (folder/"inputs.json").read_bytes()==(source/f"edge_{edge}/inputs.json").read_bytes()
        view=json.loads((folder/"calibration_feedback_candidates.json").read_text())
        assert view["fitting_feedback_catalog"]==list(range(1,33))
        assert set(view["recent_items"])==set(map(str,population["fit"]+population["validation"]))
        assert not {999,1000}&set(view["fitting_feedback_catalog"])
        for uid in population["fit"]:
            chosen=candidates_for(uid,view["recent_items"][str(uid)],1000,count=16,recent_fraction=.5)
            assert len(chosen)==len(set(chosen))==16
            assert set(chosen[:8])<=set(range(1,33))
    assert {str(p.relative_to(source)):sha256(p) for p in source.rglob("*") if p.is_file()}==original_hashes
    assert prepare(source,output,64)==result
