"""Current-chain resolution using only tiny fake metadata and weight files."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from design import competitor_models as models


@pytest.fixture
def model_chains(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "ROOT", tmp_path)

    def write(path, text):
        path = tmp_path / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return str(path.relative_to(tmp_path)), hashlib.sha256(text.encode()).hexdigest()

    for scale, layers, known in (("medium", 6, 20), ("large", 10, 30)):
        mapping, mapping_hash = write(f"data/{scale}/item_mapping.parquet", "fake mapping")
        dataset, dataset_hash = write(f"data/{scale}/dataset.json", json.dumps({
            "scale": scale, "foundation_items": known, "users_path": "users.parquet",
            "item_mapping_path": "item_mapping.parquet",
        }))
        contract, contract_hash = write(f"configs/{scale}.yaml", yaml.safe_dump({
            "frozen_inputs": {
                "dataset_manifest": dataset, "dataset_manifest_sha256": dataset_hash,
                "item_mapping": mapping, "item_mapping_sha256": mapping_hash,
            },
            "model": {"known_items_from_dataset_manifest": known, "oov_buckets": 2},
        }))
        configuration, configuration_hash = write(f"runs/{scale}/configuration.json", json.dumps({
            "contract": contract, "contract_sha256": contract_hash,
        }))
        config = {"num_items": known + 2, "num_layers": layers, "hidden_size": layers * 32,
                  "num_heads": layers, "max_seq_len": 1024}
        versions = []
        for number in range(6):
            version = f"v{number}"
            path, digest = write(f"runs/{scale}/{version}/checkpoint.pt", f"{scale}-{version}")
            parent_hash = versions[-1]["checkpoint_sha256"] if versions else None
            seal_value = {"version": version, "checkpoint_sha256": digest,
                          "parent_checkpoint_sha256": parent_hash, "contract_sha256": contract_hash}
            selected = scale == "large" and number == 5
            if selected:
                seal_value = {"v5_e1": seal_value, "v5_e2": {"checkpoint_sha256": "not-selected"}}
            seal, seal_hash = write(f"runs/{scale}/{version}/checkpoint.seal.json", json.dumps(seal_value))
            record = dict(version=version, checkpoint=path, checkpoint_sha256=digest,
                          seal=seal, seal_sha256=seal_hash, config=config,
                          parent_version=f"v{number-1}" if number else None,
                          parent_checkpoint_sha256=parent_hash, epochs=1,
                          training_days_half_open=[217 + (number-1)*14, 217 + number*14])
            if number:
                admission_value = {"primary_horizon": "E14", "contract_sha256": contract_hash,
                                   "original_metric_gates_pass": not selected,
                                   "gates": {"quality": True, "bootstrap": not selected},
                                   "full_only_report_sha256": f"report-{scale}-{version}"}
                if selected:
                    admission_value = {"candidates": {"v5_e1": admission_value,
                        "v5_e2": {**admission_value, "original_metric_gates_pass": True}}}
                    record.update(seal_entry="v5_e1", selection="user_provisional_epoch1",
                                  original_metric_gates_pass=False)
                admission, admission_hash = write(f"runs/{scale}/{version}/admission.json", json.dumps(admission_value))
                record.update(configuration=configuration, configuration_sha256=configuration_hash,
                              admission=admission, admission_sha256=admission_hash,
                              E14={"report_sha256": f"report-{scale}-{version}",
                                   "completeness": "partial" if selected else "complete"})
            versions.append(record)
        write(models.CHAIN_MANIFESTS[scale], json.dumps({
            "status": "user_selected_chain", "seed": 17, "population_users": 100,
            "versions": versions,
        }))
    return tmp_path


def test_describe_both_current_chains_skips_weights_and_parquet(model_chains, monkeypatch):
    original = models.sha256_file

    def metadata_hash(path):
        assert Path(path).suffix not in (".pt", ".parquet")
        return original(path)

    monkeypatch.setattr(models, "sha256_file", metadata_hash)
    for scale, layers, known in (("medium", 6, 20), ("large", 10, 30)):
        for edge in range(5):
            pair = models.load_model_pair(scale, edge, verify_hashes=False)
            assert pair["config"]["num_layers"] == layers
            assert pair["dataset"]["known_items"] == known
            assert pair["current"]["version"] == f"v{edge+1}"
            assert pair["cutover_day"] == 231 + edge * 14
            assert pair["admission"]["reuse_eligible"] == (scale != "large" or edge != 4)
            assert not pair["weights_hash_verified"]
            json.dumps(pair)
    assert pair["current"]["selection"] == "user_provisional_epoch1"
    assert not pair["admission"]["original_metric_gates_pass"]


def test_verify_rejects_changed_selected_weights_and_parent_lineage(model_chains):
    pair = models.load_model_pair("medium", 1)
    assert pair["weights_hash_verified"]
    path = Path(pair["current"]["checkpoint"])
    path.write_text("changed weight")
    with pytest.raises(RuntimeError, match="checkpoint hash differs"):
        models.load_model_pair("medium", 1)
    manifest = model_chains / models.CHAIN_MANIFESTS["medium"]
    value = json.loads(manifest.read_text())
    value["versions"][2]["parent_checkpoint_sha256"] = "wrong parent"
    manifest.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="chain parent hash/version"):
        models.load_model_pair("medium", 1, verify_hashes=False)


def test_reused_prefix_admission_still_uses_new_chain_selected_weight(model_chains):
    manifest = model_chains / models.CHAIN_MANIFESTS["medium"]
    chain = json.loads(manifest.read_text())
    current = chain["versions"][1]
    for key in ("admission", "admission_sha256", "configuration", "configuration_sha256", "E14"):
        current.pop(key)
    current["historical_source"] = "old/D14/checkpoints/v1/checkpoint.pt"
    manifest.write_text(json.dumps(chain))
    seal = json.loads((model_chains / current["seal"]).read_text())
    admission = model_chains / "old/D14/admission/v0_to_v1.seal.json"
    admission.parent.mkdir(parents=True)
    admission.write_text(json.dumps({
        "status": "medium_full_only_release_eligibility_sealed", "branch": "D14",
        "edge": "v0_to_v1", "primary_horizon_days": 14, "all_metric_gates_pass": True,
        "reuse_unlocked": True, "gates": {"quality": True},
        "contract_sha256": seal["contract_sha256"], "full_only_report_sha256": "retained report",
    }))
    pair = models.load_model_pair("medium", 0, verify_hashes=False)
    assert pair["admission"]["kind"] == "historical_prefix_full_only"
    assert pair["current"]["checkpoint"] == str(model_chains / current["checkpoint"])
    assert pair["admission"]["reuse_eligible"]
