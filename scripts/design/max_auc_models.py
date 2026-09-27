"""Resolve the two fixed Max epoch-one model pairs from retained training seals."""

import json
from pathlib import Path

import yaml

from design.competitor_models import sha256_file

ROOT = Path(__file__).resolve().parents[2]
CHAIN = ROOT / "results/insight/max_unified_auc_10k_20260921/model_chain.json"


def checked_json(path, expected=None):
    path = ROOT / path
    if expected is not None:
        assert sha256_file(path) == expected, str(path)
    return json.loads(path.read_text())


def load_model_pair(scale, edge_index, verify_hashes=True):
    assert scale.lower() == "max" and edge_index in (0, 1)
    chain = checked_json(CHAIN)
    parent, current = chain["versions"][edge_index:edge_index + 2]
    assert current["parent_checkpoint_sha256"] == parent["checkpoint_sha256"]
    assert parent["config"] == current["config"]
    resolved = {}
    for role, record in (("parent", parent), ("current", current)):
        seal = checked_json(record["seal"], record["seal_sha256"])
        if record["seal_entry"]:
            seal = seal[record["seal_entry"]]
        assert seal["checkpoint_sha256"] == record["checkpoint_sha256"]
        assert seal["parent_checkpoint_sha256"] == record["parent_checkpoint_sha256"]
        assert seal["version"] == record["version"] and seal["epochs"] == 1
        configuration = checked_json(record["configuration"], record["configuration_sha256"])
        contract = ROOT / record["contract"]
        assert sha256_file(contract) == record["contract_sha256"] == configuration["contract_sha256"]
        assert seal["contract_sha256"] == record["contract_sha256"]
        if verify_hashes:
            assert sha256_file(ROOT / record["checkpoint"]) == record["checkpoint_sha256"]
        resolved[role] = {k: record[k] for k in (
            "version", "checkpoint_sha256", "seal_sha256", "parent_checkpoint_sha256", "epochs", "selection")}
        resolved[role].update(checkpoint=str(ROOT / record["checkpoint"]), seal=str(ROOT / record["seal"]))
    frozen = chain["frozen_data"]
    contract = yaml.safe_load((ROOT / current["contract"]).read_text())
    for key, value in frozen.items():
        assert contract["frozen_inputs"][key] == value
    manifest = ROOT / frozen["dataset_manifest"]
    dataset = checked_json(manifest, frozen["dataset_manifest_sha256"])
    checked_json(frozen["request_manifest"], frozen["request_manifest_sha256"])
    checked_json(frozen["data_audit"], frozen["data_audit_sha256"])
    mapping = ROOT / frozen["item_mapping"]
    if verify_hashes:
        assert sha256_file(mapping) == frozen["item_mapping_sha256"]
    assert dataset["foundation_items"] + dataset["oov_buckets"] == current["config"]["num_items"]
    assert dataset["oov_bucket_start"] == dataset["foundation_items"] + 1
    assert dataset["history_tie_order"] == "timestamp_raw_item_behavior"
    admission = checked_json(current["admission"], current["admission_sha256"])
    if current["seal_entry"]:
        admission = admission["candidates"][current["seal_entry"]]
    assert admission["contract_sha256"] == current["contract_sha256"]
    assert admission["full_only_report_sha256"] == current["report_sha256"]
    assert sha256_file(ROOT / current["report"]) == current["report_sha256"]
    passed = admission["original_metric_gates_pass"] and all(admission["gates"].values())
    assert passed and admission["primary_horizon"] == "E14"
    day = current["training_days_half_open"][1]
    return dict(scale="max", seed=17, edge_index=edge_index,
        edge=f"v{edge_index}_to_v{edge_index + 1}", cutover_day=day, cutover=day * 86400,
        chain_manifest=str(CHAIN), chain_manifest_sha256=sha256_file(CHAIN),
        chain_status=chain["status"], interpretation=chain["interpretation"], population_users=200000,
        config=current["config"], **resolved,
        dataset=dict(manifest=str(manifest), manifest_sha256=frozen["dataset_manifest_sha256"],
            item_mapping=str(mapping), item_mapping_sha256=frozen["item_mapping_sha256"],
            users=str(manifest.parent / dataset["users_path"]), known_items=dataset["foundation_items"],
            oov_buckets=dataset["oov_buckets"], oov_bucket_start=dataset["oov_bucket_start"],
            known_item_upper_exclusive=dataset["oov_bucket_start"],
            history_tie_order=dataset["history_tie_order"], contract=str(ROOT / current["contract"]),
            contract_sha256=current["contract_sha256"]),
        admission=dict(path=str(ROOT / current["admission"]), sha256=current["admission_sha256"],
            kind="current_run_full_only", scope="complete", original_metric_gates_pass=True,
            reuse_eligible=True, gates=admission["gates"], selection=current["selection"],
            report_sha256=current["report_sha256"], serving_lineage_promoted=False),
        weights_hash_verified=verify_hashes, item_mapping_hash_verified=verify_hashes)
