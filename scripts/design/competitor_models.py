"""Resolve the current consolidated model chains without importing torch.

Selection into a working chain and Full-only admission are separate fields.
Metadata inspection never reads checkpoints, parquet data, or quality reports.
"""

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CHAIN_MANIFESTS = {
    "medium": "results/unified_training_2026_09/medium/seed17/checkpoints/chain.manifest.json",
    "large": "results/unified_training_2026_09/large/seed17/checkpoints/chain.v0_v5.manifest.json",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, expected_hash=None):
    path = ROOT / path
    if expected_hash is not None and sha256_file(path) != expected_hash:
        raise RuntimeError(f"metadata hash differs: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_model_pair(scale: str, edge_index: int, verify_hashes: bool = True) -> dict:
    """Return JSON-safe pair metadata; false skips all weight/parquet hashing.

    The caller must require admission.reuse_eligible before model evaluation.
    Describing a selected but rejected candidate, including Large V5 epoch 1,
    remains valid and never substitutes the retained alternative endpoint.
    """
    scale = scale.lower()
    if scale not in CHAIN_MANIFESTS or edge_index not in range(5):
        raise ValueError("expected scale medium/large and edge_index 0..4")
    manifest_path = ROOT / CHAIN_MANIFESTS[scale]
    chain = _json(manifest_path)
    versions = {value["version"]: value for value in chain["versions"]}
    parent, current = (versions[f"v{i}"] for i in (edge_index, edge_index + 1))
    edge = f"{parent['version']}_to_{current['version']}"
    if (current["parent_version"] != parent["version"]
            or current["parent_checkpoint_sha256"] != parent["checkpoint_sha256"]):
        raise RuntimeError("chain parent hash/version does not match the selected pair")
    if parent["config"] != current["config"]:
        raise RuntimeError("selected pair has different architecture or vocabulary configuration")
    resolved, seals = {}, {}
    for name, record in (("parent", parent), ("current", current)):
        seal = _json(record["seal"], record["seal_sha256"])
        if record.get("seal_entry"):
            seal = seal[record["seal_entry"]]
        if (seal["version"] != record["version"]
                or seal["checkpoint_sha256"] != record["checkpoint_sha256"]
                or ("parent_checkpoint_sha256" in seal and
                    seal["parent_checkpoint_sha256"] != record["parent_checkpoint_sha256"])):
            raise RuntimeError("checkpoint seal differs from the selected chain entry")
        path = ROOT / record["checkpoint"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if verify_hashes and sha256_file(path) != record["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash differs: {record['version']}")
        resolved[name] = {
            key: record.get(key) for key in
            ("version", "checkpoint_sha256", "seal_sha256", "parent_checkpoint_sha256", "epochs")
        }
        resolved[name].update(checkpoint=str(path), seal=str(ROOT / record["seal"]),
                              selection=record.get("selection", chain["status"]))
        seals[name] = seal

    # The shared data binding comes from this round's recorded configuration;
    # reused prefix versions have no new run configuration of their own.
    authority = current if current.get("configuration") else next(
        record for record in chain["versions"] if record.get("configuration")
    )
    configuration = _json(authority["configuration"], authority["configuration_sha256"])
    contract_path = ROOT / configuration["contract"]
    if sha256_file(contract_path) != configuration["contract_sha256"]:
        raise RuntimeError("recorded training contract hash differs")
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    frozen = contract["frozen_inputs"]
    manifest = ROOT / frozen["dataset_manifest"]
    dataset = _json(manifest, frozen["dataset_manifest_sha256"])
    mapping = ROOT / frozen["item_mapping"]
    if (dataset["scale"].lower() != scale
            or (manifest.parent / dataset["item_mapping_path"]).resolve() != mapping.resolve()):
        raise RuntimeError("dataset scale or item mapping differs from its training contract")
    known, oov = dataset["foundation_items"], contract["model"]["oov_buckets"]
    if (known != contract["model"]["known_items_from_dataset_manifest"]
            or current["config"]["num_items"] != known + oov):
        raise RuntimeError("dataset vocabulary differs from the selected model")
    if verify_hashes and sha256_file(mapping) != frozen["item_mapping_sha256"]:
        raise RuntimeError("item mapping hash differs")

    if current.get("admission"):
        admission_path = ROOT / current["admission"]
        admission = _json(admission_path, current["admission_sha256"])
        if current.get("seal_entry"):
            admission = admission["candidates"][current["seal_entry"]]
        passed = admission.get("original_metric_gates_pass") is True
        kind = "current_run_full_only"
        if admission.get("primary_horizon") != "E14":
            raise RuntimeError("current run admission is not the recorded E14 assessment")
    else:
        # Historical weights are selected by the new chain; only their original
        # admission provenance comes from the retained historical run.
        source = ROOT / current.get("historical_source", current.get("source_checkpoint", ""))
        admission_path = source.parents[2] / "admission" / f"{edge}.seal.json"
        admission = _json(admission_path)
        expected_status = ("medium_full_only_release_eligibility_sealed" if scale == "medium"
                           else "large_full_only_release_admission_sealed")
        if (admission.get("status") != expected_status or admission.get("branch") != "D14"
                or admission.get("edge") != edge or admission.get("primary_horizon_days") != 14):
            raise RuntimeError("historical admission does not match the selected chain edge")
        passed = admission.get("all_metric_gates_pass") is True
        if scale == "medium":
            passed = passed and admission.get("reuse_unlocked") is True
        kind = "historical_prefix_full_only"
    if admission["contract_sha256"] != seals["current"]["contract_sha256"]:
        raise RuntimeError("admission and checkpoint seal refer to different contracts")
    evidence = current.get("E14", current.get("historical_E14", {}))
    report_hash = evidence.get("report_sha256", evidence.get("adjudication_sha256"))
    if report_hash is not None and admission["full_only_report_sha256"] != report_hash:
        raise RuntimeError("admission report hash differs from the selected chain evidence")
    gates = admission.get("gates", {})
    eligible = passed and bool(gates) and all(value is True for value in gates.values())
    if "original_metric_gates_pass" in current and current["original_metric_gates_pass"] != passed:
        raise RuntimeError("chain and admission disagree about the selected endpoint gates")
    cutover_day = int(current["training_days_half_open"][1])
    return {
        "scale": scale, "seed": chain["seed"], "edge_index": edge_index, "edge": edge,
        "cutover_day": cutover_day, "cutover": cutover_day * 86400,
        "chain_manifest": str(manifest_path), "chain_manifest_sha256": sha256_file(manifest_path),
        "chain_status": chain["status"], "interpretation": chain.get("interpretation"),
        "population_users": chain["population_users"], "config": current["config"], **resolved,
        "dataset": {
            "manifest": str(manifest), "manifest_sha256": frozen["dataset_manifest_sha256"],
            "item_mapping": str(mapping), "item_mapping_sha256": frozen["item_mapping_sha256"],
            "users": str(manifest.parent / dataset["users_path"]),
            "known_items": known, "oov_buckets": oov,
            "contract": str(contract_path), "contract_sha256": configuration["contract_sha256"],
        },
        "admission": {
            "path": str(admission_path), "sha256": sha256_file(admission_path), "kind": kind,
            "scope": evidence.get("completeness", contract.get("evaluation", {}).get("completeness", "E14")),
            "original_metric_gates_pass": passed, "reuse_eligible": eligible, "gates": gates,
            "selection": resolved["current"]["selection"],
            "report_sha256": admission["full_only_report_sha256"], "serving_lineage_promoted": False,
        },
        "weights_hash_verified": verify_hashes, "item_mapping_hash_verified": verify_hashes,
    }
