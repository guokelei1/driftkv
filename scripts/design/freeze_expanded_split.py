#!/usr/bin/env python3
"""Freeze population-stratified Design users without reading model outputs."""

import hashlib
import json

import numpy as np
import pyarrow.parquet as pq
from design.data import DATASET, ROOT, SPLIT_PATH, fixed_split


def main():
    path = ROOT / "data/manifests/evokv_design_medium_6000_v1/split.json"
    if path.exists():
        raise FileExistsError(path)
    base = fixed_split()
    users = pq.read_table(DATASET.parent / "users.parquet",
                          columns=["uid", "n_theta0"]).to_pandas()
    assert len(users) == 30000 and users.uid.is_unique and users.n_theta0.min() > 0
    users["stratum"] = np.searchsorted([256, 1024, 4096], users.n_theta0, side="right")
    used, inputs = set(), {}
    for config in sorted((ROOT / "results/design").glob("*/configuration.json")):
        value = json.loads(config.read_text())
        fitted = value.get("calibration_uids", [])
        if fitted:
            used.update(fitted)
            inputs[str(config.relative_to(ROOT))] = hashlib.sha256(config.read_bytes()).hexdigest()
    population = users.groupby("stratum").size().reindex(range(4), fill_value=0).to_numpy()
    quotas = np.floor(population * .2).astype(int)
    remainder = 6000 - quotas.sum()
    for i in np.argsort(-(population * .2 - quotas), kind="stable")[:remainder]:
        quotas[i] += 1
    groups = {"development": list(base["development"]),
              "confirmation": list(base["confirmation"])}
    reserved = set(base["reserved_legacy_confirmation"])
    occupied = used | reserved | set(groups["development"]) | set(groups["confirmation"])
    for role, selected in groups.items():
        for stratum, count in enumerate(quotas):
            within = users[users.stratum == stratum]
            needed = int(count) - int(within.uid.isin(selected).sum())
            assert needed >= 0
            candidates = [int(uid) for uid in within.uid if uid not in occupied]
            candidates.sort(key=lambda uid: hashlib.sha256(f"evokv-design-6000:17:{role}:{uid}".encode()).digest())
            assert len(candidates) >= needed
            selected.extend(candidates[:needed])
            occupied.update(candidates[:needed])
        assert len(selected) == len(set(selected)) == 6000
    # Preserve the established fitting order, then include previously excluded
    # short histories as available calibration users. Evaluation UIDs stay out.
    excluded = reserved | set(groups["development"]) | set(groups["confirmation"])
    calibration = [uid for uid in base["calibration"] if uid not in excluded]
    calibration_set = set(calibration)
    calibration.extend(sorted(int(uid) for uid in users.uid if uid not in excluded | calibration_set))
    assert used <= set(calibration)
    sets = [set(v) for v in [*groups.values(), calibration, list(reserved)]]
    assert sum(map(len, sets)) == 30000 and len(set.union(*sets)) == 30000
    result = dict(status="uid_frozen_before_expanded_method_outputs", seed=17,
        base_split_path=str(SPLIT_PATH.relative_to(ROOT)),
        base_split_sha256=hashlib.sha256(SPLIT_PATH.read_bytes()).hexdigest(),
        selection="20 percent per n_theta0 stratum; SHA256(seed, role, uid); retain original development and confirmation",
        strata=["1_to_255", "256_to_1023", "1024_to_4095", "4096_plus"],
        population_stratum_counts=population.tolist(), evaluation_stratum_counts=quotas.tolist(),
        **groups, calibration=calibration, reserved_legacy_confirmation=base["reserved_legacy_confirmation"],
        historical_fitted_uids=sorted(used), historical_fit_configuration_sha256=inputs,
        confirmation_target_outputs_read=False, cost_target_population=30000,
        scope="6000 development users plus separately reserved 6000 confirmation users; confirmation stays unopened")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         population=population.tolist(), per_evaluation=quotas.tolist(),
                         calibration=len(calibration), historical_fitted=len(used))))


if __name__ == "__main__":
    main()
