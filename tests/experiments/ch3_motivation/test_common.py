import importlib
from pathlib import Path
import sys

import pytest

common=importlib.import_module("experiments.ch3_motivation.code.common")


def test_small_population_is_diagnostic_only():
    with pytest.raises(ValueError,match="512"):
        common.quality_role(list(range(511)))
    assert common.quality_role(list(range(32)),diagnostic=True)=="numerical_performance_diagnostic"
    assert common.quality_role(list(range(512)))=="measured_partial_population"


def test_proportional_nested_sampling_and_seed_sensitivity():
    rows=[(u,100 if u<800 else 1024,4 if u<800 else 50) for u in range(1000)]
    first=common.stratified_order(rows,17)
    second=common.stratified_order(rows,18)
    assert set(first)==set(second)==set(range(1000))
    assert first!=second
    assert sum(u<800 for u in first[:500])==400
    assert first==common.stratified_order(list(reversed(rows)),17)


def test_population_roles_are_disjoint_and_all_e6_sizes_met():
    import json
    root=common.ROOT/"experiments/ch3_motivation/inputs"
    if not (root/"population.json").exists():
        pytest.skip("CPU packs are prepared separately")
    roles=json.loads((root/"population.json").read_text())
    names=["fit","validation","development","mismatch","confirmation"]
    for i,name in enumerate(names):
        for other in names[i+1:]:
            assert not set(roles[name])&set(roles[other])
    assert set(roles["main"])<=set(roles["mismatch"])
    for edge in range(1,6):
        path=root/f"edge_{edge}/inputs.json"
        if path.exists():
            binding=json.loads(path.read_text())
            assert len(set(binding["e6_users"]))==2000
            assert not set(binding["e6_users"])&set(roles["fit"]+roles["validation"]+roles["development"])
