import importlib
from pathlib import Path
import sys

module=importlib.import_module("experiments.ch3_motivation.code.prepare_candidates")


def test_recent_feedback_mix_includes_last_known_and_unique_fill():
    values=module.candidates_for(5,[10,10,9,0,11],10,count=6,recent_fraction=.5)
    assert values[:2]==[10,9]
    assert len(values)==len(set(values))==6
    assert all(1<=x<=10 for x in values)
    assert values==module.candidates_for(5,[10,10,9,0,11],10,count=6,recent_fraction=.5)


def test_zero_recent_fraction_does_not_keep_recent_order():
    values=module.candidates_for(5,[10,9,8],10,count=6,recent_fraction=0.)
    assert values==module.candidates_for(5,[],10,count=6,recent_fraction=0.)
