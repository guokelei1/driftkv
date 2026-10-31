"""The data handoff preserves primary precision and population scope."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[3]

from experiments.ch3_motivation.code.aggregate_mismatch import median_ci, merge_exact_csv


def test_exact_csv_merge_keeps_all_decimal_digits(tmp_path):
    a,b=tmp_path/"a.csv",tmp_path/"b.csv"
    a.write_bytes(b"edge,value\r\n1,0.12345678901234567\r\n")
    b.write_bytes(b"edge,value\r\n2,0.98765432109876543\r\n")
    output=tmp_path/"points.csv"
    assert merge_exact_csv([a,b],output)==2
    assert output.read_bytes()==b"edge,value\r\n1,0.12345678901234567\r\n2,0.98765432109876543\r\n"


def test_median_bootstrap_is_population_guarded_and_reproducible():
    with pytest.raises(ValueError,match="diagnostic"):
        median_ci([.2]*16,replicates=32)
    actual=median_ci(np.full(512,.25),replicates=32,seed=17)
    assert actual["median"]==.25 and actual["median_uid_ci95"]==[.25,.25]
    assert actual["users"]==512 and actual["replicates"]==32
