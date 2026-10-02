from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from read_correction_2026_09.prepare import choose_users


def test_fitting_membership_excludes_evaluation_and_is_nested():
    population = list(range(100))
    excluded = {2, 8, 32, 60}
    larger = choose_users(population, excluded, 32)
    assert set(larger).isdisjoint(excluded)
    assert choose_users(reversed(population), excluded, 8) == larger[:8]
