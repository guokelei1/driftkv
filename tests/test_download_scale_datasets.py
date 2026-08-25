from pathlib import Path
import importlib.util
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "download_scale_datasets", ROOT / "scripts/download_scale_datasets.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_recflow_output_names() -> None:
    assert MODULE.recflow_output_name({"name": "realshow.tar", "file_ext": "gz"}) == (
        "realshow.tar.gz"
    )
    assert MODULE.recflow_output_name({"name": "2024-02-18", "file_ext": "feather"}) == (
        "2024-02-18.feather"
    )


def test_human_bytes_uses_binary_units() -> None:
    assert MODULE.human_bytes(1024**3) == "1.000 GiB"
