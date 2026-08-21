from __future__ import annotations

import hashlib
import json

import pytest

from hstu_kvcache.data import QualificationUnlock, load_compact_index


def test_qualification_index_is_sealed_by_default(tmp_path) -> None:
    path = tmp_path / "qualification.index.json"
    path.write_text(json.dumps({"split": "qualification", "request_shards": []}) + "\n")
    with pytest.raises(PermissionError, match="sealed"):
        load_compact_index(path)

    index_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    unlock = QualificationUnlock(
        qualification_index_hash=index_hash,
        frozen_base_hashes=("base",),
        frozen_checkpoint_hashes=("m0", "m1"),
        checkpoint_selection_complete=True,
    )
    assert load_compact_index(path, qualification_unlock=unlock)["split"] == "qualification"


def test_nonqualification_index_does_not_require_unlock(tmp_path) -> None:
    path = tmp_path / "development.index.json"
    path.write_text(json.dumps({"split": "development", "request_shards": []}) + "\n")
    assert load_compact_index(path)["split"] == "development"
