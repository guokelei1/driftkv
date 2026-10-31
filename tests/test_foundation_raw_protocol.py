from __future__ import annotations

import pyarrow as pa
import pytest

from hstu_kvcache.evaluation.raw_protocol import PATHS, validate_raw_table


def _table(include_label: bool = False) -> pa.Table:
    rows = []
    for path in PATHS:
        rows.append({
            "request_id": "q1", "uid": 1, "query_timestamp": 20,
            "edge": "v0_to_v1", "path": path, "base_logit": 0.1,
            "residual_logit": 0.2, "append_count_since_cutover": 0,
            "seconds_since_cutover": 10, "history_length": 2,
            "cache_length": 2, "checkpoint_sha256": "a",
            "manifest_sha256": "b", **({"label": 1} if include_label else {}),
        })
    return pa.Table.from_pylist(rows)


def test_raw_protocol_requires_all_six_paths_before_label_join() -> None:
    assert validate_raw_table(_table())["request_edge_groups"] == 1
    with pytest.raises(ValueError, match="label"):
        validate_raw_table(_table(include_label=True))
    with pytest.raises(ValueError, match="six paths"):
        validate_raw_table(_table().slice(0, 5))
