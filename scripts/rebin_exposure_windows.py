from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hstu_kvcache.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--base-prefix", type=int, default=64)
    parser.add_argument("--window-size", type=int, required=True)
    parser.add_argument("--windows", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source)
    output_path = Path(args.output)
    metadata_path = Path(args.metadata_output)
    with np.load(source_path, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    metadata = json.loads(str(arrays.pop("metadata_json").item()))
    if not str(metadata["dataset"]).startswith("tenrec"):
        raise ValueError("window rebinning requires Tenrec ordinal time_ms")
    positions = arrays["time_ms"] // 1000
    limit = args.base_prefix + args.window_size * args.windows
    if int(positions.max()) >= limit:
        raise ValueError("source contains positions beyond the requested horizon")
    window_index = np.where(
        positions < args.base_prefix,
        -1,
        (positions - args.base_prefix) // args.window_size,
    ).astype(np.int8)
    if np.any((window_index < -1) | (window_index >= args.windows)):
        raise ValueError("rebinned window index is out of range")
    arrays["window_index"] = window_index
    split_rows = {
        "base": int(np.count_nonzero(window_index == -1)),
        **{
            f"window_{index}": int(np.count_nonzero(window_index == index))
            for index in range(args.windows)
        },
    }
    split_positive = {
        name: int(
            arrays["label"][
                window_index == (-1 if name == "base" else int(name[7:]))
            ].sum()
        )
        for name in split_rows
    }
    metadata.update(
        {
            "protocol": "ordered_exposure_prepared_rebinned_v1",
            "source_prepared": str(source_path),
            "base_prefix": args.base_prefix,
            "window_size": args.window_size,
            "window_count": args.windows,
            "split_rows": split_rows,
            "split_positive_rows": split_positive,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **arrays,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    metadata["output"] = str(output_path)
    metadata["output_bytes"] = output_path.stat().st_size
    save_json(metadata, metadata_path)
    print(metadata_path)
    print(output_path)


if __name__ == "__main__":
    main()
