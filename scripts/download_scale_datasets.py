#!/usr/bin/env python3
"""Plan or download the raw datasets needed by the scale/external tracks.

The safe default is a read-only plan.  ``--download`` fetches only the core
bundle: Yambda-500M listens/likes/dislikes and RecFlow realshow.  RecFlow stage
companions and all-stage rows require explicit flags because they are not needed
for the first protocol/H gates.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
YAMBDA_API = (
    "https://huggingface.co/api/datasets/yandex/yambda/tree/main/flat/500m"
    "?recursive=true&expand=true"
)
YAMBDA_RESOLVE = "https://huggingface.co/datasets/yandex/yambda/resolve/main"
RECFLOW_LINK = "f8e5adc0-2e57-11ef-bea5-3b4cac9d110e"
RECFLOW_API = "https://recapi.ustc.edu.cn/api/v2/link/target"
YAMBDA_CORE = {
    "flat/500m/listens.parquet",
    "flat/500m/likes.parquet",
    "flat/500m/dislikes.parquet",
}
RECFLOW_CORE = {"realshow.tar"}
RECFLOW_COMPANION = {"request_id_dict.tar", "others.tar"}
GIB = 1024**3


@dataclass(frozen=True)
class DownloadFile:
    dataset: str
    logical_path: str
    output: str
    size: int
    url: str | None = None
    sha256: str | None = None
    resource_number: str | None = None


def fetch_json(url: str, payload: dict[str, Any] | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"User-Agent": "EvoKV-data-downloader/1.0"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8-sig"))


def yambda_files() -> list[DownloadFile]:
    rows = fetch_json(YAMBDA_API)
    selected: list[DownloadFile] = []
    for row in rows:
        path = row.get("path")
        if path not in YAMBDA_CORE:
            continue
        lfs = row.get("lfs") or {}
        selected.append(
            DownloadFile(
                dataset="yambda500m",
                logical_path=path,
                output=str(Path("data/raw/yambda") / path),
                size=int(row["size"]),
                url=f"{YAMBDA_RESOLVE}/{path}?download=true",
                sha256=lfs.get("oid"),
            )
        )
    missing = YAMBDA_CORE - {item.logical_path for item in selected}
    if missing:
        raise RuntimeError(f"Yambda official listing is missing: {sorted(missing)}")
    return sorted(selected, key=lambda item: item.logical_path)


def recflow_list(resource_number: str | None = None) -> list[dict[str, Any]]:
    payload: dict[str, Any] = {"link_number": RECFLOW_LINK}
    if resource_number is not None:
        payload["link_resource_number"] = resource_number
    response = fetch_json(f"{RECFLOW_API}/resource/list", payload)
    if response.get("status_code") != 200:
        raise RuntimeError(f"RecFlow listing failed: {response}")
    return list(response["entity"])


def recflow_output_name(row: dict[str, Any]) -> str:
    name = str(row["name"])
    extension = str(row.get("file_ext") or "")
    if not extension:
        return name
    if name.endswith(f".{extension}"):
        return name
    if extension == "gz" and name.endswith(".tar"):
        return f"{name}.gz"
    return f"{name}.{extension}"


def recflow_files(
    include_companion: bool, include_all_stage: bool
) -> list[DownloadFile]:
    top = recflow_list()
    data_folder = next((row for row in top if row["name"] == "data"), None)
    if data_folder is None:
        raise RuntimeError("RecFlow share does not contain the expected data folder")
    rows = recflow_list(str(data_folder["number"]))
    wanted = set(RECFLOW_CORE)
    if include_companion:
        wanted |= RECFLOW_COMPANION
    selected: list[DownloadFile] = []
    for row in rows:
        if row["type"] == "file" and row["name"] in wanted:
            name = recflow_output_name(row)
            selected.append(
                DownloadFile(
                    dataset="recflow",
                    logical_path=f"data/{name}",
                    output=str(Path("data/raw/recflow/downloads") / name),
                    size=int(row["bytes"]),
                    resource_number=str(row["number"]),
                )
            )
    missing = wanted - {Path(item.logical_path).name.removesuffix(".gz") for item in selected}
    if missing:
        raise RuntimeError(f"RecFlow official listing is missing: {sorted(missing)}")

    if include_all_stage:
        folder = next(
            (row for row in rows if row["type"] == "folder" and row["name"] == "all_stage"),
            None,
        )
        if folder is None:
            raise RuntimeError("RecFlow share does not contain all_stage")
        for row in recflow_list(str(folder["number"])):
            if row["type"] != "file":
                raise RuntimeError(f"Unexpected nested all_stage entry: {row}")
            name = recflow_output_name(row)
            selected.append(
                DownloadFile(
                    dataset="recflow",
                    logical_path=f"data/all_stage/{name}",
                    output=str(Path("data/raw/recflow/downloads/all_stage") / name),
                    size=int(row["bytes"]),
                    resource_number=str(row["number"]),
                )
            )
    return sorted(selected, key=lambda item: item.logical_path)


def recflow_signed_url(resource_number: str) -> str:
    response = fetch_json(
        f"{RECFLOW_API}/download",
        {"link_number": RECFLOW_LINK, "link_resources_list": [resource_number]},
    )
    if response.get("status_code") != 200:
        raise RuntimeError(f"RecFlow signed URL request failed: {response}")
    return str(response["entity"][resource_number])


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.3f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def remaining_bytes(files: Iterable[DownloadFile]) -> int:
    remaining = 0
    for item in files:
        path = ROOT / item.output
        partial = path.with_name(path.name + ".part")
        existing = path.stat().st_size if path.exists() else 0
        if not path.exists() and partial.exists():
            existing = partial.stat().st_size
        remaining += max(0, item.size - existing)
    return remaining


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(item: DownloadFile) -> dict[str, Any]:
    output = ROOT / item.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size == item.size:
        if item.sha256 and sha256_file(output) != item.sha256:
            raise RuntimeError(f"Checksum mismatch for existing file: {output}")
        print(f"SKIP {item.logical_path} ({human_bytes(item.size)})", flush=True)
        return {**asdict(item), "status": "already_complete"}
    if output.exists():
        raise RuntimeError(
            f"Existing final file has wrong size; move it aside before retrying: {output}"
        )

    partial = output.with_name(output.name + ".part")
    if partial.exists() and partial.stat().st_size > item.size:
        raise RuntimeError(f"Partial file is larger than expected: {partial}")
    url = item.url
    if item.dataset == "recflow":
        if not item.resource_number:
            raise RuntimeError(f"Missing RecFlow resource number for {item.logical_path}")
        url = recflow_signed_url(item.resource_number)
    if not url:
        raise RuntimeError(f"Missing URL for {item.logical_path}")

    print(f"GET  {item.logical_path} ({human_bytes(item.size)})", flush=True)
    command = [
        "curl",
        "--location",
        "--fail",
        "--show-error",
        "--retry",
        "8",
        "--retry-delay",
        "5",
        "--retry-all-errors",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        url,
    ]
    subprocess.run(command, check=True)
    actual = partial.stat().st_size
    if actual != item.size:
        raise RuntimeError(
            f"Size mismatch for {partial}: expected {item.size}, found {actual}"
        )
    digest = sha256_file(partial)
    if item.sha256 and digest != item.sha256:
        raise RuntimeError(
            f"SHA256 mismatch for {partial}: expected {item.sha256}, found {digest}"
        )
    os.replace(partial, output)
    print(f"DONE {item.logical_path} sha256={digest}", flush=True)
    return {**asdict(item), "status": "downloaded", "observed_sha256": digest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("yambda500m", "recflow"),
        default=("yambda500m", "recflow"),
        help="Datasets to plan/download (default: both)",
    )
    parser.add_argument(
        "--include-recflow-companion",
        action="store_true",
        help="Also fetch request_id_dict and others (about 20.7 GiB)",
    )
    parser.add_argument(
        "--include-recflow-all-stage",
        action="store_true",
        help="Also fetch all 37 all_stage files (about 55 GiB; not needed initially)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download files. Without this flag the command is read-only.",
    )
    parser.add_argument("--jobs", type=int, default=2, help="Concurrent files (default: 2)")
    parser.add_argument(
        "--reserve-gib",
        type=float,
        default=20.0,
        help="Free disk space that must remain after compressed download",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    if (args.include_recflow_companion or args.include_recflow_all_stage) and (
        "recflow" not in args.datasets
    ):
        raise SystemExit("RecFlow optional flags require --datasets recflow")

    files: list[DownloadFile] = []
    if "yambda500m" in args.datasets:
        files.extend(yambda_files())
    if "recflow" in args.datasets:
        files.extend(
            recflow_files(args.include_recflow_companion, args.include_recflow_all_stage)
        )

    total = sum(item.size for item in files)
    remaining = remaining_bytes(files)
    free = shutil.disk_usage(ROOT).free
    print(f"Files: {len(files)}")
    for dataset in sorted({item.dataset for item in files}):
        subtotal = sum(item.size for item in files if item.dataset == dataset)
        print(f"  {dataset}: {human_bytes(subtotal)}")
    print(f"Compressed total: {human_bytes(total)}")
    print(f"Remaining download: {human_bytes(remaining)}")
    print(f"Filesystem free: {human_bytes(free)}")
    print("Extraction and derived manifests are not included in these byte counts.")
    if not args.download:
        print("PLAN ONLY. Add --download to fetch the listed bundle.")
        return 0

    required = remaining + int(args.reserve_gib * GIB)
    if free < required:
        raise SystemExit(
            f"Insufficient free disk: need {human_bytes(required)}, have {human_bytes(free)}"
        )
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(download_one, item) for item in files]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    manifest_path = ROOT / "data/raw/download_manifest_v1.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "official_sources": {
            "yambda": "https://huggingface.co/datasets/yandex/yambda",
            "recflow": f"https://rec.ustc.edu.cn/share/{RECFLOW_LINK}",
        },
        "compressed_total_bytes": total,
        "files": sorted(results, key=lambda row: row["logical_path"]),
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    print(f"Manifest: {manifest_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

