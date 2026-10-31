"""Parallel, restartable raw-data audit and compact daily candidate conversion.

No model scores are read. Each completed day has immutable source and output
hashes. Failed partial directories are retained, never mistaken for completion.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from .data_common import (
    FEATURES,
    RAW_DTYPE,
    check_exposure_join,
    deduplicate_records,
    json_save,
    rank_membership,
    sha256,
)
from .protocol import day_bounds_ms, day_date


def emit(**value):
    print(json.dumps(value, sort_keys=True), flush=True)


def extract_exposures(source, output):
    output.mkdir(parents=True, exist_ok=True)
    done = output / "manifest.json"
    source_hash = sha256(source)
    if done.exists():
        manifest = json.loads(done.read_text())
        if manifest["source_sha256"] != source_hash:
            raise ValueError("Exposure archive changed")
        for file in manifest["files"]:
            if sha256(output / file["name"]) != file["sha256"]:
                raise ValueError("Extracted exposure file changed")
        return manifest
    files, seen = [], set()
    process = subprocess.Popen(["pigz", "-dc", "-p", "4", str(source)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".feather"):
                    continue
                name = Path(member.name).name
                if name not in {f"{day_date(d)}.feather" for d in range(1, 38)} or name in seen:
                    raise ValueError(f"Unexpected/duplicated exposure member {member.name}")
                seen.add(name)
                target, temporary = output / name, output / (name + ".partial")
                with archive.extractfile(member) as inp, temporary.open("wb") as dest:
                    shutil.copyfileobj(inp, dest, length=8 * 1024**2)
                digest = sha256(temporary)
                if target.exists():
                    if sha256(target) != digest:
                        raise ValueError(f"Existing extracted exposure differs: {target}")
                    temporary.unlink()
                else:
                    os.replace(temporary, target)
                files.append({"name": name, "bytes": target.stat().st_size, "sha256": digest})
        if process.wait() != 0:
            raise ValueError("pigz failed while reading exposure archive")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
    if len(files) != 37:
        raise ValueError(f"Expected 37 exposure days, found {len(files)}")
    manifest = {"source": str(source.resolve()), "source_sha256": source_hash, "files": files}
    json_save(done, manifest)
    return manifest


def projected_reader(source, columns):
    mapping = pa.memory_map(str(source), "r")
    header = ipc.open_file(mapping)
    names = [name.strip() for name in header.schema.names]
    if len(set(names)) != len(names):
        raise ValueError("Column normalization collision")
    missing = set(columns) - set(names)
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")
    ids = [i for i, name in enumerate(names) if name in columns]
    return ipc.open_file(mapping, options=ipc.IpcReadOptions(included_fields=ids, use_threads=True))


def read_compact(source, destination, day, kind):
    fields = ["user_id", "request_timestamp", "request_id", *FEATURES,
              "effective_view", "like", "realshow"]
    if kind == "stage":
        fields += ["rank_index", "rerank_index", "rank_pos", "rank_neg", "rerank_pos",
                   "final_neg", "coarse_neg", "prerank_neg"]
    reader = projected_reader(source, fields)
    start_ms, end_ms = day_bounds_ms(day)
    rows, retained = 0, 0
    stats = {"rank_index_counts": {}, "flag_rank_index_counts": {}, "timestamp_min": None,
             "timestamp_max": None, "source_batches": reader.num_record_batches}
    with destination.open("wb") as output:
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            if any(column.null_count for column in batch.columns):
                raise ValueError("Null raw values require explicit handling")
            values = {name.strip(): batch.column(i).to_numpy() for i, name in enumerate(batch.schema.names)}
            timestamps = values["request_timestamp"]
            if timestamps.min() < start_ms or timestamps.max() >= end_ms:
                raise ValueError(f"D{day} timestamp outside Asia/Shanghai day")
            stats["timestamp_min"] = min(stats["timestamp_min"] or end_ms, int(timestamps.min()))
            stats["timestamp_max"] = max(stats["timestamp_max"] or start_ms, int(timestamps.max()))
            for field in ["effective_view", "like", "realshow"]:
                if not np.isin(values[field], [0, 1]).all():
                    raise ValueError(f"Nonbinary {field}")
            if kind == "stage":
                rank = values["rank_index"]
                keep = rank_membership(rank)
                if np.any((values["realshow"] == 1) & ~keep):
                    raise ValueError("An exposure lies outside inferred ranking membership")
                for flag, sentinel in [("coarse_neg", 3000), ("prerank_neg", 8000)]:
                    if np.any((values[flag] == 1) & (rank != sentinel)):
                        raise ValueError(f"{flag} disagrees with sentinel {sentinel}")
                for flag in ["realshow", "rank_pos", "rank_neg", "rerank_pos", "final_neg"]:
                    if np.any((values[flag] == 1) & ~keep):
                        raise ValueError(f"{flag} lies outside inferred ranking membership")
                unique, counts = np.unique(rank, return_counts=True)
                for val, count in zip(unique, counts, strict=True):
                    key = str(int(val))
                    stats["rank_index_counts"][key] = stats["rank_index_counts"].get(key, 0) + int(count)
            else:
                if not np.all(values["realshow"] == 1):
                    raise ValueError("Unexposed rows in exposure archive")
                keep = np.ones(len(timestamps), dtype=bool)
            raw = np.empty(int(keep.sum()), RAW_DTYPE)
            for target, field in [("uid", "user_id"), ("rid", "request_id"), ("ts", "request_timestamp")]:
                incoming = values[field][keep]
                bounds = np.iinfo(RAW_DTYPE.fields[target][0])
                if incoming.size and (incoming.min() < 0 or incoming.max() > bounds.max):
                    raise ValueError(f"{field} exceeds nonnegative storage type")
                raw[target] = incoming
            for col, field in enumerate(FEATURES):
                incoming = values[field][keep]
                if incoming.size and (incoming.min() < 0 or incoming.max() > np.iinfo(np.int32).max):
                    raise ValueError(f"{field} exceeds feature storage type")
                raw["features"][:, col] = incoming
            raw["behavior"] = (values["effective_view"][keep] + 2 * values["like"][keep]).astype(np.uint8)
            raw["label"] = np.where(values["realshow"][keep] == 1, values["effective_view"][keep], -1)
            raw.tofile(output)
            rows += len(timestamps)
            retained += len(raw)
    stats.update({"raw_rows": rows, "retained_rows_before_dedup": retained})
    return stats


def convert_day(day, kind, input_dir, output_root, threads):
    pa.set_cpu_count(threads)
    pa.set_io_thread_count(threads)
    source = Path(input_dir) / f"{day_date(day)}.feather"
    target = Path(output_root) / ("exposures" if kind == "exposure" else "stage") / f"d{day:02d}"
    manifest_path = target / "manifest.json"
    source_digest = sha256(source)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["source_sha256"] != source_digest or manifest["status"] != "passed":
            raise ValueError("Completed input changed or manifest invalid")
        for file in manifest["outputs"]:
            if sha256(target / file["name"]) != file["sha256"]:
                raise ValueError(f"Completed output changed: {target / file['name']}")
        emit(day=day, kind=kind, status="verified_existing")
        return manifest
    if target.exists():
        raise ValueError(f"Unfinished directory retained for diagnosis: {target}")
    target.mkdir(parents=True)
    started = time.monotonic()
    temporary = target / "raw.partial.bin"
    emit(day=day, kind=kind, status="decoding")
    try:
        stats = read_compact(source, temporary, day, kind)
        raw = np.memmap(temporary, mode="r", dtype=RAW_DTYPE)
        raw, keys, requests, duplicates = deduplicate_records(raw)
        if kind == "stage":
            exposure_root = Path(output_root) / "exposures" / f"d{day:02d}"
            exposure_manifest = json.loads((exposure_root / "manifest.json").read_text())
            if exposure_manifest["status"] != "passed":
                raise ValueError("Exposure prerequisite incomplete")
            exposed = np.load(exposure_root / "events.npy", mmap_mode="r")
            stats["exposure_join"] = check_exposure_join(raw, keys, exposed)
            stats["exposure_manifest_sha256"] = sha256(exposure_root / "manifest.json")
            np.save(target / "features.npy", raw["features"])
            np.save(target / "labels.npy", raw["label"])
        else:
            np.save(target / "events.npy", raw)
        np.save(target / "requests.npy", requests)
        outputs = [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)}
                   for p in sorted(target.glob("*.npy"))]
        stats.update({"status": "passed", "day": day, "date": day_date(day), "kind": kind,
                      "source": str(source.resolve()), "source_sha256": source_digest,
                      "duplicate_rows_removed": duplicates, "rows": len(raw), "requests": len(requests),
                      "positive_requests": int((requests["positive"] > 0).sum()),
                      "zero_exposure_requests": int((requests["observed"] == 0).sum()),
                      "candidate_count_quantiles": np.quantile(requests["stop"] - requests["start"],
                                                              [0, .5, .95, .99, 1]).tolist(),
                      "outputs": outputs, "wall_seconds": time.monotonic() - started,
                      "arrow_threads": threads,
                      "preparer_sha256": sha256(__file__),
                      "contracts_sha256": sha256(Path(__file__).with_name("data_common.py"))})
        json_save(manifest_path, stats)
        temporary.unlink()
        emit(day=day, kind=kind, status="passed", rows=len(raw), requests=len(requests),
             wall_seconds=round(stats["wall_seconds"], 2))
        return stats
    except BaseException as error:
        json_save(target / "failure.json", {"error": repr(error), "source_sha256": source_digest,
                                           "wall_seconds": time.monotonic() - started})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, default=Path("data/raw/recflow/all_stage"))
    parser.add_argument("--exposure-archive", type=Path, default=Path("data/raw/recflow/downloads/realshow.tar.gz"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/recflow_request_ranking_v1"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--arrow-threads", type=int, default=4)
    parser.add_argument("--phase", choices=["all", "exposure", "stage"], default="all")
    parser.add_argument("--days", type=int, nargs="+")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Small control file identifies actual new work, never an old experiment.
    started = time.monotonic()
    context = multiprocessing.get_context("spawn")
    summaries = {}
    if args.phase in ["all", "exposure"]:
        extracted = args.output / "source_exposures"
        extract_exposures(args.exposure_archive, extracted)
        with concurrent.futures.ProcessPoolExecutor(args.workers, mp_context=context) as pool:
            futures = [pool.submit(convert_day, day, "exposure", extracted, args.output, args.arrow_threads)
                       for day in (args.days or range(1, 38))]
            summaries["exposure"] = [future.result() for future in concurrent.futures.as_completed(futures)]
    if args.phase in ["all", "stage"]:
        with concurrent.futures.ProcessPoolExecutor(args.workers, mp_context=context) as pool:
            futures = [pool.submit(convert_day, day, "stage", args.stage_dir, args.output, args.arrow_threads)
                       for day in (args.days or range(24, 38))]
            summaries["stage"] = [future.result() for future in concurrent.futures.as_completed(futures)]
    json_save(args.output / f"conversion_{args.phase}.json", {"status": "passed", "summaries": summaries,
              "wall_seconds": time.monotonic() - started, "workers": args.workers,
              "arrow_threads_per_worker": args.arrow_threads})


if __name__ == "__main__":
    main()
