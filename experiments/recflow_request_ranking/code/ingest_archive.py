"""Extract only the 14 required daily files; retain the original archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from .protocol import download_files


def selected_name(name: str, expected: set[str]) -> str | None:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.name not in expected:
        return None
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive member: {name}")
    if len(path.parts) > 1 and "all_stage" not in path.parts[:-1]:
        # A whole RecFlow archive can also contain realshow/day.feather. Never
        # confuse those with the stage data sharing the same basename.
        return None
    return path.name


def write_daily(stream, destination: Path, expected_bytes: int, reserve_bytes: int) -> dict:
    exists = destination.exists()
    if destination.is_symlink() or (exists and not destination.is_file()):
        raise ValueError(f"Destination must be a regular file: {destination}")
    if not exists and shutil.disk_usage(destination.parent).free < expected_bytes + reserve_bytes:
        raise OSError(f"Insufficient disk space for {destination.name} with reserve")
    temporary = None
    output = None
    digest = hashlib.sha256()
    length = 0
    try:
        if not exists:
            output = tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", suffix=".partial",
                                                 dir=destination.parent, delete=False)
            temporary = Path(output.name)
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
            length += len(chunk)
            if output is not None:
                output.write(chunk)
        if length != expected_bytes:
            raise ValueError(f"Truncated member {destination.name}: {length} != {expected_bytes}")
        if exists:
            with destination.open("rb") as current:
                current_hash = hashlib.file_digest(current, "sha256").hexdigest()
            if current_hash != digest.hexdigest():
                raise ValueError(f"Existing file differs; refusing to overwrite {destination}")
        else:
            output.flush()
            os.fsync(output.fileno())
            output.close()
            output = None
            # Atomic publication without replacing a concurrently created file.
            os.link(temporary, destination)
        return {"path": str(destination), "bytes": length, "sha256": digest.hexdigest(),
                "action": "verified_existing" if exists else "extracted"}
    finally:
        if output is not None:
            output.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def extract_selected(archive: Path, destination: Path, *, names=None, reserve_bytes=20 * 1024**3,
                     on_record=None) -> dict:
    archive, destination = Path(archive), Path(destination)
    expected = set(download_files() if names is None else names)
    destination.mkdir(parents=True, exist_ok=True)
    seen, records = set(), []

    def consume(name, size, regular, opener):
        selected = selected_name(name, expected)
        if selected is None:
            return
        if not regular:
            raise ValueError(f"Selected member is not a regular file: {name}")
        if selected in seen:
            raise ValueError(f"Duplicate selected day in archive: {selected}")
        seen.add(selected)
        with opener() as stream:
            record = write_daily(stream, destination / selected, size, reserve_bytes)
        record["archive_member"] = name
        records.append(record)
        if on_record is not None:
            on_record(record)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                regular = not member.is_dir() and not stat.S_ISLNK(member.external_attr >> 16)
                consume(member.filename, member.file_size, regular, lambda m=member: source.open(m))
    else:
        # A gzip tar requires a sequential scan, but only selected files are
        # written. No extractall, no full uncompressed tar or all-days copy.
        with tarfile.open(archive, "r|*") as source:
            for member in source:
                consume(member.name, member.size, member.isfile(), lambda m=member: source.extractfile(m))
    missing = sorted(expected - seen)
    if missing:
        raise ValueError(f"Archive missing required days: {missing}; complete files retained for resume")
    return {"status": "selected_days_extracted_not_yet_candidate_audited", "archive": str(archive),
            "archive_bytes": archive.stat().st_size, "destination": str(destination),
            "files": records, "training_ready": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path("data/raw/recflow/all_stage"))
    parser.add_argument("--reserve-gib", type=float, default=20)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.reserve_gib < 0:
        raise SystemExit("reserve-gib must be nonnegative")
    if args.report.exists():
        raise SystemExit("Use a new report path; retain earlier extraction records")
    completed = []

    def progress(record):
        completed.append(record)
        print(json.dumps(record), flush=True)

    report = {"status": "extraction_interrupted", "completed": completed, "training_ready": False}
    try:
        report = extract_selected(args.archive, args.destination, reserve_bytes=int(args.reserve_gib * 1024**3),
                                  on_record=progress)
    except BaseException as error:
        report = {"status": "extraction_failed", "error": f"{type(error).__name__}: {error}", "completed": completed,
                  "archive": str(args.archive), "training_ready": False}
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.report), "files": len(report["files"]),
                      "training_ready": False}), flush=True)


if __name__ == "__main__":
    main()
