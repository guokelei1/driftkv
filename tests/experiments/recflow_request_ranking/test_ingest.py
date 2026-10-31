import io
import tarfile
import zipfile

import pytest

from experiments.recflow_request_ranking.code.ingest_archive import extract_selected
from experiments.recflow_request_ranking.code.preflight import stage_inventory
from experiments.recflow_request_ranking.code.protocol import download_files


def tar_archive(path, entries):
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in entries:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def test_extract_required_only_retains_archive_and_verifies_resume(tmp_path):
    source = tmp_path / "all_stage.tar.gz"
    first, second = download_files()[:2]
    tar_archive(source, [("2024-01-13.feather", b"old"),
                         (f"realshow/{first}", b"wrong-source"),
                         (f"all_stage/{first}", b"first"), (f"all_stage/{second}", b"second")])
    destination = tmp_path / "output"
    report = extract_selected(source, destination, names=[first, second], reserve_bytes=0)
    assert source.exists() and {p.name for p in destination.iterdir()} == {first, second}
    assert (destination / first).read_bytes() == b"first"
    assert len(report["files"]) == 2 and not report["training_ready"]
    resumed = extract_selected(source, destination, names=[first, second], reserve_bytes=0)
    assert all(row["action"] == "verified_existing" for row in resumed["files"])
    (destination / first).write_bytes(b"different existing data")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        extract_selected(source, destination, names=[first, second], reserve_bytes=0)
    assert (destination / first).read_bytes() == b"different existing data"


def test_zip_supported_and_missing_dates_rejected(tmp_path):
    source = tmp_path / "all_stage.zip"
    first, second = download_files()[:2]
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(first, b"first")
    with pytest.raises(ValueError, match="missing required days"):
        extract_selected(source, tmp_path / "output", names=[first, second], reserve_bytes=0)
    assert (tmp_path / "output" / first).read_bytes() == b"first"


def test_duplicate_days_and_unsafe_paths_rejected(tmp_path):
    first = download_files()[0]
    source = tmp_path / "all_stage.tar.gz"
    tar_archive(source, [(first, b"first"), (first, b"duplicate")])
    with pytest.raises(ValueError, match="Duplicate selected day"):
        extract_selected(source, tmp_path / "duplicate", names=[first], reserve_bytes=0)
    tar_archive(source, [(f"../{first}", b"escape")])
    with pytest.raises(ValueError, match="Unsafe archive member"):
        extract_selected(source, tmp_path / "unsafe", names=[first], reserve_bytes=0)
    assert not (tmp_path / first).exists()


def test_disk_reserve_and_archive_discovery(tmp_path):
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    source = downloads / "all_stage.tar.gz"
    first = download_files()[0]
    tar_archive(source, [(first, b"first")])
    report = stage_inventory(tmp_path / "all_stage")
    assert report["status"] == "archive_available_requires_extraction"
    assert report["archives"] == [str(source)]
    with pytest.raises(OSError, match="Insufficient disk"):
        extract_selected(source, tmp_path / "output", names=[first], reserve_bytes=10**30)
    assert not list((tmp_path / "output").iterdir())
