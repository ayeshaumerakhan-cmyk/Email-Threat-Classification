import io
import tarfile
from pathlib import Path

import src.email_threat.dataset as dataset


def test_corpus_rows_are_labeled_and_featured(monkeypatch, tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    archive_path = raw_dir / "sample.tar.bz2"
    message = (
        b"From: sender@example.com\r\n"
        b"Subject: Free prize\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n\r\n"
        b"You won a prize. Visit https://example.com\r\n"
    )
    with tarfile.open(archive_path, mode="w:bz2") as archive:
        info = tarfile.TarInfo(name="sample/1")
        info.size = len(message)
        archive.addfile(info, io.BytesIO(message))

    monkeypatch.setattr(dataset, "RAW_DIR", raw_dir)
    monkeypatch.setattr(dataset, "ARCHIVES", {"sample.tar.bz2": 1})

    rows = list(dataset.iter_dataset_rows())

    assert len(rows) == 1
    assert rows[0]["label"] == 1
    assert rows[0]["url_count"] == 1
    assert rows[0]["body_has_winner"] == 1
    assert rows[0]["source_archive"] == "sample.tar.bz2"
