"""Download and transform the public SpamAssassin raw email corpus."""

from __future__ import annotations

import csv
import codecs
import hashlib
import tarfile
import urllib.request
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Iterator

from src.email_threat.features import FEATURE_NAMES, extract_features

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_PATH = ROOT / "data" / "processed" / "email_features.csv"
CORPUS_BASE_URL = "https://spamassassin.apache.org/old/publiccorpus/"
ARCHIVES = {
    "20030228_spam.tar.bz2": 1,
    "20030228_easy_ham.tar.bz2": 0,
    "20030228_hard_ham.tar.bz2": 0,
}
CSV_COLUMNS = ["label", *FEATURE_NAMES, "source_archive", "message_id"]


def _download_archive(name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / name
    if target.is_file() and target.stat().st_size > 0:
        return target

    request = urllib.request.Request(
        CORPUS_BASE_URL + name, headers={"User-Agent": "email-threat-capstone/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
    except Exception as exc:
        raise RuntimeError(f"Could not download {name} from {CORPUS_BASE_URL}") from exc
    if not content:
        raise RuntimeError(f"Downloaded an empty archive: {name}")
    target.write_bytes(content)
    return target


def _text_part(part) -> str:
    try:
        content = part.get_content()
    except (LookupError, UnicodeError, TypeError):
        payload = part.get_payload(decode=True)
        if payload is None:
            return ""
        charset = part.get_content_charset() or "utf-8"
        try:
            codecs.lookup(charset)
        except LookupError:
            charset = "utf-8"
        content = payload.decode(charset, errors="replace")
    return content if isinstance(content, str) else ""


def parse_message(raw: bytes) -> tuple[str, str, str, int]:
    message = BytesParser(policy=policy.default).parsebytes(raw)
    subject = str(message.get("subject", ""))
    sender = str(message.get("from", ""))
    attachment_count = 0
    body_parts: list[str] = []

    for part in message.walk():
        if part.is_multipart():
            continue
        if part.get_content_disposition() == "attachment" or part.get_filename():
            attachment_count += 1
        elif part.get_content_maintype() == "text":
            body_parts.append(_text_part(part))

    return subject, "\n".join(body_parts), sender, attachment_count


def iter_dataset_rows() -> Iterator[dict[str, object]]:
    """Yield labeled, engineered rows from all configured source archives."""
    for archive_name, label in ARCHIVES.items():
        archive_path = _download_archive(archive_name)
        try:
            with tarfile.open(archive_path, mode="r:bz2") as archive:
                members = (member for member in archive if member.isfile())
                for member in members:
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        continue
                    raw = extracted.read()
                    subject, body, sender, attachment_count = parse_message(raw)
                    row: dict[str, object] = {
                        "label": label,
                        **extract_features(subject, body, sender, attachment_count),
                        "source_archive": archive_name,
                        "message_id": hashlib.sha256(raw).hexdigest()[:16],
                    }
                    yield row
        except (OSError, tarfile.TarError) as exc:
            raise RuntimeError(f"Could not read corpus archive {archive_path}") from exc


def build_dataset(output_path: Path = OUTPUT_PATH) -> int:
    """Build the derived CSV and return its number of email records."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in iter_dataset_rows():
            writer.writerow(row)
            count += 1
    if count == 0:
        raise RuntimeError("No messages were found in the configured corpus archives.")
    return count


if __name__ == "__main__":
    records = build_dataset()
    print(f"Wrote {records:,} email records to {OUTPUT_PATH}")
