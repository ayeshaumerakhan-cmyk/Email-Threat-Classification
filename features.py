"""Shared, deterministic feature extraction for training and inference."""

from __future__ import annotations

import html
import re
from collections.abc import Mapping

import pandas as pd

FEATURE_NAMES = [
    "message_length_chars",
    "word_count",
    "url_count",
    "attachment_count",
    "subject_has_urgent",
    "subject_has_account",
    "body_has_urgent",
    "body_has_verify",
    "body_has_winner",
    "body_has_password",
    "sender_is_noreply",
    "sender_domain_has_digit",
    "has_html",
    "subject_length_chars",
    "uppercase_ratio",
]

URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)
WORD_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
URGENT_TERMS = ("urgent", "immediately", "action required", "act now")


def extract_features(
    subject: str = "",
    body: str = "",
    sender: str = "",
    attachment_count: int = 0,
) -> dict[str, float]:
    """Extract a stable numeric feature row from basic email fields."""
    if attachment_count < 0:
        raise ValueError("attachment_count must be non-negative")

    subject_text = str(subject or "")
    raw_body = str(body or "")
    decoded_body = html.unescape(raw_body)
    has_html = bool(HTML_TAG_PATTERN.search(decoded_body))
    body_text = HTML_TAG_PATTERN.sub(" ", decoded_body)
    body_lower = body_text.casefold()
    subject_lower = subject_text.casefold()
    sender_lower = str(sender or "").casefold()
    sender_domain = sender_lower.rsplit("@", 1)[-1] if "@" in sender_lower else ""
    words = WORD_PATTERN.findall(body_text)
    letters = [char for char in body_text if char.isalpha()]
    uppercase_ratio = (
        sum(char.isupper() for char in letters) / len(letters) if letters else 0.0
    )

    return {
        "message_length_chars": float(len(body_text)),
        "word_count": float(len(words)),
        "url_count": float(len(URL_PATTERN.findall(body_text))),
        "attachment_count": float(attachment_count),
        "subject_has_urgent": float(any(term in subject_lower for term in URGENT_TERMS)),
        "subject_has_account": float("account" in subject_lower),
        "body_has_urgent": float(any(term in body_lower for term in URGENT_TERMS)),
        "body_has_verify": float(any(term in body_lower for term in ("verify", "verification"))),
        "body_has_winner": float(
            "winner" in body_lower
            or "won a prize" in body_lower
            or bool(re.search(r"\bwon\b", body_lower))
        ),
        "body_has_password": float("password" in body_lower),
        "sender_is_noreply": float("no-reply" in sender_lower or "noreply" in sender_lower),
        "sender_domain_has_digit": float(any(char.isdigit() for char in sender_domain)),
        "has_html": float(has_html),
        "subject_length_chars": float(len(subject_text)),
        "uppercase_ratio": float(uppercase_ratio),
    }


def feature_frame(features: Mapping[str, float]) -> pd.DataFrame:
    """Return one feature row in the exact order expected by the model."""
    return pd.DataFrame([{name: float(features[name]) for name in FEATURE_NAMES}])
