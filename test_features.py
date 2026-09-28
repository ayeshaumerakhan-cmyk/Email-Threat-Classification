from email.message import EmailMessage

import pytest

from src.email_threat.dataset import parse_message
from src.email_threat.features import FEATURE_NAMES, extract_features, feature_frame


def test_feature_extraction_captures_expected_email_signals():
    result = extract_features(
        subject="URGENT: Verify your account",
        body="YOU WON! Visit https://example.com now to verify your password.",
        sender="no-reply@secure123.example",
        attachment_count=2,
    )

    assert result["message_length_chars"] > 0
    assert result["url_count"] == 1
    assert result["attachment_count"] == 2
    assert result["subject_has_urgent"] == 1
    assert result["subject_has_account"] == 1
    assert result["body_has_winner"] == 1
    assert result["body_has_password"] == 1
    assert result["sender_is_noreply"] == 1
    assert result["sender_domain_has_digit"] == 1
    assert list(feature_frame(result).columns) == FEATURE_NAMES


def test_feature_extraction_rejects_negative_attachment_count():
    with pytest.raises(ValueError, match="non-negative"):
        extract_features(attachment_count=-1)


def test_message_parser_extracts_mime_attachment_and_body():
    message = EmailMessage()
    message["From"] = "sender@example.com"
    message["Subject"] = "Hello"
    message.set_content("Please review https://example.com")
    message.add_attachment(b"report", maintype="text", subtype="plain", filename="report.txt")

    subject, body, sender, attachment_count = parse_message(message.as_bytes())

    assert subject == "Hello"
    assert "https://example.com" in body
    assert sender == "sender@example.com"
    assert attachment_count == 1
