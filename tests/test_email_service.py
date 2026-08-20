"""
Tests for the config-gated SMTP email service. Real network calls are
never made - smtplib.SMTP is mocked - these test the gating logic and
error handling, not actual mail delivery.
"""

import pytest

from app.services import email_service


def test_not_configured_by_default(monkeypatch):
    monkeypatch.setattr(email_service, "SMTP_HOST", "")
    monkeypatch.setattr(email_service, "SMTP_USER", "")
    monkeypatch.setattr(email_service, "SMTP_PASSWORD", "")
    assert email_service.is_configured() is False


def test_configured_when_all_three_set(monkeypatch):
    monkeypatch.setattr(email_service, "SMTP_HOST", "smtp.example.com")
    monkeypatch.setattr(email_service, "SMTP_USER", "user@example.com")
    monkeypatch.setattr(email_service, "SMTP_PASSWORD", "secret")
    assert email_service.is_configured() is True


def test_partially_configured_is_still_not_configured(monkeypatch):
    """A host with no credentials should never attempt a real send -
    that would just fail loudly with a confusing auth error instead of
    a clear "not configured" state."""
    monkeypatch.setattr(email_service, "SMTP_HOST", "smtp.example.com")
    monkeypatch.setattr(email_service, "SMTP_USER", "")
    monkeypatch.setattr(email_service, "SMTP_PASSWORD", "")
    assert email_service.is_configured() is False


def test_send_email_raises_when_not_configured(monkeypatch):
    monkeypatch.setattr(email_service, "SMTP_HOST", "")
    with pytest.raises(email_service.EmailSendError):
        email_service.send_email("to@example.com", "Subject", "Body")


def test_send_email_uses_smtp_with_starttls_and_login(monkeypatch):
    monkeypatch.setattr(email_service, "SMTP_HOST", "smtp.example.com")
    monkeypatch.setattr(email_service, "SMTP_PORT", 587)
    monkeypatch.setattr(email_service, "SMTP_USER", "user@example.com")
    monkeypatch.setattr(email_service, "SMTP_PASSWORD", "secret")
    monkeypatch.setattr(email_service, "SMTP_FROM_EMAIL", "user@example.com")

    calls = {"starttls": 0, "login": None, "sent": None}

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            calls["host"] = host
            calls["port"] = port

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            calls["starttls"] += 1

        def login(self, user, password):
            calls["login"] = (user, password)

        def send_message(self, msg):
            calls["sent"] = msg

    import smtplib
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)

    email_service.send_email("to@example.com", "Hello", "Body text")

    assert calls["host"] == "smtp.example.com"
    assert calls["starttls"] == 1
    assert calls["login"] == ("user@example.com", "secret")
    assert calls["sent"]["To"] == "to@example.com"
    assert calls["sent"]["Subject"] == "Hello"


def test_send_email_with_attachment_includes_pdf(monkeypatch):
    monkeypatch.setattr(email_service, "SMTP_HOST", "smtp.example.com")
    monkeypatch.setattr(email_service, "SMTP_USER", "user@example.com")
    monkeypatch.setattr(email_service, "SMTP_PASSWORD", "secret")

    sent = {}

    class FakeSMTP:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self): pass
        def login(self, *a): pass
        def send_message(self, msg): sent["msg"] = msg

    import smtplib
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)

    email_service.send_email("to@example.com", "Invoice", "See attached", attachment=("INV-0001.pdf", b"%PDF-fake"))

    attachments = [part for part in sent["msg"].iter_attachments()]
    assert len(attachments) == 1
    assert attachments[0].get_filename() == "INV-0001.pdf"


def test_send_email_wraps_smtp_failures(monkeypatch):
    monkeypatch.setattr(email_service, "SMTP_HOST", "smtp.example.com")
    monkeypatch.setattr(email_service, "SMTP_USER", "user@example.com")
    monkeypatch.setattr(email_service, "SMTP_PASSWORD", "wrong-password")

    class FailingSMTP:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self): pass
        def login(self, *a):
            raise Exception("535 Authentication failed")
        def send_message(self, msg): pass

    import smtplib
    monkeypatch.setattr(smtplib, "SMTP", FailingSMTP)

    with pytest.raises(email_service.EmailSendError, match="Authentication failed"):
        email_service.send_email("to@example.com", "Subject", "Body")
