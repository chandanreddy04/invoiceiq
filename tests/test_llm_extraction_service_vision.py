"""
Tests for extract_invoice_from_image() - the vision counterpart to
extract_invoice_with_llm(). Same contract, so the same testing shape:
mock chat_with_image() directly (no real model call, no real image
needed) and check the schema-validation/fallback behavior, mirroring
how the rest of this project tests an agent's LLM-facing seam rather
than the model itself.
"""

import json

import pytest

from app.services.llm_extraction_service import extract_invoice_from_image
from app.services.llm_client import LLMUnavailableError


VALID_RESPONSE = json.dumps({
    "vendor_name": "Acme Corp", "vendor_address": None, "vendor_tax_id": None,
    "invoice_number": "AC-1", "invoice_date": "2026-01-01", "due_date": "2026-01-31",
    "currency": "USD", "tax": 0, "discount": 0, "line_items": [],
})


def test_extract_invoice_from_image_returns_validated_result(monkeypatch):
    import app.services.llm_extraction_service as svc
    monkeypatch.setattr(svc, "chat_with_image", lambda text, image_bytes, schema=None: VALID_RESPONSE)

    result = extract_invoice_from_image(b"fake-image-bytes")

    assert result.vendor_name == "Acme Corp"
    assert result.invoice_number == "AC-1"


def test_extract_invoice_from_image_propagates_llm_unavailable(monkeypatch):
    import app.services.llm_extraction_service as svc

    def _raise(*a, **kw):
        raise LLMUnavailableError("vision model not reachable")
    monkeypatch.setattr(svc, "chat_with_image", _raise)

    with pytest.raises(LLMUnavailableError):
        extract_invoice_from_image(b"fake-image-bytes")


def test_extract_invoice_from_image_raises_on_malformed_json(monkeypatch):
    """A vision model is more prone to drifting off-schema than a
    text-only structured-output call - this must surface as the same
    LLMUnavailableError every other bad-output case does, not crash."""
    import app.services.llm_extraction_service as svc
    monkeypatch.setattr(svc, "chat_with_image", lambda text, image_bytes, schema=None: "not json at all")

    with pytest.raises(LLMUnavailableError):
        extract_invoice_from_image(b"fake-image-bytes")
