"""
Tests for the config-gated Stripe integration. No real Stripe account
needed - the Checkout Session call is mocked, and webhook signature
verification is tested against Stripe's own SDK-generated signatures
(stripe.WebhookSignature.generate_signature_header), the same helper
Stripe's own test suite uses - so this proves real signature
verification works, not just that some function returns True.
"""

import json

import pytest
import stripe

from app.services import payment_service
from app.models.models import Invoice, InvoiceDirection, InvoiceStatus, PaymentStatus
from datetime import date, timedelta
from decimal import Decimal


def make_outgoing_invoice(**overrides):
    defaults = dict(
        organization_id=1, direction=InvoiceDirection.outgoing, invoice_number="PAY-TEST-1",
        customer_id=1, invoice_date=date.today(), due_date=date.today() + timedelta(days=30),
        subtotal=Decimal("100.00"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("100.00"),
        currency="USD", payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
    )
    defaults.update(overrides)
    return Invoice(**defaults)


def test_not_configured_by_default(monkeypatch):
    monkeypatch.setattr(payment_service, "STRIPE_SECRET_KEY", "")
    assert payment_service.is_configured() is False


def test_configured_when_key_set(monkeypatch):
    monkeypatch.setattr(payment_service, "STRIPE_SECRET_KEY", "sk_test_fake")
    assert payment_service.is_configured() is True


def test_create_checkout_session_raises_when_not_configured(monkeypatch):
    monkeypatch.setattr(payment_service, "STRIPE_SECRET_KEY", "")
    with pytest.raises(payment_service.PaymentServiceError):
        payment_service.create_checkout_session(make_outgoing_invoice(), "http://x/success", "http://x/cancel")


def test_create_checkout_session_uses_correct_amount_and_currency(monkeypatch):
    monkeypatch.setattr(payment_service, "STRIPE_SECRET_KEY", "sk_test_fake")
    captured = {}

    class FakeSession:
        url = "https://checkout.stripe.com/fake-session"

    def fake_create(**kwargs):
        captured.update(kwargs)
        return FakeSession()

    monkeypatch.setattr(stripe.checkout.Session, "create", staticmethod(fake_create))

    invoice = make_outgoing_invoice(total=Decimal("49.99"), currency="USD")
    url = payment_service.create_checkout_session(invoice, "http://x/success", "http://x/cancel")

    assert url == "https://checkout.stripe.com/fake-session"
    assert captured["line_items"][0]["price_data"]["unit_amount"] == 4999  # cents
    assert captured["line_items"][0]["price_data"]["currency"] == "usd"
    assert captured["metadata"]["invoice_id"] == str(invoice.id)


def test_create_checkout_session_handles_zero_decimal_currency(monkeypatch):
    """JPY has no cents - Stripe expects the raw integer amount, not x100."""
    monkeypatch.setattr(payment_service, "STRIPE_SECRET_KEY", "sk_test_fake")
    captured = {}

    class FakeSession:
        url = "https://checkout.stripe.com/fake-session"

    def fake_create(**kwargs):
        captured.update(kwargs)
        return FakeSession()

    monkeypatch.setattr(stripe.checkout.Session, "create", staticmethod(fake_create))

    invoice = make_outgoing_invoice(total=Decimal("500"), currency="JPY")
    payment_service.create_checkout_session(invoice, "http://x/success", "http://x/cancel")
    assert captured["line_items"][0]["price_data"]["unit_amount"] == 500


def test_create_checkout_session_wraps_stripe_errors(monkeypatch):
    monkeypatch.setattr(payment_service, "STRIPE_SECRET_KEY", "sk_test_fake")

    def failing_create(**kwargs):
        raise stripe.InvalidRequestError("Invalid API Key", param=None)

    monkeypatch.setattr(stripe.checkout.Session, "create", staticmethod(failing_create))

    with pytest.raises(payment_service.PaymentServiceError):
        payment_service.create_checkout_session(make_outgoing_invoice(), "http://x/success", "http://x/cancel")


def test_verify_webhook_accepts_genuinely_signed_payload(monkeypatch):
    monkeypatch.setattr(payment_service, "STRIPE_WEBHOOK_SECRET", "whsec_test_secret")
    payload = json.dumps({
        "id": "evt_test", "type": "checkout.session.completed",
        "data": {"object": {"id": "cs_test", "metadata": {"invoice_id": "42"}}},
    })
    sig_header = stripe.WebhookSignature.generate_signature_header(payload, "whsec_test_secret")

    event = payment_service.verify_and_parse_webhook(payload.encode(), sig_header)
    assert event["type"] == "checkout.session.completed"
    assert event["data"]["object"]["metadata"]["invoice_id"] == "42"


def test_verify_webhook_rejects_wrong_secret(monkeypatch):
    monkeypatch.setattr(payment_service, "STRIPE_WEBHOOK_SECRET", "whsec_real_secret")
    payload = json.dumps({"id": "evt_test", "type": "checkout.session.completed", "data": {"object": {}}})
    # Signed with a DIFFERENT secret than what's configured - simulates a spoofed request.
    sig_header = stripe.WebhookSignature.generate_signature_header(payload, "whsec_attacker_secret")

    with pytest.raises(payment_service.PaymentServiceError, match="verification failed"):
        payment_service.verify_and_parse_webhook(payload.encode(), sig_header)


def test_verify_webhook_raises_when_secret_not_configured(monkeypatch):
    monkeypatch.setattr(payment_service, "STRIPE_WEBHOOK_SECRET", "")
    with pytest.raises(payment_service.PaymentServiceError, match="not configured"):
        payment_service.verify_and_parse_webhook(b"{}", "some-sig")
