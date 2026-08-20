"""
Real online payment collection via Stripe Checkout - config-gated the
same way llm_client.py and email_service.py are: unset STRIPE_SECRET_KEY
means "not configured", and every caller already has a working
fallback for that ("mark paid" stays the only option, exactly as
before this file existed).

The actual card entry happens entirely on Stripe's own hosted Checkout
page, never in this app - this project has no business handling raw
card numbers (PCI scope), and Stripe Checkout is the standard way to
avoid ever needing to.
"""

import logging
from decimal import Decimal

import stripe

from app.core.config import STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET
from app.models.models import Invoice

logger = logging.getLogger(__name__)

# Stripe's own list of currencies with no fractional/smallest unit -
# the amount is passed as-is, not multiplied by 100 like USD cents.
# https://docs.stripe.com/currencies#zero-decimal
_ZERO_DECIMAL_CURRENCIES = {
    "bif", "clp", "djf", "gnf", "jpy", "kmf", "krw", "mga", "pyg",
    "rwf", "ugx", "vnd", "vuv", "xaf", "xof", "xpf",
}


class PaymentServiceError(Exception):
    """Raised on a Stripe API failure, or a webhook that fails
    signature verification (see verify_and_parse_webhook) - never
    silently swallowed, same discipline as EmailSendError."""


def is_configured() -> bool:
    return bool(STRIPE_SECRET_KEY)


def _amount_in_smallest_unit(amount: Decimal, currency: str) -> int:
    if currency.lower() in _ZERO_DECIMAL_CURRENCIES:
        return int(amount)
    return int((amount * 100).to_integral_value())


def create_checkout_session(invoice: Invoice, success_url: str, cancel_url: str) -> str:
    """Returns the Stripe-hosted Checkout URL to redirect the customer
    to. metadata.invoice_id is how the webhook handler below knows
    which invoice a completed session paid for - Stripe echoes it back
    unmodified on the event, it's never something the client can alter."""
    if not is_configured():
        raise PaymentServiceError("Stripe is not configured (STRIPE_SECRET_KEY unset).")

    try:
        session = stripe.checkout.Session.create(
            api_key=STRIPE_SECRET_KEY,
            mode="payment",
            line_items=[{
                "price_data": {
                    "currency": invoice.currency.lower(),
                    "product_data": {"name": f"Invoice #{invoice.invoice_number}"},
                    "unit_amount": _amount_in_smallest_unit(invoice.total, invoice.currency),
                },
                "quantity": 1,
            }],
            metadata={"invoice_id": str(invoice.id)},
            success_url=success_url,
            cancel_url=cancel_url,
        )
    except stripe.StripeError as e:
        logger.warning("Stripe Checkout Session creation failed for invoice %s: %s", invoice.id, e)
        raise PaymentServiceError(str(e)) from e

    return session.url


def verify_and_parse_webhook(payload: bytes, sig_header: str):
    """Confirms `payload` genuinely came from Stripe (HMAC signature
    check against STRIPE_WEBHOOK_SECRET) before the caller trusts
    anything in it - marking an invoice paid is exactly the kind of
    action that must never be spoofable by a POST from anyone else."""
    if not STRIPE_WEBHOOK_SECRET:
        raise PaymentServiceError("Stripe webhook secret is not configured (STRIPE_WEBHOOK_SECRET unset).")
    try:
        return stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe.SignatureVerificationError) as e:
        raise PaymentServiceError(f"Webhook verification failed: {e}") from e
