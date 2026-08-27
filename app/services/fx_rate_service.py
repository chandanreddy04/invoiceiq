"""
Live currency-conversion lookup - the one place in this project that
reaches outside its own database for a fact it can never store itself.
An exchange rate changes every day; baking a guessed rate into a
stored total would be exactly the kind of silent, unverifiable number
totals_by_currency()'s own docstring already refuses to produce ("never
add different currencies together into one number"). This module is
what lets financial_analysis_agent give a single combined total when
asked anyway - fetched fresh, shown once, never written back to the
database.

Deliberately a plain function, not an LLM tool call: "is there more
than one currency in this result" is a simple deterministic check, not
a judgment call, so nothing here is decided by a model - see
financial_analysis_agent.format_answer() for the only place this gets
used, and only to narrate an already-computed number, same "narrate,
never decide" boundary as every other LLM-facing feature in this app.

Frankfurter (frankfurter.dev) is free, requires no API key, and
sources rates from the European Central Bank - matching this project's
"free by default" pattern (Ollama, Groq's free tier) rather than
gating this feature behind yet another API key just to demo it.
"""

import logging
from decimal import Decimal

import httpx

logger = logging.getLogger(__name__)

FX_API_URL = "https://api.frankfurter.dev/v1/latest"

# No per-organization currency setting exists yet (Organization has no
# `currency` column) - USD is the same default used everywhere else in
# this codebase (Invoice.currency, etc.), so it's the honest default
# here too rather than inventing a new convention for one feature.
DEFAULT_TARGET_CURRENCY = "USD"


class FXRateUnavailableError(Exception):
    """Raised when the exchange-rate lookup fails or times out. The one
    caller of this (format_answer()) treats it as optional - it falls
    back to the existing per-currency breakdown rather than showing a
    combined total that's silently wrong or incomplete."""


def get_exchange_rate(from_currency: str, to_currency: str) -> Decimal:
    """Today's rate: 1 from_currency = ? to_currency. Same currency is
    always exactly 1 - short-circuited rather than making a pointless
    network call for it."""
    if from_currency == to_currency:
        return Decimal("1")
    try:
        resp = httpx.get(FX_API_URL, params={"base": from_currency, "symbols": to_currency}, timeout=5)
        resp.raise_for_status()
        rate = resp.json()["rates"][to_currency]
        return Decimal(str(rate))
    except Exception as e:
        logger.warning("Exchange rate lookup failed (%s -> %s): %s", from_currency, to_currency, e)
        raise FXRateUnavailableError(str(e)) from e


def convert_totals_to_single_currency(totals: dict[str, Decimal], target_currency: str = DEFAULT_TARGET_CURRENCY) -> Decimal | None:
    """Best-effort, fail-closed: returns None - never a partial or
    silently-wrong number - if any currency present can't be converted,
    so the caller can fall back to the existing per-currency breakdown
    instead of showing a total that's quietly missing a chunk of the
    real figure."""
    if not totals:
        return Decimal("0")
    combined = Decimal("0")
    for currency, amount in totals.items():
        try:
            rate = get_exchange_rate(currency, target_currency)
        except FXRateUnavailableError:
            return None
        combined += amount * rate
    return combined
