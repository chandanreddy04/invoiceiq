"""
Tests for fx_rate_service.py - the one place this project reaches
outside its own database. No real network calls here; httpx.get is
mocked, same discipline as mocking ollama.chat elsewhere in this suite.
"""

from decimal import Decimal

import pytest

from app.services import fx_rate_service


class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json


def test_get_exchange_rate_same_currency_short_circuits_without_a_network_call(monkeypatch):
    def _fail(*a, **kw):
        raise AssertionError("should never call out for a same-currency rate")
    monkeypatch.setattr(fx_rate_service.httpx, "get", _fail)

    assert fx_rate_service.get_exchange_rate("USD", "USD") == Decimal("1")


def test_get_exchange_rate_returns_the_looked_up_rate(monkeypatch):
    monkeypatch.setattr(
        fx_rate_service.httpx, "get",
        lambda url, params, timeout: _FakeResponse({"amount": 1.0, "base": "EUR", "date": "2026-08-27", "rates": {"USD": 1.1645}}),
    )
    assert fx_rate_service.get_exchange_rate("EUR", "USD") == Decimal("1.1645")


def test_get_exchange_rate_raises_fx_unavailable_on_network_failure(monkeypatch):
    def _raise(*a, **kw):
        raise ConnectionError("no network")
    monkeypatch.setattr(fx_rate_service.httpx, "get", _raise)

    with pytest.raises(fx_rate_service.FXRateUnavailableError):
        fx_rate_service.get_exchange_rate("EUR", "USD")


def test_convert_totals_empty_dict_is_zero(monkeypatch):
    assert fx_rate_service.convert_totals_to_single_currency({}, "USD") == Decimal("0")


def test_convert_totals_combines_multiple_currencies(monkeypatch):
    def fake_rate(from_currency, to_currency):
        return {"EUR": Decimal("1.10"), "USD": Decimal("1")}[from_currency]
    monkeypatch.setattr(fx_rate_service, "get_exchange_rate", fake_rate)

    result = fx_rate_service.convert_totals_to_single_currency(
        {"EUR": Decimal("100.00"), "USD": Decimal("50.00")}, "USD",
    )
    assert result == Decimal("160.00")  # 100 * 1.10 + 50 * 1


def test_convert_totals_returns_none_fail_closed_when_any_rate_unavailable(monkeypatch):
    """Never show a partial/wrong combined total - if even one currency
    in the set can't be converted, the whole conversion is abandoned."""
    def flaky_rate(from_currency, to_currency):
        if from_currency == "GBP":
            raise fx_rate_service.FXRateUnavailableError("rate lookup failed")
        return Decimal("1.10")
    monkeypatch.setattr(fx_rate_service, "get_exchange_rate", flaky_rate)

    result = fx_rate_service.convert_totals_to_single_currency(
        {"EUR": Decimal("100.00"), "GBP": Decimal("50.00")}, "USD",
    )
    assert result is None
