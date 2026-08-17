"""
Unit tests for the deterministic invoice math (Section 15). None of
this touches the LLM or the database - it's pure arithmetic, so it
should be trivially, exhaustively testable, and it is.
"""

from datetime import date
from decimal import Decimal

import pytest

from app.schemas.invoice import InvoiceItemCreate
from app.services.validation_service import (
    calculate_line_total,
    calculate_invoice_totals,
    validate_invoice_input,
    InvoiceValidationError,
)


def test_calculate_line_total_basic():
    assert calculate_line_total(Decimal("3"), Decimal("2.50")) == Decimal("7.50")


def test_calculate_line_total_rounds_half_up():
    # 3 * 0.125 = 0.375 -> rounds to 0.38, not banker's-rounds to 0.37
    assert calculate_line_total(Decimal("3"), Decimal("0.125")) == Decimal("0.38")


def test_calculate_invoice_totals_sums_items_plus_tax_minus_discount():
    items = [
        InvoiceItemCreate(description="A", quantity=Decimal("2"), unit_price=Decimal("10.00")),
        InvoiceItemCreate(description="B", quantity=Decimal("1"), unit_price=Decimal("5.00")),
    ]
    subtotal, total, line_totals = calculate_invoice_totals(items, tax=Decimal("2.50"), discount=Decimal("1.00"))
    assert subtotal == Decimal("25.00")
    assert total == Decimal("26.50")  # 25.00 + 2.50 - 1.00
    assert line_totals == [Decimal("20.00"), Decimal("5.00")]


def test_calculate_invoice_totals_empty_items_gives_zero_subtotal():
    subtotal, total, line_totals = calculate_invoice_totals([], tax=Decimal("0"), discount=Decimal("0"))
    assert subtotal == Decimal("0.00")
    assert total == Decimal("0.00")
    assert line_totals == []


def test_validate_invoice_input_rejects_empty_items():
    with pytest.raises(InvoiceValidationError, match="at least one line item"):
        validate_invoice_input([], due_date=date(2026, 2, 1), invoice_date=date(2026, 1, 1))


def test_validate_invoice_input_rejects_due_before_invoice_date():
    items = [InvoiceItemCreate(description="A", quantity=Decimal("1"), unit_price=Decimal("1"))]
    with pytest.raises(InvoiceValidationError, match="Due date cannot be before"):
        validate_invoice_input(items, due_date=date(2026, 1, 1), invoice_date=date(2026, 2, 1))


def test_validate_invoice_input_rejects_non_positive_quantity():
    items = [InvoiceItemCreate(description="Bad item", quantity=Decimal("0"), unit_price=Decimal("5"))]
    with pytest.raises(InvoiceValidationError, match="non-positive quantity"):
        validate_invoice_input(items, due_date=date(2026, 2, 1), invoice_date=date(2026, 1, 1))


def test_validate_invoice_input_rejects_negative_unit_price():
    items = [InvoiceItemCreate(description="Bad item", quantity=Decimal("1"), unit_price=Decimal("-5"))]
    with pytest.raises(InvoiceValidationError, match="negative unit price"):
        validate_invoice_input(items, due_date=date(2026, 2, 1), invoice_date=date(2026, 1, 1))


def test_validate_invoice_input_accepts_equal_due_and_invoice_date():
    items = [InvoiceItemCreate(description="A", quantity=Decimal("1"), unit_price=Decimal("1"))]
    validate_invoice_input(items, due_date=date(2026, 1, 1), invoice_date=date(2026, 1, 1))  # should not raise
