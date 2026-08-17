"""
Deterministic invoice math - no LLM, no agent, just arithmetic.
Per Section 7/15 of the project design: totals and validation are
plain code, on purpose. An LLM would be slower, costlier, and less
reliable than the four lines of arithmetic below.

This module is called directly by the invoices API router in
Phase 1. From Phase 2 onward, the Extraction Agent's output also
flows through these same functions before anything gets saved -
one validation implementation, used everywhere.
"""

from decimal import Decimal, ROUND_HALF_UP

from app.schemas.invoice import InvoiceItemCreate

TWO_PLACES = Decimal("0.01")


def calculate_line_total(quantity: Decimal, unit_price: Decimal) -> Decimal:
    return (quantity * unit_price).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)


def calculate_invoice_totals(
    items: list[InvoiceItemCreate], tax: Decimal, discount: Decimal
) -> tuple[Decimal, Decimal, list[Decimal]]:
    """Returns (subtotal, total, per-item line totals)."""
    line_totals = [calculate_line_total(item.quantity, item.unit_price) for item in items]
    subtotal = sum(line_totals, Decimal("0")).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
    total = (subtotal + tax - discount).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
    return subtotal, total, line_totals


class InvoiceValidationError(Exception):
    """Raised when an invoice fails a deterministic validation rule."""


def validate_invoice_input(items: list[InvoiceItemCreate], due_date, invoice_date) -> None:
    if not items:
        raise InvoiceValidationError("An invoice must have at least one line item.")
    if due_date < invoice_date:
        raise InvoiceValidationError("Due date cannot be before the invoice date.")
    for item in items:
        if item.quantity <= 0:
            raise InvoiceValidationError(f"Line item '{item.description}' has non-positive quantity.")
        if item.unit_price < 0:
            raise InvoiceValidationError(f"Line item '{item.description}' has a negative unit price.")
