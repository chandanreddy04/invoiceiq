"""
Tests for generate_invoice_pdf() - deterministic, no LLM involved.
Verifies actual PDF content via PyMuPDF (already a dependency, used
elsewhere in this project for the opposite direction: reading an
uploaded PDF's text), not just that bytes came back.
"""

from datetime import date, timedelta
from decimal import Decimal

import fitz

from app.services.invoice_pdf_service import generate_invoice_pdf, BUSINESS_NAME
from app.models.models import Invoice, InvoiceItem, InvoiceDirection, InvoiceStatus, PaymentStatus


def make_outgoing_invoice(customer, **overrides):
    defaults = dict(
        organization_id=customer.organization_id, direction=InvoiceDirection.outgoing,
        invoice_number="PDF-TEST-1", customer_id=customer.id,
        invoice_date=date.today(), due_date=date.today() + timedelta(days=30),
        subtotal=Decimal("120.00"), tax=Decimal("10.00"), discount=Decimal("5.00"), total=Decimal("125.00"),
        currency="USD", payment_terms="Net 30",
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
    )
    defaults.update(overrides)
    return Invoice(**defaults)


def _text_of(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return doc[0].get_text()


def test_generates_valid_pdf_bytes(db_session, customer):
    invoice = make_outgoing_invoice(customer)
    invoice.items = [InvoiceItem(description="Widget", quantity=Decimal("2"), unit_price=Decimal("60.00"), line_total=Decimal("120.00"))]
    db_session.add(invoice)
    db_session.commit()
    db_session.refresh(invoice)

    pdf_bytes = generate_invoice_pdf(invoice)
    assert pdf_bytes[:5] == b"%PDF-"
    assert len(pdf_bytes) > 500


def test_pdf_contains_invoice_number_and_business_name(db_session, customer):
    invoice = make_outgoing_invoice(customer)
    invoice.items = [InvoiceItem(description="Widget", quantity=Decimal("2"), unit_price=Decimal("60.00"), line_total=Decimal("120.00"))]
    db_session.add(invoice)
    db_session.commit()
    db_session.refresh(invoice)

    text = _text_of(generate_invoice_pdf(invoice))
    assert BUSINESS_NAME in text
    assert invoice.invoice_number in text
    assert customer.name in text


def test_pdf_contains_line_items_and_totals(db_session, customer):
    invoice = make_outgoing_invoice(customer)
    invoice.items = [
        InvoiceItem(description="Consulting hours", quantity=Decimal("3"), unit_price=Decimal("40.00"), line_total=Decimal("120.00")),
    ]
    db_session.add(invoice)
    db_session.commit()
    db_session.refresh(invoice)

    text = _text_of(generate_invoice_pdf(invoice))
    assert "Consulting hours" in text
    assert "120.00" in text  # subtotal
    assert "125.00" in text  # total (subtotal + tax - discount)


def test_whole_number_quantity_renders_without_trailing_zeros(db_session, customer):
    invoice = make_outgoing_invoice(customer)
    invoice.items = [InvoiceItem(description="Item", quantity=Decimal("2.000"), unit_price=Decimal("60.00"), line_total=Decimal("120.00"))]
    db_session.add(invoice)
    db_session.commit()
    db_session.refresh(invoice)

    text = _text_of(generate_invoice_pdf(invoice))
    assert "2.000" not in text


def test_fractional_quantity_still_shows_its_decimal_part(db_session, customer):
    invoice = make_outgoing_invoice(customer)
    invoice.items = [InvoiceItem(description="Item", quantity=Decimal("2.500"), unit_price=Decimal("10.00"), line_total=Decimal("25.00"))]
    db_session.add(invoice)
    db_session.commit()
    db_session.refresh(invoice)

    text = _text_of(generate_invoice_pdf(invoice))
    assert "2.5" in text
