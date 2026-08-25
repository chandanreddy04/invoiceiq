"""
Tests for credit_note_service.py - deterministic, no LLM involved.
"""

from datetime import date, timedelta
from decimal import Decimal

import pytest

from app.models.models import Invoice, InvoiceDirection, InvoiceStatus, Payment, PaymentStatus
from app.services import credit_note_service


def make_outgoing_invoice(customer, total=Decimal("100.00")):
    return Invoice(
        organization_id=customer.organization_id, direction=InvoiceDirection.outgoing,
        invoice_number="CN-TEST-1", customer_id=customer.id,
        invoice_date=date.today(), due_date=date.today() + timedelta(days=30),
        subtotal=total, tax=Decimal("0"), discount=Decimal("0"), total=total,
        currency="USD", payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
    )


def test_suggest_next_credit_note_number_starts_at_0001(db_session, org):
    assert credit_note_service.suggest_next_credit_note_number(db_session, org.id) == "CN-0001"


def test_create_credit_note_saves_and_returns_it(db_session, org, customer):
    invoice = make_outgoing_invoice(customer)
    db_session.add(invoice)
    db_session.commit()
    db_session.refresh(invoice)

    note = credit_note_service.create_credit_note(db_session, org.id, invoice, reason="Returned item", amount=Decimal("30.00"), created_by="owner@test.example")

    assert note.credit_note_number == "CN-0001"
    assert note.amount == Decimal("30.00")
    assert note.currency == "USD"
    assert note.invoice_id == invoice.id


def test_create_credit_note_rejects_zero_or_negative_amount(db_session, org, customer):
    invoice = make_outgoing_invoice(customer)
    db_session.add(invoice)
    db_session.commit()
    db_session.refresh(invoice)

    with pytest.raises(credit_note_service.CreditNoteError):
        credit_note_service.create_credit_note(db_session, org.id, invoice, reason="X", amount=Decimal("0"), created_by="owner@test.example")

    with pytest.raises(credit_note_service.CreditNoteError):
        credit_note_service.create_credit_note(db_session, org.id, invoice, reason="X", amount=Decimal("-5"), created_by="owner@test.example")


def test_create_credit_note_rejects_amount_exceeding_invoice_total(db_session, org, customer):
    invoice = make_outgoing_invoice(customer, total=Decimal("100.00"))
    db_session.add(invoice)
    db_session.commit()
    db_session.refresh(invoice)

    with pytest.raises(credit_note_service.CreditNoteError, match="exceeds"):
        credit_note_service.create_credit_note(db_session, org.id, invoice, reason="X", amount=Decimal("150.00"), created_by="owner@test.example")


def test_create_credit_note_rejects_amount_exceeding_remaining_after_partial_credit(db_session, org, customer):
    """The real edge case: two credit notes together must never exceed
    the invoice total, even though each one alone would be valid."""
    invoice = make_outgoing_invoice(customer, total=Decimal("100.00"))
    db_session.add(invoice)
    db_session.commit()
    db_session.refresh(invoice)

    credit_note_service.create_credit_note(db_session, org.id, invoice, reason="First", amount=Decimal("70.00"), created_by="a@test.example")

    with pytest.raises(credit_note_service.CreditNoteError, match="exceeds"):
        credit_note_service.create_credit_note(db_session, org.id, invoice, reason="Second", amount=Decimal("50.00"), created_by="a@test.example")


def test_total_credited_sums_all_notes(db_session, org, customer):
    invoice = make_outgoing_invoice(customer, total=Decimal("100.00"))
    db_session.add(invoice)
    db_session.commit()
    db_session.refresh(invoice)

    credit_note_service.create_credit_note(db_session, org.id, invoice, reason="A", amount=Decimal("20.00"), created_by="a@test.example")
    credit_note_service.create_credit_note(db_session, org.id, invoice, reason="B", amount=Decimal("30.00"), created_by="a@test.example")

    assert credit_note_service.total_credited(db_session, invoice.id) == Decimal("50.00")
    assert credit_note_service.remaining_creditable(db_session, invoice) == Decimal("50.00")


def test_remaining_creditable_nets_out_payments_already_received(db_session, org, customer):
    """The gap this closes: a partially-paid invoice used to show the
    full total as still creditable, which both lied to the person
    filling out the form and would have let them issue a credit note
    bigger than what the customer actually still owes."""
    invoice = make_outgoing_invoice(customer, total=Decimal("1100.00"))
    db_session.add(invoice)
    db_session.commit()
    db_session.refresh(invoice)

    db_session.add(Payment(invoice_id=invoice.id, amount=Decimal("1050.00"), paid_date=date.today(), status="completed"))
    db_session.commit()

    assert credit_note_service.total_paid(db_session, invoice.id) == Decimal("1050.00")
    assert credit_note_service.remaining_creditable(db_session, invoice) == Decimal("50.00")

    with pytest.raises(credit_note_service.CreditNoteError, match="exceeds"):
        credit_note_service.create_credit_note(db_session, org.id, invoice, reason="X", amount=Decimal("200.00"), created_by="a@test.example")

    note = credit_note_service.create_credit_note(db_session, org.id, invoice, reason="Billing correction", amount=Decimal("50.00"), created_by="a@test.example")
    assert note.amount == Decimal("50.00")


def test_credit_note_never_changes_invoice_payment_status(db_session, org, customer):
    """Deliberate design decision: a credit note is a separate document,
    not the same fact as "we received money" - payment_status is only
    ever changed by an explicit human action elsewhere in this app."""
    invoice = make_outgoing_invoice(customer, total=Decimal("100.00"))
    db_session.add(invoice)
    db_session.commit()
    db_session.refresh(invoice)

    credit_note_service.create_credit_note(db_session, org.id, invoice, reason="Full refund", amount=Decimal("100.00"), created_by="a@test.example")

    assert invoice.payment_status == PaymentStatus.unpaid
