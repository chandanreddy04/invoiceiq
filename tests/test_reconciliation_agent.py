"""
Tests for the Reconciliation Agent's REASONING layer - matching is
100% deterministic (see reconciliation_agent.py's own docstring), so
these exercise score_match()/find_best_match()/reconcile_transaction()
directly against a real database, the same way test_fraud_risk_agent.py
tests compute_risk_signals()/score_risk().
"""

from datetime import date, timedelta
from decimal import Decimal

import pytest

from app.agents.reconciliation_agent import (
    reconcile_transaction, confirm_match, explain_unmatched_with_llm, MATCH_WINDOW_DAYS,
)
from app.models.models import (
    BankTransaction, Invoice, InvoiceDirection, InvoiceStatus, PaymentStatus, Payment,
)
from app.services.llm_client import LLMUnavailableError


def make_invoice(db_session, org, direction, total, due_date, vendor=None, customer=None, currency="USD", number="INV-1"):
    invoice = Invoice(
        organization_id=org.id, direction=direction, invoice_number=number,
        vendor_id=vendor.id if vendor else None, customer_id=customer.id if customer else None,
        invoice_date=due_date - timedelta(days=30), due_date=due_date,
        subtotal=total, tax=Decimal("0"), discount=Decimal("0"), total=total, currency=currency,
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
    )
    db_session.add(invoice)
    db_session.commit()
    db_session.refresh(invoice)
    return invoice


def make_txn(db_session, org, amount, txn_date, description="payment", currency="USD"):
    txn = BankTransaction(
        organization_id=org.id, transaction_date=txn_date, description=description,
        amount=amount, currency=currency,
    )
    db_session.add(txn)
    db_session.commit()
    db_session.refresh(txn)
    return txn


def test_exact_match_auto_matches_and_creates_payment(db_session, org, vendor):
    """A real gap this whole agent closes: the Payment table existed in
    the schema but nothing ever wrote to it - this is the assertion
    that pins that down."""
    due = date(2026, 6, 1)
    invoice = make_invoice(db_session, org, InvoiceDirection.incoming, Decimal("500.00"), due, vendor=vendor)
    txn = make_txn(db_session, org, Decimal("-500.00"), due + timedelta(days=5))

    reconcile_transaction(db_session, org.id, txn)
    db_session.commit()

    assert txn.status == "matched"
    assert txn.match_confidence == "exact"
    assert txn.matched_invoice_id == invoice.id

    db_session.refresh(invoice)
    assert invoice.payment_status == PaymentStatus.paid

    payment = db_session.query(Payment).filter(Payment.invoice_id == invoice.id).first()
    assert payment is not None
    assert payment.amount == Decimal("500.00")
    assert payment.paid_date == txn.transaction_date


def test_no_candidates_leaves_unmatched_with_no_suggestion(db_session, org, vendor):
    txn = make_txn(db_session, org, Decimal("-999.00"), date(2026, 6, 1))
    reconcile_transaction(db_session, org.id, txn)

    assert txn.status == "unmatched"
    assert txn.match_confidence is None
    assert txn.suggested_invoice_id is None


def test_multiple_exact_candidates_is_ambiguous_not_guessed(db_session, org, vendor):
    """Never picks between two equally-good candidates - a human must."""
    due = date(2026, 6, 1)
    make_invoice(db_session, org, InvoiceDirection.incoming, Decimal("300.00"), due, vendor=vendor, number="INV-A")
    make_invoice(db_session, org, InvoiceDirection.incoming, Decimal("300.00"), due, vendor=vendor, number="INV-B")
    txn = make_txn(db_session, org, Decimal("-300.00"), due)

    reconcile_transaction(db_session, org.id, txn)

    assert txn.status == "unmatched"
    assert txn.match_confidence == "ambiguous"
    assert txn.suggested_invoice_id is None


def test_amount_match_outside_window_is_suggested_not_auto_matched(db_session, org, vendor):
    due = date(2026, 1, 1)
    invoice = make_invoice(db_session, org, InvoiceDirection.incoming, Decimal("750.00"), due, vendor=vendor)
    txn = make_txn(db_session, org, Decimal("-750.00"), due + timedelta(days=MATCH_WINDOW_DAYS + 10))

    reconcile_transaction(db_session, org.id, txn)

    assert txn.status == "unmatched"
    assert txn.match_confidence == "likely"
    assert txn.suggested_invoice_id == invoice.id

    db_session.refresh(invoice)
    assert invoice.payment_status == PaymentStatus.unpaid  # never auto-matched


def test_negative_amount_never_matches_an_outgoing_customer_invoice(db_session, org, customer):
    """Direction guard: a vendor-bill payment (money out, negative) must
    never match against a customer invoice (money we're owed), even if
    the totals happen to coincide."""
    due = date(2026, 6, 1)
    make_invoice(db_session, org, InvoiceDirection.outgoing, Decimal("400.00"), due, customer=customer)
    txn = make_txn(db_session, org, Decimal("-400.00"), due)

    reconcile_transaction(db_session, org.id, txn)

    assert txn.status == "unmatched"
    assert txn.suggested_invoice_id is None


def test_currency_mismatch_is_never_a_candidate(db_session, org, vendor):
    due = date(2026, 6, 1)
    make_invoice(db_session, org, InvoiceDirection.incoming, Decimal("200.00"), due, vendor=vendor, currency="EUR")
    txn = make_txn(db_session, org, Decimal("-200.00"), due, currency="USD")

    reconcile_transaction(db_session, org.id, txn)

    assert txn.status == "unmatched"
    assert txn.suggested_invoice_id is None


def test_confirm_match_produces_the_same_payment_as_auto_match(db_session, org, vendor):
    due = date(2026, 1, 1)
    invoice = make_invoice(db_session, org, InvoiceDirection.incoming, Decimal("120.00"), due, vendor=vendor)
    txn = make_txn(db_session, org, Decimal("-120.00"), due + timedelta(days=MATCH_WINDOW_DAYS + 5))
    reconcile_transaction(db_session, org.id, txn)  # leaves it "likely", suggested

    confirm_match(db_session, invoice, txn)
    db_session.commit()

    assert txn.status == "matched"
    assert txn.matched_invoice_id == invoice.id
    db_session.refresh(invoice)
    assert invoice.payment_status == PaymentStatus.paid
    assert db_session.query(Payment).filter(Payment.invoice_id == invoice.id).count() == 1


def test_explain_unmatched_falls_back_cleanly_when_llm_unavailable(db_session, org, vendor, monkeypatch):
    import app.agents.reconciliation_agent as ra

    def _raise(*a, **kw):
        raise LLMUnavailableError("no model")
    monkeypatch.setattr(ra, "chat", _raise)

    txn = make_txn(db_session, org, Decimal("-50.00"), date(2026, 1, 1))
    no_suggestion = explain_unmatched_with_llm(txn, None)
    assert "no invoice on file" in no_suggestion.lower()

    invoice = make_invoice(db_session, org, InvoiceDirection.incoming, Decimal("50.00"), date(2026, 3, 1), vendor=vendor)
    with_suggestion = explain_unmatched_with_llm(txn, invoice)
    assert invoice.invoice_number in with_suggestion
