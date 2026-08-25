"""
Tests for the Cash Application Agent's REASONING and ACTION layers - both
are 100% deterministic (see cash_application_agent.py's own docstring),
so these exercise find_split_candidates()/find_partial_candidate()/
propose_allocation()/apply_allocation() directly against a real database,
the same way test_reconciliation_agent.py tests its sibling agent.
"""

from datetime import date, timedelta
from decimal import Decimal

from app.agents.cash_application_agent import (
    find_split_candidates, find_partial_candidate, propose_allocation,
    apply_allocation, run_cash_application, explain_allocation_with_llm,
    MIN_PARTIAL_FRACTION,
)
from app.models.models import (
    BankTransaction, Invoice, InvoiceDirection, InvoiceStatus, PaymentStatus, Payment,
    SuggestedAllocation, Customer,
)
from app.services.llm_client import LLMUnavailableError


def make_invoice(db_session, org, customer, total, due_date, currency="USD", number="INV-1"):
    invoice = Invoice(
        organization_id=org.id, direction=InvoiceDirection.outgoing, invoice_number=number,
        customer_id=customer.id,
        invoice_date=due_date - timedelta(days=30), due_date=due_date,
        subtotal=total, tax=Decimal("0"), discount=Decimal("0"), total=total, currency=currency,
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
    )
    db_session.add(invoice)
    db_session.commit()
    db_session.refresh(invoice)
    return invoice


def make_txn(db_session, org, amount, txn_date=date(2026, 6, 1), description="payment", currency="USD"):
    txn = BankTransaction(
        organization_id=org.id, transaction_date=txn_date, description=description,
        amount=amount, currency=currency,
    )
    db_session.add(txn)
    db_session.commit()
    db_session.refresh(txn)
    return txn


def test_split_finds_exact_sum_across_two_invoices(db_session, org, customer):
    inv_a = make_invoice(db_session, org, customer, Decimal("100.00"), date(2026, 6, 1), number="INV-A")
    inv_b = make_invoice(db_session, org, customer, Decimal("250.00"), date(2026, 6, 1), number="INV-B")
    txn = make_txn(db_session, org, Decimal("350.00"))

    result = find_split_candidates(db_session, org.id, txn)

    assert result is not None
    assert {inv.id for inv in result} == {inv_a.id, inv_b.id}


def test_split_never_mixes_two_customers_invoices(db_session, org):
    customer_a = Customer(organization_id=org.id, name="Customer A", email="a@test.example")
    customer_b = Customer(organization_id=org.id, name="Customer B", email="b@test.example")
    db_session.add_all([customer_a, customer_b])
    db_session.commit()
    db_session.refresh(customer_a)
    db_session.refresh(customer_b)

    make_invoice(db_session, org, customer_a, Decimal("100.00"), date(2026, 6, 1), number="INV-A")
    make_invoice(db_session, org, customer_b, Decimal("250.00"), date(2026, 6, 1), number="INV-B")
    txn = make_txn(db_session, org, Decimal("350.00"))

    result = find_split_candidates(db_session, org.id, txn)

    assert result is None  # 100 + 250 only sums correctly across customers, which never counts


def test_split_returns_none_for_a_non_positive_amount(db_session, org, customer):
    make_invoice(db_session, org, customer, Decimal("100.00"), date(2026, 6, 1), number="INV-A")
    make_invoice(db_session, org, customer, Decimal("250.00"), date(2026, 6, 1), number="INV-B")
    txn = make_txn(db_session, org, Decimal("-350.00"))

    assert find_split_candidates(db_session, org.id, txn) is None


def test_partial_candidate_requires_minimum_fraction(db_session, org, customer):
    invoice = make_invoice(db_session, org, customer, Decimal("1000.00"), date(2026, 6, 1))
    too_small = make_txn(db_session, org, invoice.total * MIN_PARTIAL_FRACTION - Decimal("0.01"))

    assert find_partial_candidate(db_session, org.id, too_small) is None

    big_enough = make_txn(db_session, org, invoice.total * MIN_PARTIAL_FRACTION)
    assert find_partial_candidate(db_session, org.id, big_enough) is not None


def test_partial_candidate_is_none_when_multiple_invoices_could_match(db_session, org, customer):
    make_invoice(db_session, org, customer, Decimal("1000.00"), date(2026, 6, 1), number="INV-A")
    make_invoice(db_session, org, customer, Decimal("900.00"), date(2026, 6, 1), number="INV-B")
    txn = make_txn(db_session, org, Decimal("500.00"))  # a plausible partial toward either

    assert find_partial_candidate(db_session, org.id, txn) is None


def test_propose_allocation_prefers_split_over_partial(db_session, org, customer):
    """An exact-sum split is stronger evidence than a fractional guess,
    so it must win even when a lone partial candidate also exists."""
    inv_a = make_invoice(db_session, org, customer, Decimal("100.00"), date(2026, 6, 1), number="INV-A")
    inv_b = make_invoice(db_session, org, customer, Decimal("250.00"), date(2026, 6, 1), number="INV-B")
    txn = make_txn(db_session, org, Decimal("350.00"))

    propose_allocation(db_session, org.id, txn)
    db_session.commit()

    allocations = db_session.query(SuggestedAllocation).filter(SuggestedAllocation.bank_transaction_id == txn.id).all()
    assert {a.invoice_id for a in allocations} == {inv_a.id, inv_b.id}
    assert all(a.kind == "split_share" for a in allocations)


def test_propose_allocation_clears_stale_proposal_first(db_session, org, customer):
    invoice = make_invoice(db_session, org, customer, Decimal("500.00"), date(2026, 6, 1))
    txn = make_txn(db_session, org, Decimal("500.00") * MIN_PARTIAL_FRACTION)
    propose_allocation(db_session, org.id, txn)
    db_session.commit()
    first_pass_count = db_session.query(SuggestedAllocation).filter(SuggestedAllocation.bank_transaction_id == txn.id).count()
    assert first_pass_count == 1

    propose_allocation(db_session, org.id, txn)  # re-running must not duplicate rows
    db_session.commit()
    assert db_session.query(SuggestedAllocation).filter(SuggestedAllocation.bank_transaction_id == txn.id).count() == 1


def test_apply_allocation_split_pays_each_invoice_in_full(db_session, org, customer):
    inv_a = make_invoice(db_session, org, customer, Decimal("100.00"), date(2026, 6, 1), number="INV-A")
    inv_b = make_invoice(db_session, org, customer, Decimal("250.00"), date(2026, 6, 1), number="INV-B")
    txn = make_txn(db_session, org, Decimal("350.00"))
    propose_allocation(db_session, org.id, txn)
    db_session.commit()

    apply_allocation(db_session, txn)
    db_session.commit()

    db_session.refresh(inv_a)
    db_session.refresh(inv_b)
    assert inv_a.payment_status == PaymentStatus.paid
    assert inv_b.payment_status == PaymentStatus.paid
    assert txn.status == "matched"
    assert txn.matched_invoice_id is None  # ambiguous which single invoice - only set for single-allocation case
    assert db_session.query(Payment).filter(Payment.bank_transaction_id == txn.id).count() == 2
    assert db_session.query(SuggestedAllocation).filter(SuggestedAllocation.bank_transaction_id == txn.id).count() == 0


def test_apply_allocation_partial_sets_partially_paid(db_session, org, customer):
    invoice = make_invoice(db_session, org, customer, Decimal("1000.00"), date(2026, 6, 1))
    txn = make_txn(db_session, org, Decimal("300.00"))
    propose_allocation(db_session, org.id, txn)
    db_session.commit()

    apply_allocation(db_session, txn)
    db_session.commit()

    db_session.refresh(invoice)
    assert invoice.payment_status == PaymentStatus.partially_paid
    assert txn.status == "matched"
    assert txn.matched_invoice_id == invoice.id  # single allocation - unambiguous
    payment = db_session.query(Payment).filter(Payment.invoice_id == invoice.id).first()
    assert payment.amount == Decimal("300.00")
    assert payment.bank_transaction_id == txn.id


def test_run_cash_application_skips_transactions_reconciliation_already_suggested(db_session, org, customer):
    invoice = make_invoice(db_session, org, customer, Decimal("1000.00"), date(2026, 6, 1))
    txn = make_txn(db_session, org, Decimal("300.00"))
    txn.suggested_invoice_id = invoice.id  # pretend Reconciliation already has an answer
    db_session.commit()

    result = run_cash_application(db_session, org.id)

    assert result == {"scanned": 0, "proposed": 0}
    assert db_session.query(SuggestedAllocation).filter(SuggestedAllocation.bank_transaction_id == txn.id).count() == 0


def test_run_cash_application_scans_and_proposes(db_session, org, customer):
    make_invoice(db_session, org, customer, Decimal("1000.00"), date(2026, 6, 1))
    make_txn(db_session, org, Decimal("300.00"))  # a valid partial candidate, no Reconciliation suggestion

    result = run_cash_application(db_session, org.id)

    assert result == {"scanned": 1, "proposed": 1}


def test_explain_allocation_falls_back_cleanly_when_llm_unavailable(db_session, org, customer, monkeypatch):
    import app.agents.cash_application_agent as caa

    def _raise(*a, **kw):
        raise LLMUnavailableError("no model")
    monkeypatch.setattr(caa, "chat", _raise)

    inv_a = make_invoice(db_session, org, customer, Decimal("100.00"), date(2026, 6, 1), number="INV-A")
    inv_b = make_invoice(db_session, org, customer, Decimal("250.00"), date(2026, 6, 1), number="INV-B")
    txn = make_txn(db_session, org, Decimal("350.00"))
    propose_allocation(db_session, org.id, txn)
    db_session.commit()
    allocations = db_session.query(SuggestedAllocation).filter(SuggestedAllocation.bank_transaction_id == txn.id).all()

    split_explanation = explain_allocation_with_llm(txn, allocations)
    assert "2 invoices" in split_explanation

    apply_allocation(db_session, txn)  # settle inv_a/inv_b so they can't also count as partial candidates below
    db_session.commit()

    partial_invoice = make_invoice(db_session, org, customer, Decimal("500.00"), date(2026, 6, 1), number="INV-C")
    single_txn = make_txn(db_session, org, Decimal("50.00"), description="partial")
    propose_allocation(db_session, org.id, single_txn)
    db_session.commit()
    single_alloc = db_session.query(SuggestedAllocation).filter(SuggestedAllocation.bank_transaction_id == single_txn.id).all()
    assert len(single_alloc) == 1
    partial_explanation = explain_allocation_with_llm(single_txn, single_alloc)
    assert "partial payment" in partial_explanation.lower()
