"""
Tests for compute_priority()/rank_outstanding_receivables() - the
deterministic ranking logic, no LLM involved (the LLM only narrates
the already-decided order). Mirrors tests/test_payment_ap_agent.py's
structure since collections_agent.py is the receivables mirror of
payment_ap_agent.py.
"""

from datetime import date, timedelta
from decimal import Decimal

from app.agents.collections_agent import compute_priority, rank_outstanding_receivables, ESCALATE_RISK_THRESHOLD
from app.models.models import Invoice, InvoiceDirection, InvoiceStatus, PaymentStatus, Customer


def make_invoice(due_date, total=Decimal("100"), risk_score=None):
    return Invoice(
        organization_id=1, direction=InvoiceDirection.outgoing, invoice_number="X",
        customer_id=1, invoice_date=date.today(), due_date=due_date,
        subtotal=total, tax=Decimal("0"), discount=Decimal("0"), total=total,
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
        risk_score=Decimal(str(risk_score)) if risk_score is not None else None,
    )


def test_overdue_invoice_scores_higher_than_upcoming():
    overdue = make_invoice(due_date=date.today() - timedelta(days=5))
    upcoming = make_invoice(due_date=date.today() + timedelta(days=5))
    overdue_score, _ = compute_priority(overdue)
    upcoming_score, _ = compute_priority(upcoming)
    assert overdue_score > upcoming_score


def test_more_overdue_scores_higher():
    slightly_overdue = make_invoice(due_date=date.today() - timedelta(days=2))
    very_overdue = make_invoice(due_date=date.today() - timedelta(days=30))
    slight_score, _ = compute_priority(slightly_overdue)
    very_score, _ = compute_priority(very_overdue)
    assert very_score > slight_score


def test_larger_invoice_scores_higher_among_equally_overdue():
    """The real addition over Payment/AP's ranking: invoice size is a
    tie-breaker among invoices equally overdue - chasing a $50,000
    invoice 5 days late is more urgent than chasing a $50 one 5 days
    late."""
    small = make_invoice(due_date=date.today() - timedelta(days=5), total=Decimal("50"))
    large = make_invoice(due_date=date.today() - timedelta(days=5), total=Decimal("50000"))
    small_score, _ = compute_priority(small)
    large_score, _ = compute_priority(large)
    assert large_score > small_score


def test_amount_nudge_never_outranks_a_more_overdue_invoice():
    """The nudge is capped precisely so it can never cross from the
    not-yet-due band into the overdue band - a $1,000,000 invoice due
    next week must never outrank a $10 invoice that's already overdue."""
    huge_upcoming = make_invoice(due_date=date.today() + timedelta(days=5), total=Decimal("1000000"))
    tiny_overdue = make_invoice(due_date=date.today() - timedelta(days=1), total=Decimal("10"))
    huge_score, _ = compute_priority(huge_upcoming)
    tiny_score, _ = compute_priority(tiny_overdue)
    assert tiny_score > huge_score


def test_high_risk_invoice_is_escalated_but_still_scored():
    """Unlike Payment/AP's "hold back, don't pay" (which zeroes out
    urgency entirely), a risky customer still owes money and still
    needs chasing - escalation changes HOW you follow up, not whether
    the invoice is still ranked."""
    very_overdue_and_risky = make_invoice(due_date=date.today() - timedelta(days=60), risk_score=ESCALATE_RISK_THRESHOLD)
    score, escalate = compute_priority(very_overdue_and_risky)
    assert escalate is True
    assert score > 100  # still a real, positive urgency score, not zeroed/negative


def test_low_risk_invoice_is_not_escalated():
    invoice = make_invoice(due_date=date.today() - timedelta(days=5), risk_score=0.2)
    _, escalate = compute_priority(invoice)
    assert escalate is False


def test_risk_exactly_at_threshold_is_escalated():
    invoice = make_invoice(due_date=date.today(), risk_score=ESCALATE_RISK_THRESHOLD)
    _, escalate = compute_priority(invoice)
    assert escalate is True


def test_rank_outstanding_receivables_excludes_paid_and_incoming_invoices(db_session, org, customer, vendor):
    paid = Invoice(
        organization_id=org.id, direction=InvoiceDirection.outgoing, invoice_number="PAID-1",
        customer_id=customer.id, invoice_date=date.today(), due_date=date.today() - timedelta(days=5),
        subtotal=Decimal("100"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("100"),
        payment_status=PaymentStatus.paid, invoice_status=InvoiceStatus.validated,
    )
    incoming = Invoice(
        organization_id=org.id, direction=InvoiceDirection.incoming, invoice_number="VEND-1",
        vendor_id=vendor.id, invoice_date=date.today(), due_date=date.today() - timedelta(days=5),
        subtotal=Decimal("100"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("100"),
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
    )
    unpaid_outgoing = Invoice(
        organization_id=org.id, direction=InvoiceDirection.outgoing, invoice_number="OUT-1",
        customer_id=customer.id, invoice_date=date.today(), due_date=date.today() - timedelta(days=5),
        subtotal=Decimal("100"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("100"),
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
    )
    db_session.add_all([paid, incoming, unpaid_outgoing])
    db_session.commit()

    ranked, escalate = rank_outstanding_receivables(db_session, org.id)
    numbers = {inv.invoice_number for _, inv in ranked + escalate}
    assert numbers == {"OUT-1"}
