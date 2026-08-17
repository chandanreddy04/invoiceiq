"""
Tests for compute_priority() - the deterministic ranking logic, no
LLM involved (the LLM only narrates the already-decided order).
"""

from datetime import date, timedelta
from decimal import Decimal

from app.agents.payment_ap_agent import compute_priority, RISK_HOLD_THRESHOLD
from app.models.models import Invoice, InvoiceDirection, InvoiceStatus, PaymentStatus


def make_invoice(due_date, risk_score=None):
    return Invoice(
        organization_id=1, direction=InvoiceDirection.incoming, invoice_number="X",
        vendor_id=1, invoice_date=date.today(), due_date=due_date,
        subtotal=Decimal("100"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("100"),
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


def test_high_risk_invoice_is_held_regardless_of_urgency():
    very_overdue_but_risky = make_invoice(due_date=date.today() - timedelta(days=60), risk_score=RISK_HOLD_THRESHOLD)
    score, held = compute_priority(very_overdue_but_risky)
    assert held is True
    assert score < 0


def test_low_risk_invoice_is_not_held():
    invoice = make_invoice(due_date=date.today() - timedelta(days=5), risk_score=0.2)
    _, held = compute_priority(invoice)
    assert held is False


def test_risk_exactly_at_threshold_is_held():
    invoice = make_invoice(due_date=date.today(), risk_score=RISK_HOLD_THRESHOLD)
    _, held = compute_priority(invoice)
    assert held is True
