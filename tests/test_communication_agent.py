"""
Tests for the Communication Agent's LLM-unavailable fallback.
Regression test for a real gap found on the live Postgres/no-LLM
deployment: every other agent degrades to a fallback when the LLM is
unreachable, but this one used to just raise and create nothing.
"""

from datetime import date, timedelta
from decimal import Decimal

from app.agents import communication_agent
from app.models.models import Invoice, InvoiceDirection, InvoiceStatus, PaymentStatus, Communication


def test_draft_reminder_falls_back_to_template_when_llm_unavailable(db_session, org, customer, monkeypatch):
    def _broken_chat(*args, **kwargs):
        raise ConnectionError("simulated: no LLM reachable")

    import ollama
    monkeypatch.setattr(ollama, "chat", _broken_chat)

    invoice = Invoice(
        organization_id=org.id, direction=InvoiceDirection.outgoing, invoice_number="FALLBACK-1",
        customer_id=customer.id, invoice_date=date.today() - timedelta(days=20),
        due_date=date.today() - timedelta(days=5),
        subtotal=Decimal("100"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("100"),
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
    )
    db_session.add(invoice)
    db_session.commit()
    db_session.refresh(invoice)

    result = communication_agent.draft_reminder(db_session, invoice)

    assert result is not None  # the real bug: this used to be impossible to get here
    assert isinstance(result, Communication)
    assert result.status == "draft"
    assert "LLM unavailable" in result.subject
    assert invoice.invoice_number in result.body
    assert str(invoice.total) in result.body or "100" in result.body

    saved = db_session.query(Communication).filter(Communication.invoice_id == invoice.id).first()
    assert saved is not None
