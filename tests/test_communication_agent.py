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


def test_draft_invoice_email_falls_back_to_template_when_llm_unavailable(db_session, org, customer, monkeypatch):
    def _broken_chat(*args, **kwargs):
        raise ConnectionError("simulated: no LLM reachable")

    import ollama
    monkeypatch.setattr(ollama, "chat", _broken_chat)

    invoice = Invoice(
        organization_id=org.id, direction=InvoiceDirection.outgoing, invoice_number="EMAIL-FALLBACK-1",
        customer_id=customer.id, invoice_date=date.today(), due_date=date.today() + timedelta(days=30),
        subtotal=Decimal("200"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("200"),
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
    )
    db_session.add(invoice)
    db_session.commit()
    db_session.refresh(invoice)

    result = communication_agent.draft_invoice_email(db_session, invoice)

    assert result is not None
    assert isinstance(result, Communication)
    assert result.status == "draft"
    assert result.recipient == customer.email
    assert invoice.invoice_number in result.body
    assert "LLM unavailable" in result.subject


def test_draft_invoice_email_returns_none_for_incoming_invoice(db_session, org, vendor, mock_ollama_chat):
    """This is specifically the "here is your new invoice" message - an
    incoming/vendor invoice is one we received, not one we're sending,
    so there's nothing for this function to do with it."""
    invoice = Invoice(
        organization_id=org.id, direction=InvoiceDirection.incoming, invoice_number="INCOMING-1",
        vendor_id=vendor.id, invoice_date=date.today(), due_date=date.today() + timedelta(days=30),
        subtotal=Decimal("200"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("200"),
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
    )
    db_session.add(invoice)
    db_session.commit()
    db_session.refresh(invoice)

    assert communication_agent.draft_invoice_email(db_session, invoice) is None


def test_draft_invoice_email_returns_none_when_customer_has_no_email(db_session, org, mock_ollama_chat):
    from app.models.models import Customer
    no_email_customer = Customer(organization_id=org.id, name="No Email Co.", email=None)
    db_session.add(no_email_customer)
    db_session.commit()

    invoice = Invoice(
        organization_id=org.id, direction=InvoiceDirection.outgoing, invoice_number="NO-EMAIL-1",
        customer_id=no_email_customer.id, invoice_date=date.today(), due_date=date.today() + timedelta(days=30),
        subtotal=Decimal("200"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("200"),
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
    )
    db_session.add(invoice)
    db_session.commit()
    db_session.refresh(invoice)

    assert communication_agent.draft_invoice_email(db_session, invoice) is None


def test_draft_invoice_email_creates_approval_request(db_session, org, customer, mock_ollama_chat):
    from app.models.models import ApprovalRequest

    invoice = Invoice(
        organization_id=org.id, direction=InvoiceDirection.outgoing, invoice_number="APPROVAL-1",
        customer_id=customer.id, invoice_date=date.today(), due_date=date.today() + timedelta(days=30),
        subtotal=Decimal("200"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("200"),
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
    )
    db_session.add(invoice)
    db_session.commit()
    db_session.refresh(invoice)

    comm = communication_agent.draft_invoice_email(db_session, invoice)

    req = db_session.query(ApprovalRequest).filter(ApprovalRequest.related_id == comm.id, ApprovalRequest.type == "send_communication").first()
    assert req is not None
    assert req.status == "pending"
    assert "download the pdf" in req.reason.lower()
