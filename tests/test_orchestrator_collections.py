"""
Tests that run_collections_scan() actually logs to AgentLog - the
specific behavior requested (Collections/AR should show up on the
Agent Activity page, unlike Payment/AP, which isn't logged at all).
"""

from datetime import date, timedelta
from decimal import Decimal

from app.agents import orchestrator
from app.models.models import Invoice, InvoiceDirection, InvoiceStatus, PaymentStatus, AgentLog


def test_run_collections_scan_writes_agent_log(db_session, org, customer, mock_ollama_chat):
    db_session.add(Invoice(
        organization_id=org.id, direction=InvoiceDirection.outgoing, invoice_number="COLL-1",
        customer_id=customer.id, invoice_date=date.today(), due_date=date.today() - timedelta(days=5),
        subtotal=Decimal("100"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("100"),
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
    ))
    db_session.commit()

    result = orchestrator.run_collections_scan(db_session, org.id)
    assert len(result["recommended"]) == 1

    logs = db_session.query(AgentLog).filter(AgentLog.agent_name == "collections_agent").all()
    assert len(logs) == 1
    assert logs[0].status == "success"
    assert logs[0].invoice_id is None  # a scan, not tied to one specific invoice
    assert "1 recommended" in logs[0].output_summary


def test_run_collections_scan_summary_has_no_raw_object_reprs(db_session, org, customer, mock_ollama_chat):
    """Regression guard for the reason summarize= was added to _log_step:
    without it, output_summary would be str(result) - a dict full of
    Invoice ORM objects, rendering as unreadable "<...Invoice object at
    0x...>" text on the Agent Activity page."""
    db_session.add(Invoice(
        organization_id=org.id, direction=InvoiceDirection.outgoing, invoice_number="COLL-2",
        customer_id=customer.id, invoice_date=date.today(), due_date=date.today() - timedelta(days=5),
        subtotal=Decimal("100"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("100"),
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
    ))
    db_session.commit()

    orchestrator.run_collections_scan(db_session, org.id)
    log = db_session.query(AgentLog).filter(AgentLog.agent_name == "collections_agent").first()
    assert "object at 0x" not in log.output_summary
