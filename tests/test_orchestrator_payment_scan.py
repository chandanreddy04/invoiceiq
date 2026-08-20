"""
Tests that run_payment_scan() actually logs to AgentLog - requested
directly, mirroring the same change already made for Collections/AR's
run_collections_scan() (see tests/test_orchestrator_collections.py).
"""

from datetime import date, timedelta
from decimal import Decimal

from app.agents import orchestrator
from app.models.models import Invoice, InvoiceDirection, InvoiceStatus, PaymentStatus, AgentLog


def test_run_payment_scan_writes_agent_log(db_session, org, vendor, mock_ollama_chat):
    db_session.add(Invoice(
        organization_id=org.id, direction=InvoiceDirection.incoming, invoice_number="PAY-1",
        vendor_id=vendor.id, invoice_date=date.today(), due_date=date.today() - timedelta(days=5),
        subtotal=Decimal("100"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("100"),
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
    ))
    db_session.commit()

    result = orchestrator.run_payment_scan(db_session, org.id)
    assert len(result["recommended"]) == 1

    logs = db_session.query(AgentLog).filter(AgentLog.agent_name == "payment_ap_agent").all()
    assert len(logs) == 1
    assert logs[0].status == "success"
    assert logs[0].invoice_id is None  # a scan, not tied to one specific invoice
    assert "1 recommended" in logs[0].output_summary


def test_run_payment_scan_summary_has_no_raw_object_reprs(db_session, org, vendor, mock_ollama_chat):
    """Same regression guard as Collections/AR's equivalent test: without
    summarize=, output_summary would be str(result) - a dict full of
    Invoice ORM objects, rendering as unreadable reprs on Agent Activity."""
    db_session.add(Invoice(
        organization_id=org.id, direction=InvoiceDirection.incoming, invoice_number="PAY-2",
        vendor_id=vendor.id, invoice_date=date.today(), due_date=date.today() - timedelta(days=5),
        subtotal=Decimal("100"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("100"),
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.validated,
    ))
    db_session.commit()

    orchestrator.run_payment_scan(db_session, org.id)
    log = db_session.query(AgentLog).filter(AgentLog.agent_name == "payment_ap_agent").first()
    assert "object at 0x" not in log.output_summary


def test_run_payment_scan_reports_held_for_review_count(db_session, org, vendor, mock_ollama_chat):
    db_session.add(Invoice(
        organization_id=org.id, direction=InvoiceDirection.incoming, invoice_number="PAY-RISKY",
        vendor_id=vendor.id, invoice_date=date.today(), due_date=date.today() - timedelta(days=5),
        subtotal=Decimal("100"), tax=Decimal("0"), discount=Decimal("0"), total=Decimal("100"),
        payment_status=PaymentStatus.unpaid, invoice_status=InvoiceStatus.pending_review,
        risk_score=Decimal("0.9"),
    ))
    db_session.commit()

    result = orchestrator.run_payment_scan(db_session, org.id)
    assert len(result["held_for_review"]) == 1

    log = db_session.query(AgentLog).filter(AgentLog.agent_name == "payment_ap_agent").first()
    assert "1 held for review" in log.output_summary
