"""
Integration tests for invoice_service - the shared logic both the API
and the web UI call. These exercise the REAL orchestrator/agent
pipeline (fraud check + classification run for real), but with
ollama.chat mocked (see conftest.mock_ollama_chat) so they run fast
and deterministically instead of taking ~30s of real inference per
test. What's being tested here is the PLUMBING - does creating an
invoice actually trigger the pipeline, get logged, produce a
FraudFlag row - not the LLM's judgment quality (that's what
test_fraud_risk_agent.py and the marked @pytest.mark.llm tests are for).
"""

from datetime import date, timedelta
from decimal import Decimal

import pytest

from app.models.models import AgentLog, FraudFlag
from app.schemas.invoice import InvoiceCreate, InvoiceItemCreate, InvoiceUpdate
from app.services import invoice_service
from app.services.validation_service import InvoiceValidationError


def make_payload(invoice_number="INV-1", vendor_id=None, customer_id=None, direction="incoming", **kw):
    defaults = dict(
        direction=direction, invoice_number=invoice_number, vendor_id=vendor_id, customer_id=customer_id,
        invoice_date=date.today(), due_date=date.today() + timedelta(days=30),
        tax=Decimal("0"), discount=Decimal("0"), currency="USD",
        items=[InvoiceItemCreate(description="Widget", quantity=Decimal("2"), unit_price=Decimal("10.00"))],
    )
    defaults.update(kw)
    return InvoiceCreate(**defaults)


def test_create_invoice_computes_totals_server_side(db_session, org, vendor, mock_ollama_chat):
    payload = make_payload(vendor_id=vendor.id, tax=Decimal("5"))
    invoice = invoice_service.create_invoice(db_session, org.id, payload)
    assert invoice.subtotal == Decimal("20.00")
    assert invoice.total == Decimal("25.00")


def test_create_invoice_triggers_orchestrator_pipeline(db_session, org, vendor, mock_ollama_chat):
    payload = make_payload(vendor_id=vendor.id)
    invoice = invoice_service.create_invoice(db_session, org.id, payload)

    logs = db_session.query(AgentLog).filter(AgentLog.invoice_id == invoice.id).all()
    agent_names = {log.agent_name for log in logs}
    assert "fraud_risk_agent" in agent_names
    assert "classification_agent" in agent_names

    flag = db_session.query(FraudFlag).filter(FraudFlag.invoice_id == invoice.id).first()
    assert flag is not None


def test_create_invoice_runs_fraud_check_for_outgoing(db_session, org, customer, mock_ollama_chat):
    """Fraud/Risk Agent generalization: outgoing/customer invoices used to
    get no risk assessment at all (a real gap found while exploring
    customer invoicing). It now runs for every invoice regardless of
    direction - see app/agents/fraud_risk_agent.py's module docstring."""
    payload = make_payload(invoice_number="OUT-1", customer_id=customer.id, direction="outgoing")
    invoice = invoice_service.create_invoice(db_session, org.id, payload)

    logs = db_session.query(AgentLog).filter(AgentLog.invoice_id == invoice.id).all()
    agent_names = {log.agent_name for log in logs}
    assert "fraud_risk_agent" in agent_names
    assert invoice.risk_score is not None


def test_create_invoice_rejects_duplicate(db_session, org, vendor, mock_ollama_chat):
    invoice_service.create_invoice(db_session, org.id, make_payload(invoice_number="DUP-1", vendor_id=vendor.id))
    with pytest.raises(InvoiceValidationError, match="already exists"):
        invoice_service.create_invoice(db_session, org.id, make_payload(invoice_number="DUP-1", vendor_id=vendor.id))


def test_create_invoice_rejects_duplicate_for_customer(db_session, org, customer, mock_ollama_chat):
    """A real gap found while exploring customer invoicing: check_duplicate_invoice
    was vendor-only, so an outgoing invoice reusing an existing invoice
    number for the same customer slipped through with no check at all."""
    payload = make_payload(invoice_number="OUT-DUP-1", customer_id=customer.id, direction="outgoing")
    invoice_service.create_invoice(db_session, org.id, payload)
    with pytest.raises(InvoiceValidationError, match="already exists for this customer"):
        invoice_service.create_invoice(db_session, org.id, payload)


def test_create_invoice_allows_same_number_for_different_customers(db_session, org, customer, mock_ollama_chat):
    from app.models.models import Customer
    other_customer = Customer(organization_id=org.id, name="Other Client", email="other@test.example")
    db_session.add(other_customer)
    db_session.commit()

    invoice_service.create_invoice(
        db_session, org.id, make_payload(invoice_number="SHARED-1", customer_id=customer.id, direction="outgoing")
    )
    # No error - invoice numbering is customer-specific, same as it already is for vendors.
    invoice_service.create_invoice(
        db_session, org.id, make_payload(invoice_number="SHARED-1", customer_id=other_customer.id, direction="outgoing")
    )


def test_update_invoice_preserves_category_across_unrelated_edit(db_session, org, vendor, mock_ollama_chat):
    """Regression test for the real Phase 5 bug: editing any field used
    to silently wipe every item's category because the update path
    always rebuilt InvoiceItem rows from the resubmitted form."""
    invoice = invoice_service.create_invoice(db_session, org.id, make_payload(vendor_id=vendor.id))
    invoice.items[0].category = "Raw Ingredients"
    db_session.commit()

    # Resubmit the SAME item data (as the web form always does), only
    # changing an unrelated field.
    payload = InvoiceUpdate(
        payment_status="paid",
        items=[InvoiceItemCreate(description="Widget", quantity=Decimal("2"), unit_price=Decimal("10.00"))],
    )
    updated = invoice_service.update_invoice(db_session, invoice, payload)
    assert updated.items[0].category == "Raw Ingredients"
    assert updated.payment_status.value == "paid"


def test_update_invoice_recomputes_total_when_tax_changes(db_session, org, vendor, mock_ollama_chat):
    invoice = invoice_service.create_invoice(db_session, org.id, make_payload(vendor_id=vendor.id))
    assert invoice.total == Decimal("20.00")

    updated = invoice_service.update_invoice(db_session, invoice, InvoiceUpdate(tax=Decimal("3.00")))
    assert updated.total == Decimal("23.00")


def test_delete_invoice_removes_related_rows_no_orphans(db_session, org, vendor, mock_ollama_chat):
    """Regression test for a real bug found on the live Postgres
    deployment: deleting an invoice with related FraudFlag/AgentLog/
    Communication rows raised a foreign-key violation on Postgres,
    because those tables had an invoice_id FK but nothing cascaded
    from the Invoice side. SQLite doesn't enforce foreign keys by
    default, so this same operation silently left orphaned rows
    behind there instead of erroring - which is why this test checks
    for orphans directly (asserting only "delete_invoice() didn't
    raise" would pass even without the fix, since SQLite never
    enforced the constraint in the first place)."""
    invoice = invoice_service.create_invoice(db_session, org.id, make_payload(vendor_id=vendor.id))
    invoice_id = invoice.id

    # The mocked pipeline already created a FraudFlag + 2 AgentLog rows
    # (fraud_risk_agent + classification_agent). Add a Communication too.
    from app.agents import communication_agent
    invoice.due_date = date.today() - timedelta(days=1)  # make it "overdue" for a plausible draft
    db_session.commit()
    communication_agent.draft_reminder(db_session, invoice)

    assert db_session.query(FraudFlag).filter(FraudFlag.invoice_id == invoice_id).count() == 1
    assert db_session.query(AgentLog).filter(AgentLog.invoice_id == invoice_id).count() >= 1

    invoice_service.delete_invoice(db_session, invoice)

    assert db_session.query(FraudFlag).filter(FraudFlag.invoice_id == invoice_id).count() == 0
    assert db_session.query(AgentLog).filter(AgentLog.invoice_id == invoice_id).count() == 0
    from app.models.models import Communication
    assert db_session.query(Communication).filter(Communication.invoice_id == invoice_id).count() == 0


def test_list_invoices_search_by_invoice_number_or_party_name(db_session, org, vendor, customer, mock_ollama_chat):
    invoice_service.create_invoice(db_session, org.id, make_payload(invoice_number="GGM-0847", vendor_id=vendor.id))
    invoice_service.create_invoice(db_session, org.id, make_payload(invoice_number="SPC-1102", vendor_id=vendor.id))

    by_number = invoice_service.list_invoices(db_session, org.id, q="ggm")
    assert [i.invoice_number for i in by_number] == ["GGM-0847"]

    by_party = invoice_service.list_invoices(db_session, org.id, q="test vendor")
    assert len(by_party) == 2  # both invoices are from the same `vendor` fixture


def test_suggest_next_invoice_number_starts_at_0001_when_none_exist(db_session, org):
    assert invoice_service.suggest_next_invoice_number(db_session, org.id) == "INV-0001"


def test_suggest_next_invoice_number_increments_past_highest_existing(db_session, org, customer, mock_ollama_chat):
    invoice_service.create_invoice(db_session, org.id, make_payload(invoice_number="INV-0001", customer_id=customer.id, direction="outgoing"))
    invoice_service.create_invoice(db_session, org.id, make_payload(invoice_number="INV-0005", customer_id=customer.id, direction="outgoing"))

    assert invoice_service.suggest_next_invoice_number(db_session, org.id) == "INV-0006"


def test_suggest_next_invoice_number_ignores_non_matching_and_incoming_numbers(db_session, org, customer, vendor, mock_ollama_chat):
    """A business already using its own scheme ("MSB-2201") shouldn't
    confuse the suggester, and a vendor's own incoming invoice number
    is never ours to count toward our own outgoing sequence."""
    invoice_service.create_invoice(db_session, org.id, make_payload(invoice_number="MSB-2201", customer_id=customer.id, direction="outgoing"))
    invoice_service.create_invoice(db_session, org.id, make_payload(invoice_number="INV-9999", vendor_id=vendor.id, direction="incoming"))

    assert invoice_service.suggest_next_invoice_number(db_session, org.id) == "INV-0001"


def test_create_invoice_auto_generates_public_token_for_outgoing(db_session, org, customer, mock_ollama_chat):
    invoice = invoice_service.create_invoice(db_session, org.id, make_payload(customer_id=customer.id, direction="outgoing"))
    assert invoice.public_token is not None
    assert len(invoice.public_token) > 20


def test_create_invoice_does_not_generate_public_token_for_incoming(db_session, org, vendor, mock_ollama_chat):
    invoice = invoice_service.create_invoice(db_session, org.id, make_payload(vendor_id=vendor.id))
    assert invoice.public_token is None


def test_get_or_create_public_token_is_idempotent(db_session, org, customer, mock_ollama_chat):
    invoice = invoice_service.create_invoice(db_session, org.id, make_payload(customer_id=customer.id, direction="outgoing"))
    first = invoice_service.get_or_create_public_token(db_session, invoice)
    second = invoice_service.get_or_create_public_token(db_session, invoice)
    assert first == second


def test_get_invoice_by_public_token_finds_the_right_invoice(db_session, org, customer, mock_ollama_chat):
    invoice = invoice_service.create_invoice(db_session, org.id, make_payload(invoice_number="TOKEN-LOOKUP-1", customer_id=customer.id, direction="outgoing"))
    found = invoice_service.get_invoice_by_public_token(db_session, invoice.public_token)
    assert found is not None
    assert found.invoice_number == "TOKEN-LOOKUP-1"


def test_get_invoice_by_public_token_returns_none_for_unknown_token(db_session, org):
    assert invoice_service.get_invoice_by_public_token(db_session, "not-a-real-token") is None
