"""
Tests for RecurringInvoice template CRUD and generate_due_invoices() -
the actual "turn a due template into a real invoice" logic. Every
generated invoice goes through the real invoice_service.create_invoice()
pipeline (mocked LLM), so fraud/classification genuinely run on it too.
"""

from datetime import date, timedelta
from decimal import Decimal

from app.models.models import RecurringFrequency, Invoice
from app.schemas.invoice import InvoiceItemCreate
from app.schemas.recurring_invoice import RecurringInvoiceCreate
from app.services import recurring_invoice_service


def make_payload(customer_id, **overrides):
    defaults = dict(
        customer_id=customer_id, name="Monthly retainer", frequency=RecurringFrequency.monthly,
        next_run_date=date.today(), due_days=30, currency="USD", tax=Decimal("0"), discount=Decimal("0"),
        payment_terms="Net 30", items=[InvoiceItemCreate(description="Retainer", quantity=Decimal("1"), unit_price=Decimal("500"))],
    )
    defaults.update(overrides)
    return RecurringInvoiceCreate(**defaults)


def test_create_recurring_invoice_saves_template_and_items(db_session, org, customer):
    template = recurring_invoice_service.create_recurring_invoice(db_session, org.id, make_payload(customer.id))
    assert template.id is not None
    assert template.is_active is True
    assert len(template.items) == 1
    assert template.items[0].description == "Retainer"


def test_generate_due_invoices_creates_real_invoice_for_due_template(db_session, org, customer, mock_ollama_chat):
    recurring_invoice_service.create_recurring_invoice(db_session, org.id, make_payload(customer.id, next_run_date=date.today()))

    created = recurring_invoice_service.generate_due_invoices(db_session, org.id)

    assert len(created) == 1
    assert created[0].customer_id == customer.id
    assert created[0].total == Decimal("500.00")
    saved = db_session.query(Invoice).filter(Invoice.id == created[0].id).first()
    assert saved is not None


def test_generate_due_invoices_skips_not_yet_due_template(db_session, org, customer, mock_ollama_chat):
    recurring_invoice_service.create_recurring_invoice(
        db_session, org.id, make_payload(customer.id, next_run_date=date.today() + timedelta(days=10))
    )
    created = recurring_invoice_service.generate_due_invoices(db_session, org.id)
    assert created == []


def test_generate_due_invoices_skips_paused_template(db_session, org, customer, mock_ollama_chat):
    template = recurring_invoice_service.create_recurring_invoice(db_session, org.id, make_payload(customer.id, next_run_date=date.today()))
    recurring_invoice_service.set_active(db_session, template, False)

    created = recurring_invoice_service.generate_due_invoices(db_session, org.id)
    assert created == []


def test_generate_due_invoices_advances_next_run_date_monthly(db_session, org, customer, mock_ollama_chat):
    today = date.today()
    template = recurring_invoice_service.create_recurring_invoice(db_session, org.id, make_payload(customer.id, next_run_date=today))

    recurring_invoice_service.generate_due_invoices(db_session, org.id)

    db_session.refresh(template)
    assert template.next_run_date > today
    assert template.next_run_date.month != today.month or template.next_run_date.year != today.year


def test_advance_handles_month_end_overflow_correctly():
    """Jan 31 + 1 month must land on Feb 28 (or 29), never "Mar 3" -
    calendar.monthrange-based clamping, no naive day-count math."""
    result = recurring_invoice_service._advance(date(2026, 1, 31), RecurringFrequency.monthly)
    assert result == date(2026, 2, 28)


def test_advance_weekly_adds_seven_days():
    result = recurring_invoice_service._advance(date(2026, 1, 1), RecurringFrequency.weekly)
    assert result == date(2026, 1, 8)


def test_advance_yearly_adds_twelve_months():
    result = recurring_invoice_service._advance(date(2026, 3, 15), RecurringFrequency.yearly)
    assert result == date(2027, 3, 15)


def test_due_date_computed_from_due_days(db_session, org, customer, mock_ollama_chat):
    recurring_invoice_service.create_recurring_invoice(
        db_session, org.id, make_payload(customer.id, next_run_date=date.today(), due_days=14)
    )
    created = recurring_invoice_service.generate_due_invoices(db_session, org.id)
    assert created[0].due_date == date.today() + timedelta(days=14)


def test_set_active_toggles_and_persists(db_session, org, customer):
    template = recurring_invoice_service.create_recurring_invoice(db_session, org.id, make_payload(customer.id))
    recurring_invoice_service.set_active(db_session, template, False)
    assert template.is_active is False


def test_delete_recurring_invoice_removes_it_and_its_items(db_session, org, customer):
    from app.models.models import RecurringInvoice, RecurringInvoiceItem
    template = recurring_invoice_service.create_recurring_invoice(db_session, org.id, make_payload(customer.id))
    template_id = template.id

    recurring_invoice_service.delete_recurring_invoice(db_session, template)

    assert db_session.query(RecurringInvoice).filter(RecurringInvoice.id == template_id).count() == 0
    assert db_session.query(RecurringInvoiceItem).filter(RecurringInvoiceItem.recurring_invoice_id == template_id).count() == 0


def test_list_recurring_invoices_scoped_to_org(db_session, org, customer):
    recurring_invoice_service.create_recurring_invoice(db_session, org.id, make_payload(customer.id))
    from app.models.models import Organization
    other_org = Organization(name="Other Org")
    db_session.add(other_org)
    db_session.commit()

    assert len(recurring_invoice_service.list_recurring_invoices(db_session, org.id)) == 1
    assert len(recurring_invoice_service.list_recurring_invoices(db_session, other_org.id)) == 0
