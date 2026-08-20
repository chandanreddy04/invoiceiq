"""
Turns a RecurringInvoice template into a real Invoice when it's due -
the actual "generate the next invoice" logic. Every generated invoice
goes through invoice_service.create_invoice(), the exact same path the
form and API use, so fraud/classification run on it like any other
invoice; nothing here bypasses that pipeline.

No background scheduler exists in this app (no Celery, no cron
runner) - generate_due_invoices() is meant to be triggered either by a
person clicking "Generate due invoices now" (see /web/recurring), or
by an external scheduler (e.g. Render's own Cron Jobs, or any host
that can hit a URL on a schedule) calling the same route. Both paths
call this one function, so which one triggered it never matters to
the logic itself.
"""

import calendar
import logging
from datetime import date, timedelta

from sqlalchemy.orm import Session, joinedload

from app.models.models import RecurringInvoice, RecurringInvoiceItem, RecurringFrequency, InvoiceDirection
from app.schemas.invoice import InvoiceCreate, InvoiceItemCreate
from app.schemas.recurring_invoice import RecurringInvoiceCreate
from app.services import invoice_service
from app.services.validation_service import InvoiceValidationError

logger = logging.getLogger(__name__)

_MONTHS_TO_ADD = {
    RecurringFrequency.monthly: 1,
    RecurringFrequency.quarterly: 3,
    RecurringFrequency.yearly: 12,
}


def _advance(d: date, frequency: RecurringFrequency) -> date:
    """Calendar-correct month arithmetic (Jan 31 + 1 month -> Feb 28,
    not Mar 3) - no dateutil dependency needed for just this."""
    if frequency == RecurringFrequency.weekly:
        return d + timedelta(weeks=1)

    months_to_add = _MONTHS_TO_ADD[frequency]
    total_months = d.month - 1 + months_to_add
    year = d.year + total_months // 12
    month = total_months % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def with_items(query):
    return query.options(joinedload(RecurringInvoice.items), joinedload(RecurringInvoice.customer))


def list_recurring_invoices(db: Session, org_id: int) -> list[RecurringInvoice]:
    return with_items(db.query(RecurringInvoice)).filter(RecurringInvoice.organization_id == org_id).order_by(RecurringInvoice.next_run_date.asc()).all()


def get_recurring_invoice(db: Session, recurring_id: int) -> RecurringInvoice | None:
    return with_items(db.query(RecurringInvoice)).filter(RecurringInvoice.id == recurring_id).first()


def create_recurring_invoice(db: Session, org_id: int, payload: RecurringInvoiceCreate) -> RecurringInvoice:
    template = RecurringInvoice(
        organization_id=org_id,
        customer_id=payload.customer_id,
        name=payload.name,
        frequency=payload.frequency,
        next_run_date=payload.next_run_date,
        due_days=payload.due_days,
        currency=payload.currency,
        tax=payload.tax,
        discount=payload.discount,
        payment_terms=payload.payment_terms,
    )
    template.items = [
        RecurringInvoiceItem(description=item.description, quantity=item.quantity, unit_price=item.unit_price)
        for item in payload.items
    ]
    db.add(template)
    db.commit()
    db.refresh(template)
    return template


def set_active(db: Session, template: RecurringInvoice, is_active: bool) -> None:
    template.is_active = is_active
    db.commit()


def delete_recurring_invoice(db: Session, template: RecurringInvoice) -> None:
    db.delete(template)
    db.commit()


def generate_due_invoices(db: Session, org_id: int) -> list:
    """Creates a real Invoice for every active template whose
    next_run_date has arrived, then advances that template's
    next_run_date past today. A template that fails validation (e.g. a
    duplicate invoice_number race) is skipped, not fatal to the rest -
    same "one failure doesn't stop the others" discipline the
    Orchestrator uses."""
    due = (
        with_items(db.query(RecurringInvoice))
        .filter(
            RecurringInvoice.organization_id == org_id,
            RecurringInvoice.is_active.is_(True),
            RecurringInvoice.next_run_date <= date.today(),
        )
        .all()
    )

    created = []
    for template in due:
        invoice_number = invoice_service.suggest_next_invoice_number(db, org_id)
        payload = InvoiceCreate(
            direction=InvoiceDirection.outgoing,
            invoice_number=invoice_number,
            customer_id=template.customer_id,
            invoice_date=date.today(),
            due_date=_due_date(template.due_days),
            tax=template.tax,
            discount=template.discount,
            currency=template.currency,
            payment_terms=template.payment_terms,
            items=[InvoiceItemCreate(description=i.description, quantity=i.quantity, unit_price=i.unit_price) for i in template.items],
        )
        try:
            invoice = invoice_service.create_invoice(db, org_id, payload)
            created.append(invoice)
        except InvoiceValidationError:
            logger.exception("Skipping recurring invoice template %s this cycle - will retry next time", template.id)
            continue

        template.next_run_date = _advance(template.next_run_date, template.frequency)
        db.commit()

    return created


def _due_date(due_days: int) -> date:
    return date.today() + timedelta(days=due_days)
