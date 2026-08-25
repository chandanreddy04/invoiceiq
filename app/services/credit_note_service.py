"""
Credit notes against outgoing invoices - a refund or billing
correction, recorded as its own document rather than editing the
original invoice (which stays the historical record of what was
actually billed, same as real accounting practice).
"""

import re
from decimal import Decimal

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.models import CreditNote, Invoice, Payment

_CN_NUMBER_PATTERN = re.compile(r"^CN-(\d+)$")


class CreditNoteError(Exception):
    """Raised when a credit note would exceed what's left to credit on
    the invoice - never silently clamped, the caller decides how to
    surface this to whoever's issuing it."""


def suggest_next_credit_note_number(db: Session, org_id: int) -> str:
    """Same scheme as invoice_service.suggest_next_invoice_number() -
    scans existing "CN-<digits>" numbers for this org, returns the
    next one."""
    numbers = db.query(CreditNote.credit_note_number).filter(CreditNote.organization_id == org_id).all()
    highest = 0
    for (number,) in numbers:
        match = _CN_NUMBER_PATTERN.match(number)
        if match:
            highest = max(highest, int(match.group(1)))
    return f"CN-{highest + 1:04d}"


def get_credit_notes_for_invoice(db: Session, invoice_id: int) -> list[CreditNote]:
    return db.query(CreditNote).filter(CreditNote.invoice_id == invoice_id).order_by(CreditNote.created_at.desc()).all()


def total_credited(db: Session, invoice_id: int) -> Decimal:
    notes = get_credit_notes_for_invoice(db, invoice_id)
    return sum((n.amount for n in notes), Decimal("0"))


def total_paid(db: Session, invoice_id: int) -> Decimal:
    return Decimal(db.query(func.coalesce(func.sum(Payment.amount), 0)).filter(Payment.invoice_id == invoice_id).scalar())


def remaining_creditable(db: Session, invoice: Invoice) -> Decimal:
    """What's actually still owed on this invoice - what was billed,
    minus what's already been credited AND what's already been paid.
    Without netting out payments, a partially-paid invoice showed "up to
    the full total remaining" here, which both misrepresented the
    balance on screen and would have let someone issue a credit note
    larger than what the customer still owes (the same number also sets
    the amount field's max and create_credit_note()'s own validation
    ceiling - one definition, used everywhere, not two that could drift)."""
    return invoice.total - total_credited(db, invoice.id) - total_paid(db, invoice.id)


def create_credit_note(db: Session, org_id: int, invoice: Invoice, reason: str, amount: Decimal, created_by: str) -> CreditNote:
    if amount <= 0:
        raise CreditNoteError("Credit note amount must be greater than zero.")
    remaining = remaining_creditable(db, invoice)
    if amount > remaining:
        raise CreditNoteError(f"Amount {amount} exceeds the {remaining} still creditable on this invoice.")

    note = CreditNote(
        organization_id=org_id,
        invoice_id=invoice.id,
        credit_note_number=suggest_next_credit_note_number(db, org_id),
        reason=reason,
        amount=amount,
        currency=invoice.currency,
        created_by=created_by,
    )
    db.add(note)
    db.commit()
    db.refresh(note)
    return note
