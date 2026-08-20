"""
Shared invoice CRUD logic. Both the JSON API (app/api/invoices.py) and
the HTML pages (app/web/routes.py) call these same functions, so
there is exactly one implementation of "how an invoice gets created
or updated" - no duplicated logic to drift out of sync between the
two interfaces.
"""

import re
import secrets
from decimal import Decimal, ROUND_HALF_UP

from sqlalchemy import or_
from sqlalchemy.orm import Session, joinedload

from app.models.models import Invoice, InvoiceItem, InvoiceDirection, InvoiceStatus, PaymentStatus, Vendor, Customer
from app.schemas.invoice import InvoiceCreate, InvoiceUpdate
from app.services.validation_service import (
    calculate_invoice_totals,
    validate_invoice_input,
    InvoiceValidationError,
)
from app.tools.invoice_tools import check_duplicate_invoice


def with_items(query):
    return query.options(joinedload(Invoice.items))


_AUTO_NUMBER_PATTERN = re.compile(r"^INV-(\d+)$")


def suggest_next_invoice_number(db: Session, org_id: int) -> str:
    """Real gap found from a "could a local business actually run on
    this" review: invoice_number was a required field with no
    suggestion at all - every outgoing invoice needed a number typed
    from scratch, unlike every real invoicing tool. Only meaningful for
    outgoing invoices - an incoming one already has a real number the
    vendor assigned it, which we have no business overriding or
    guessing at.

    Scans existing outgoing invoice numbers matching "INV-<digits>" for
    this org and returns the next one, INV-0001 if none exist yet.
    Deliberately just a suggestion, not enforced or auto-committed - a
    business already using its own numbering scheme (the demo data
    itself has "MSB-2201" style numbers) can freely type over it, same
    as any pre-filled form field."""
    numbers = (
        db.query(Invoice.invoice_number)
        .filter(Invoice.organization_id == org_id, Invoice.direction == InvoiceDirection.outgoing)
        .all()
    )
    highest = 0
    for (number,) in numbers:
        match = _AUTO_NUMBER_PATTERN.match(number)
        if match:
            highest = max(highest, int(match.group(1)))
    return f"INV-{highest + 1:04d}"


def list_invoices(
    db: Session,
    org_id: int,
    direction: InvoiceDirection | None = None,
    payment_status: PaymentStatus | None = None,
    q: str | None = None,
):
    """`q` is a single free-text search box on the Invoices page - matches
    either the invoice number or the vendor/customer name, whichever
    applies (an invoice has one or the other, never both)."""
    query = with_items(db.query(Invoice)).filter(Invoice.organization_id == org_id)
    if direction is not None:
        query = query.filter(Invoice.direction == direction)
    if payment_status is not None:
        query = query.filter(Invoice.payment_status == payment_status)
    if q:
        term = f"%{q}%"
        query = (
            query.outerjoin(Vendor, Invoice.vendor_id == Vendor.id)
            .outerjoin(Customer, Invoice.customer_id == Customer.id)
            .filter(or_(Invoice.invoice_number.ilike(term), Vendor.name.ilike(term), Customer.name.ilike(term)))
        )
    return query.order_by(Invoice.due_date.asc()).all()


def get_invoice(db: Session, invoice_id: int) -> Invoice | None:
    return with_items(db.query(Invoice)).filter(Invoice.id == invoice_id).first()


def get_invoice_by_public_token(db: Session, token: str) -> Invoice | None:
    return with_items(db.query(Invoice)).filter(Invoice.public_token == token).first()


def get_or_create_public_token(db: Session, invoice: Invoice) -> str | None:
    """Only outgoing invoices ever get one - there is no "customer
    view" of an invoice we owe a vendor. secrets.token_urlsafe(24) is
    ~192 bits of entropy - not sequentially guessable like the primary
    key, which matters here since this token IS the access control for
    /pay/<token> (no login required by design, so it has to be
    unguessable rather than merely private)."""
    if invoice.direction != InvoiceDirection.outgoing:
        return None
    if not invoice.public_token:
        invoice.public_token = secrets.token_urlsafe(24)
        db.commit()
        db.refresh(invoice)
    return invoice.public_token


def create_invoice(db: Session, org_id: int, payload: InvoiceCreate) -> Invoice:
    """Raises InvoiceValidationError (from validation_service) on bad input."""
    validate_invoice_input(payload.items, payload.due_date, payload.invoice_date)

    dup = check_duplicate_invoice(db, org_id, payload.vendor_id, payload.invoice_number, customer_id=payload.customer_id)
    if dup is not None:
        party_label = "vendor" if payload.vendor_id is not None else "customer"
        raise InvoiceValidationError(
            f"Invoice number '{payload.invoice_number}' already exists for this {party_label} (invoice #{dup.id})."
        )

    subtotal, total, line_totals = calculate_invoice_totals(payload.items, payload.tax, payload.discount)

    invoice = Invoice(
        organization_id=org_id,
        direction=payload.direction,
        invoice_number=payload.invoice_number,
        vendor_id=payload.vendor_id,
        customer_id=payload.customer_id,
        invoice_date=payload.invoice_date,
        due_date=payload.due_date,
        subtotal=subtotal,
        tax=payload.tax,
        discount=payload.discount,
        total=total,
        currency=payload.currency,
        payment_terms=payload.payment_terms,
        payment_status=PaymentStatus.unpaid,
        invoice_status=InvoiceStatus.validated,
        source_pdf_filename=payload.source_pdf_filename,
    )
    invoice.items = [
        InvoiceItem(
            description=item.description,
            quantity=item.quantity,
            unit_price=item.unit_price,
            line_total=line_totals[i],
        )
        for i, item in enumerate(payload.items)
    ]
    db.add(invoice)
    db.commit()
    db.refresh(invoice)

    get_or_create_public_token(db, invoice)

    # Phase 5: the Orchestrator now owns dispatching to whichever agents
    # a newly created invoice needs (fraud check, classification, ...),
    # replacing the direct Fraud/Risk Agent call from Phase 4. Every
    # invoice-creation path (API, web form, upload confirm) shares this
    # one function, so all of them get the same agent pipeline.
    from app.agents.orchestrator import run_invoice_pipeline
    run_invoice_pipeline(db, invoice)

    return invoice


def update_invoice(db: Session, invoice: Invoice, payload: InvoiceUpdate) -> Invoice:
    """Raises InvoiceValidationError (from validation_service) on bad input."""
    data = payload.model_dump(exclude_unset=True)
    items_payload = data.pop("items", None)

    for field, value in data.items():
        setattr(invoice, field, value)

    if items_payload is not None:
        validate_invoice_input(payload.items, invoice.due_date, invoice.invoice_date)
        subtotal, total, line_totals = calculate_invoice_totals(payload.items, invoice.tax, invoice.discount)

        # Rebuilding items wholesale would otherwise silently wipe the
        # category the Classification Agent already set, on every save -
        # even edits that didn't touch line items at all (the form always
        # resubmits all 8 rows). Carry the category over by description
        # match instead of re-classifying on every edit.
        existing_categories = {item.description.strip().lower(): item.category for item in invoice.items}

        invoice.items = [
            InvoiceItem(
                description=item.description,
                quantity=item.quantity,
                unit_price=item.unit_price,
                line_total=line_totals[i],
                category=existing_categories.get(item.description.strip().lower()),
            )
            for i, item in enumerate(payload.items)
        ]
        invoice.subtotal = subtotal
        invoice.total = total
    elif "tax" in data or "discount" in data:
        invoice.total = (invoice.subtotal + invoice.tax - invoice.discount).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

    db.commit()
    db.refresh(invoice)
    return invoice


def delete_invoice(db: Session, invoice: Invoice) -> None:
    db.delete(invoice)
    db.commit()
