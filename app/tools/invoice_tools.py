"""
The fixed, named tool functions from Section 18. These are plain
Python/SQLAlchemy - no LLM anywhere in this file. This is what an
agent is allowed to call; it can never send raw SQL, only invoke one
of these by name with parameters (Section 17: "do NOT allow
unrestricted raw SQL from the LLM").
"""

from datetime import date
from decimal import Decimal

from sqlalchemy.orm import Session, joinedload

from app.models.models import Invoice, InvoiceDirection, PaymentStatus


def search_invoices(
    db: Session,
    org_id: int,
    direction: str | None = None,
    payment_status: str | None = None,
    min_total: float | None = None,
    max_total: float | None = None,
    overdue_only: bool = False,
) -> list[Invoice]:
    query = db.query(Invoice).options(joinedload(Invoice.items)).filter(Invoice.organization_id == org_id)

    if direction:
        query = query.filter(Invoice.direction == InvoiceDirection(direction))
    if payment_status:
        query = query.filter(Invoice.payment_status == PaymentStatus(payment_status))
    if min_total is not None:
        query = query.filter(Invoice.total >= Decimal(str(min_total)))
    if max_total is not None:
        query = query.filter(Invoice.total <= Decimal(str(max_total)))
    if overdue_only:
        query = query.filter(Invoice.due_date < date.today(), Invoice.payment_status != PaymentStatus.paid)

    return query.order_by(Invoice.due_date.asc()).all()


def get_overdue_invoices(db: Session, org_id: int) -> list[Invoice]:
    return search_invoices(db, org_id, overdue_only=True)


def check_duplicate_invoice(
    db: Session, org_id: int, vendor_id: int | None, invoice_number: str, exclude_invoice_id: int | None = None
) -> Invoice | None:
    """Section 15/18: the duplicate check that should have existed since
    Phase 1 validation but was only added in Phase 9 while building the
    synthetic dataset - a real gap in earlier coverage, not a deliberate
    later addition. Same vendor + same invoice number is treated as a
    duplicate; two different vendors are allowed to reuse a number since
    invoice numbering is vendor-specific in the real world."""
    if vendor_id is None:
        return None
    query = db.query(Invoice).filter(
        Invoice.organization_id == org_id,
        Invoice.vendor_id == vendor_id,
        Invoice.invoice_number == invoice_number,
    )
    if exclude_invoice_id is not None:
        query = query.filter(Invoice.id != exclude_invoice_id)
    return query.first()


def generate_financial_summary(db: Session, org_id: int) -> dict:
    all_invoices = db.query(Invoice).filter(Invoice.organization_id == org_id).all()
    incoming = [i for i in all_invoices if i.direction == InvoiceDirection.incoming]
    outgoing = [i for i in all_invoices if i.direction == InvoiceDirection.outgoing]

    def unpaid_total(invoices):
        return sum((i.total for i in invoices if i.payment_status != PaymentStatus.paid), Decimal("0"))

    overdue = [i for i in all_invoices if i.due_date < date.today() and i.payment_status != PaymentStatus.paid]

    return {
        "total_payable_outstanding": unpaid_total(incoming),
        "total_receivable_outstanding": unpaid_total(outgoing),
        "count_overdue": len(overdue),
        "overdue_total": sum((i.total for i in overdue), Decimal("0")),
        "count_invoices": len(all_invoices),
    }


def get_dashboard_stats(db: Session, org_id: int) -> dict:
    """Everything the Dashboard page (Section 22) needs, in one call.
    Reuses generate_financial_summary rather than recomputing outstanding/
    overdue totals a second way - one definition of "outstanding," not two
    that could quietly drift apart."""
    from app.models.models import ApprovalRequest

    summary = generate_financial_summary(db, org_id)
    all_invoices = db.query(Invoice).filter(Invoice.organization_id == org_id).all()

    paid = [i for i in all_invoices if i.payment_status == PaymentStatus.paid]
    suspicious = [i for i in all_invoices if i.risk_score is not None and float(i.risk_score) >= 0.5]
    # ApprovalRequest has no org_id column - fine for a single-organization
    # demo, would need scoping if this became multi-tenant.
    pending_approvals = db.query(ApprovalRequest).filter(ApprovalRequest.status == "pending").count()

    return {
        **summary,
        "count_paid": len(paid),
        "total_paid_amount": sum((i.total for i in paid), Decimal("0")),
        "suspicious_invoices": sorted(suspicious, key=lambda i: float(i.risk_score), reverse=True),
        "pending_approvals": pending_approvals,
    }
