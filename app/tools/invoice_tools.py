"""
The fixed, named tool functions from Section 18. These are plain
Python/SQLAlchemy - no LLM anywhere in this file. This is what an
agent is allowed to call; it can never send raw SQL, only invoke one
of these by name with parameters (Section 17: "do NOT allow
unrestricted raw SQL from the LLM").
"""

from datetime import date
from decimal import Decimal

from sqlalchemy import or_
from sqlalchemy.orm import Session, joinedload

from app.models.models import Invoice, InvoiceDirection, PaymentStatus, Vendor, Customer, InvoiceItem

RISKY_THRESHOLD = Decimal("0.5")  # same bar the Dashboard's "Suspicious" tile uses


def search_invoices(
    db: Session,
    org_id: int,
    direction: str | None = None,
    payment_status: str | None = None,
    min_total: float | None = None,
    max_total: float | None = None,
    overdue_only: bool = False,
    party_name: str | None = None,
    invoice_number: str | None = None,
    category: str | None = None,
    risky_only: bool = False,
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
    if invoice_number:
        query = query.filter(Invoice.invoice_number.ilike(f"%{invoice_number}%"))
    if risky_only:
        query = query.filter(Invoice.risk_score.isnot(None), Invoice.risk_score >= RISKY_THRESHOLD)

    if party_name:
        # An invoice has either a vendor OR a customer, never both - two
        # outer joins plus an OR covers whichever one actually applies,
        # instead of requiring the caller to already know the direction.
        term = f"%{party_name}%"
        query = (
            query.outerjoin(Vendor, Invoice.vendor_id == Vendor.id)
            .outerjoin(Customer, Invoice.customer_id == Customer.id)
            .filter(or_(Vendor.name.ilike(term), Customer.name.ilike(term)))
        )

    if category:
        # Matching on a child table's column requires a join; .distinct()
        # avoids the same invoice appearing twice if more than one line
        # item matches the category.
        query = query.join(InvoiceItem, Invoice.id == InvoiceItem.invoice_id).filter(
            InvoiceItem.category.ilike(f"%{category}%")
        ).distinct()

    return query.order_by(Invoice.due_date.asc()).all()


def get_overdue_invoices(db: Session, org_id: int) -> list[Invoice]:
    return search_invoices(db, org_id, overdue_only=True)


def aggregate_invoices(
    db: Session, org_id: int, group_by: str, metric: str = "total", direction: str | None = None,
    ascending: bool = False,
) -> dict[str, list[dict]]:
    """The tool function query_intent.py's docstring used to describe as a
    real gap: search_invoices() only filters, it can't answer "which
    vendor do I spend the most with" or "average invoice amount by
    category" - those need grouping and a computed metric per group,
    which is what this does instead.

    group_by: "vendor" | "customer" | "category". metric: "total" |
    "average" | "count". `ascending` controls ranking direction - False
    (default) puts the highest first ("most"/"leads"), True puts the
    lowest first ("least"). A real gap found live: without this, "which
    vendor do I spend the LEAST with" returned the exact same answer as
    "the MOST with," since there was previously no way to ask for the
    opposite ranking at all. Returns {currency: [{"label", "value",
    "count"}, ...]} sorted *within* each currency - never summed across
    currencies, for the same reason totals_by_currency() exists (a $100
    USD vendor and a EUR 100 vendor aren't comparable by just adding
    their numbers together)."""
    query = db.query(Invoice).options(joinedload(Invoice.items)).filter(Invoice.organization_id == org_id)
    if direction:
        query = query.filter(Invoice.direction == InvoiceDirection(direction))
    invoices = query.options(joinedload(Invoice.vendor), joinedload(Invoice.customer)).all()

    groups: dict[tuple[str, str], list[Invoice]] = {}
    for inv in invoices:
        if group_by == "category":
            # A single invoice can have line items in more than one
            # category - it counts under each category its items touch,
            # rather than being forced into just one.
            for cat in {item.category for item in inv.items if item.category}:
                groups.setdefault((cat, inv.currency), []).append(inv)
            continue
        party = inv.vendor if group_by == "vendor" else (inv.customer if group_by == "customer" else None)
        if party is None:
            continue
        groups.setdefault((party.name, inv.currency), []).append(inv)

    by_currency: dict[str, list[dict]] = {}
    for (label, currency), invs in groups.items():
        total = sum((i.total for i in invs), Decimal("0"))
        if metric == "average":
            value = total / len(invs)
        elif metric == "count":
            value = Decimal(len(invs))
        else:
            value = total
        by_currency.setdefault(currency, []).append({"label": label, "value": value, "count": len(invs)})

    for currency, rows in by_currency.items():
        rows.sort(key=lambda r: r["value"], reverse=not ascending)

    return by_currency


def check_duplicate_invoice(
    db: Session, org_id: int, vendor_id: int | None, invoice_number: str,
    customer_id: int | None = None, exclude_invoice_id: int | None = None,
) -> Invoice | None:
    """Section 15/18: the duplicate check that should have existed since
    Phase 1 validation but was only added in Phase 9 while building the
    synthetic dataset - a real gap in earlier coverage, not a deliberate
    later addition. Same vendor + same invoice number is treated as a
    duplicate; two different vendors are allowed to reuse a number since
    invoice numbering is vendor-specific in the real world.

    Originally vendor-only - a real gap found while exploring customer
    invoicing: an outgoing invoice reusing an existing invoice number for
    the same customer slipped through with no check at all. customer_id
    added as an equivalent, same-shaped check for that side."""
    if vendor_id is not None:
        query = db.query(Invoice).filter(
            Invoice.organization_id == org_id,
            Invoice.vendor_id == vendor_id,
            Invoice.invoice_number == invoice_number,
        )
    elif customer_id is not None:
        query = db.query(Invoice).filter(
            Invoice.organization_id == org_id,
            Invoice.customer_id == customer_id,
            Invoice.invoice_number == invoice_number,
        )
    else:
        return None
    if exclude_invoice_id is not None:
        query = query.filter(Invoice.id != exclude_invoice_id)
    return query.first()


def totals_by_currency(invoices: list[Invoice]) -> dict[str, Decimal]:
    """Sums Invoice.total grouped by currency - never mix a $100 invoice
    and a €100 invoice into one meaningless "$200" figure. Real bug found
    while auditing this file: every caller of this used to sum .total
    across all invoices regardless of currency, silently, because the
    original synthetic dataset happened to be all-USD."""
    totals: dict[str, Decimal] = {}
    for inv in invoices:
        totals[inv.currency] = totals.get(inv.currency, Decimal("0")) + inv.total
    return totals


def generate_financial_summary(db: Session, org_id: int) -> dict:
    all_invoices = db.query(Invoice).filter(Invoice.organization_id == org_id).all()
    incoming = [i for i in all_invoices if i.direction == InvoiceDirection.incoming]
    outgoing = [i for i in all_invoices if i.direction == InvoiceDirection.outgoing]

    def unpaid(invoices):
        return [i for i in invoices if i.payment_status != PaymentStatus.paid]

    overdue = [i for i in all_invoices if i.due_date < date.today() and i.payment_status != PaymentStatus.paid]

    return {
        "total_payable_outstanding_by_currency": totals_by_currency(unpaid(incoming)),
        "total_receivable_outstanding_by_currency": totals_by_currency(unpaid(outgoing)),
        "count_overdue": len(overdue),
        "overdue_total_by_currency": totals_by_currency(overdue),
        "count_invoices": len(all_invoices),
    }


def format_money_by_currency(totals: dict[str, Decimal]) -> str:
    """Renders a {currency: amount} dict as human text - "$300.00" for the
    common single-currency case, "$300.00 + €50.00" if more than one
    currency is actually present. Never adds different currencies
    together into one number."""
    if not totals:
        return "$0.00"
    parts = [f"{amount:.2f} {currency}" for currency, amount in sorted(totals.items())]
    return " + ".join(parts)


AGING_BUCKETS = [("0-30", 0, 30), ("31-60", 31, 60), ("61-90", 61, 90), ("90+", 91, None)]


def compute_aging_report(overdue_invoices: list[Invoice]) -> list[dict]:
    """Standard AR/AP aging breakdown: how many overdue invoices fall into
    each days-overdue bucket, and how much they total (per currency -
    see totals_by_currency). Bucketed by days overdue rather than a
    single "overdue" flag, since a 5-day-late invoice and a 120-day-late
    one need very different attention."""
    today = date.today()
    buckets = []
    for label, lo, hi in AGING_BUCKETS:
        in_bucket = [
            inv for inv in overdue_invoices
            if (today - inv.due_date).days >= lo and (hi is None or (today - inv.due_date).days <= hi)
        ]
        buckets.append({
            "label": label,
            "count": len(in_bucket),
            "total_by_currency": totals_by_currency(in_bucket),
        })
    return buckets


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
    overdue_invoices = [
        i for i in all_invoices if i.due_date < date.today() and i.payment_status != PaymentStatus.paid
    ]
    # ApprovalRequest has no org_id column - fine for a single-organization
    # demo, would need scoping if this became multi-tenant.
    pending_approvals = db.query(ApprovalRequest).filter(ApprovalRequest.status == "pending").count()

    return {
        **summary,
        "count_paid": len(paid),
        "total_paid_amount_by_currency": totals_by_currency(paid),
        "suspicious_invoices": sorted(suspicious, key=lambda i: float(i.risk_score), reverse=True),
        "pending_approvals": pending_approvals,
        "aging_report": compute_aging_report(overdue_invoices),
    }
