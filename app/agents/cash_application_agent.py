"""
The ninth agent - and the one that picks up exactly where Reconciliation
Agent has to give up. Reconciliation only ever considers a transaction
whose amount equals ONE open invoice's total exactly; anything a
customer pays that doesn't line up 1:1 (a partial payment, or one
remittance covering several invoices at once) is real money that agent
was never built to place. This one is.

Same rule as every other agent in this project, applied to the hardest
version of it yet: deciding *which invoices* a payment settles is
matching, not judgment, so it stays deterministic - but deciding to
actually APPLY that match is exactly the kind of action this project's
approval-gate philosophy says an agent should never finish alone, even
when the deterministic match is unambiguous. Every allocation this
agent proposes waits for a human click before a single Payment row
gets written.

  INPUT LAYER          -> a BankTransaction Reconciliation Agent already
                        scanned and left unmatched with no suggestion of
                        its own (see run_cash_application()'s filter -
                        this agent never re-litigates a case
                        Reconciliation already has an answer for).
  CONTEXT LAYER         -> every open (not yet paid) outgoing/customer
                        invoice, scoped to one customer at a time -
                        this agent never mixes two customers' invoices
                        into one allocation, even if their totals
                        happen to sum correctly. A split payment from
                        one customer for one customer's bills is a real
                        pattern; a coincidence across two customers isn't.
  REASONING LAYER        -> find_split_candidates() first (an exact-sum
                        match across several invoices is stronger
                        evidence than a fractional guess), then
                        find_partial_candidate() (a smaller-than-the-
                        invoice amount, at least MIN_PARTIAL_FRACTION of
                        it, and the only such candidate). Both are 100%
                        deterministic - a bounded, explainable search
                        over a customer's actual open invoices, not a
                        general subset-sum solver.
  ACTION LAYER            -> propose_allocation() only ever WRITES a
                        SuggestedAllocation - a proposal, not a Payment.
                        apply_allocation() is the one place money
                        actually moves, and it only ever runs from a
                        human's explicit "Apply" click (see
                        web_reconciliation_apply_allocation). It sets
                        each invoice to "paid" if the allocation covers
                        its full remaining balance or "partially_paid"
                        otherwise - the first code in this project to
                        ever compute that status; before this it was a
                        dropdown option nothing automatic ever selected.
  LLM LAYER                -> explain_allocation_with_llm(): the only
                        LLM call here, on demand and cached, same
                        discipline as reconciliation_agent's explain_
                        unmatched_with_llm() - narrates a proposed
                        allocation for a human to sanity-check, never
                        picks or changes it.
  FEEDBACK LAYER           -> a human applies (writes real Payment rows)
                        or the transaction is later ignored via
                        Reconciliation's own ignore action - both
                        AuditLog entries, same as every other financial
                        decision a person makes in this app.
"""

import logging
from decimal import Decimal
from itertools import combinations

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.models import (
    BankTransaction, Invoice, InvoiceDirection, PaymentStatus, Payment, SuggestedAllocation,
)
from app.services.llm_client import LLMUnavailableError, chat

logger = logging.getLogger(__name__)

# A partial-payment candidate must be at least this fraction of the
# invoice's total - without a floor, a trivially small transaction could
# get called a "partial payment" against any much larger unpaid invoice.
MIN_PARTIAL_FRACTION = Decimal("0.10")

# Bounded, explainable search - real remittances bundle a handful of
# invoices, not dozens. A general subset-sum solver over an unbounded
# invoice list isn't the point; a small, traceable combination search is.
MAX_SPLIT_INVOICES = 6


def _open_customer_invoices(db: Session, org_id: int, customer_id: int, currency: str) -> list[Invoice]:
    return (
        db.query(Invoice)
        .filter(
            Invoice.organization_id == org_id,
            Invoice.direction == InvoiceDirection.outgoing,
            Invoice.customer_id == customer_id,
            Invoice.payment_status != PaymentStatus.paid,
            Invoice.currency == currency,
        )
        .all()
    )


def find_split_candidates(db: Session, org_id: int, txn: BankTransaction) -> list[Invoice] | None:
    """A transaction whose amount exactly equals the sum of 2+ open
    invoices for the SAME customer. Only ever considers money coming IN
    (positive amounts) - cash application is an accounts-receivable
    concept, not something that applies to a vendor bill we're paying."""
    if txn.amount <= 0:
        return None

    customer_ids = [
        row[0] for row in db.query(Invoice.customer_id).filter(
            Invoice.organization_id == org_id, Invoice.direction == InvoiceDirection.outgoing,
            Invoice.payment_status != PaymentStatus.paid, Invoice.customer_id.isnot(None),
        ).distinct().all()
    ]
    for customer_id in customer_ids:
        invoices = _open_customer_invoices(db, org_id, customer_id, txn.currency)
        if len(invoices) < 2:
            continue
        for size in range(2, min(len(invoices), MAX_SPLIT_INVOICES) + 1):
            for combo in combinations(invoices, size):
                if sum((inv.total for inv in combo), Decimal("0")) == txn.amount:
                    return list(combo)
    return None


def find_partial_candidate(db: Session, org_id: int, txn: BankTransaction) -> Invoice | None:
    """A transaction smaller than a single open invoice's total - a
    plausible partial payment, but only when it's the single such
    candidate and clears MIN_PARTIAL_FRACTION of the invoice."""
    if txn.amount <= 0:
        return None

    candidates = [
        inv for inv in db.query(Invoice).filter(
            Invoice.organization_id == org_id, Invoice.direction == InvoiceDirection.outgoing,
            Invoice.payment_status != PaymentStatus.paid, Invoice.currency == txn.currency,
            Invoice.total > txn.amount,
        ).all()
        if txn.amount >= inv.total * MIN_PARTIAL_FRACTION
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def propose_allocation(db: Session, org_id: int, txn: BankTransaction) -> None:
    """Deterministic top-level entry - clears any stale proposal from a
    previous scan, then tries a split (exact-sum evidence is stronger)
    before falling back to a single-invoice partial. Writes
    SuggestedAllocation rows only - never a Payment."""
    db.query(SuggestedAllocation).filter(SuggestedAllocation.bank_transaction_id == txn.id).delete()

    split = find_split_candidates(db, org_id, txn)
    if split:
        for inv in split:
            db.add(SuggestedAllocation(bank_transaction_id=txn.id, invoice_id=inv.id, amount=inv.total, kind="split_share"))
        return

    partial = find_partial_candidate(db, org_id, txn)
    if partial:
        db.add(SuggestedAllocation(bank_transaction_id=txn.id, invoice_id=partial.id, amount=txn.amount, kind="partial"))


def run_cash_application(db: Session, org_id: int) -> dict:
    """Only ever looks at transactions Reconciliation Agent already left
    unmatched with NO suggestion of its own - a "likely" single-invoice
    suggestion from Reconciliation already covers that transaction, so
    this agent stays out of its way rather than proposing a competing
    answer."""
    candidates = (
        db.query(BankTransaction)
        .filter(
            BankTransaction.organization_id == org_id,
            BankTransaction.status == "unmatched",
            BankTransaction.suggested_invoice_id.is_(None),
        )
        .all()
    )
    proposed = 0
    for txn in candidates:
        propose_allocation(db, org_id, txn)
        db.flush()
        if db.query(SuggestedAllocation).filter(SuggestedAllocation.bank_transaction_id == txn.id).count() > 0:
            proposed += 1
    db.commit()
    return {"scanned": len(candidates), "proposed": proposed}


def apply_allocation(db: Session, txn: BankTransaction) -> None:
    """The one place this agent moves money - only ever called from a
    human's explicit "Apply" click. Turns every SuggestedAllocation row
    for this transaction into a real Payment row (see reconciliation_
    agent._record_payment - this is that same pattern, just possibly
    writing several rows instead of one), and sets each invoice to
    "paid" once the allocation covers its full remaining balance, or
    "partially_paid" otherwise."""
    allocations = db.query(SuggestedAllocation).filter(SuggestedAllocation.bank_transaction_id == txn.id).all()
    for alloc in allocations:
        invoice = alloc.invoice
        already_paid = db.query(func.coalesce(func.sum(Payment.amount), 0)).filter(Payment.invoice_id == invoice.id).scalar()
        db.add(Payment(
            invoice_id=invoice.id, amount=alloc.amount, paid_date=txn.transaction_date,
            method="bank_transfer", status="completed", bank_transaction_id=txn.id,
        ))
        total_paid = Decimal(already_paid) + alloc.amount
        invoice.payment_status = PaymentStatus.paid if total_paid >= invoice.total else PaymentStatus.partially_paid

    if len(allocations) == 1:
        txn.matched_invoice_id = allocations[0].invoice_id
    txn.status = "matched"
    db.query(SuggestedAllocation).filter(SuggestedAllocation.bank_transaction_id == txn.id).delete()


def _allocation_prompt(txn: BankTransaction, allocations: list[SuggestedAllocation]) -> str:
    lines = "\n".join(
        f"- Invoice #{a.invoice.invoice_number}: apply {a.amount} {txn.currency} "
        f"({'covers its full remaining balance' if a.kind == 'split_share' else f'partial payment toward a total of {a.invoice.total}'})"
        for a in allocations
    )
    kind = f"split across {len(allocations)} invoices" if len(allocations) > 1 else "a partial payment against one invoice"
    return (
        f"A bank transaction dated {txn.transaction_date} for {txn.amount} {txn.currency} "
        f"(description: \"{txn.description}\") looks like {kind}:\n{lines}\n"
        "In one short sentence, tell a bookkeeper what to double-check before applying this."
    )


def explain_allocation_with_llm(txn: BankTransaction, allocations: list[SuggestedAllocation]) -> str:
    """The only LLM call in this agent. On demand, one transaction at a
    time, cached on BankTransaction.explanation - same field and
    discipline reconciliation_agent already uses for its own unmatched
    narration."""
    try:
        return chat(messages=[{"role": "user", "content": _allocation_prompt(txn, allocations)}])
    except LLMUnavailableError:
        if len(allocations) > 1:
            return f"Looks like one payment covering {len(allocations)} invoices - check the totals line up before applying."
        return "Looks like a partial payment - check the invoice number in the description before applying."
