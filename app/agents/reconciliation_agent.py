"""
The eighth agent, and a direct application of this project's central
architectural claim: every financial decision is deterministic, and the
LLM's only job is to narrate what deterministic code already decided.

  INPUT LAYER         -> a BankTransaction row, imported from a CSV
                        bank/ledger export (date, description, signed
                        amount - see web_reconciliation_upload).
  CONTEXT LAYER        -> _candidate_invoices(): every open (not yet
                        paid) invoice on the correct side of the ledger.
                        A negative bank amount can only settle an
                        incoming/vendor bill; a positive one can only
                        settle an outgoing/customer invoice. Currency
                        must also match. Never compares across
                        direction - a vendor bill and a customer invoice
                        sharing a coincidental total must never cross-match.
  REASONING LAYER      -> score_match() + find_best_match(): 100%
                        deterministic. "exact" needs an equal amount AND
                        a transaction date within MATCH_WINDOW_DAYS of
                        the invoice's due date. "likely" covers an equal
                        amount outside that window. Multiple exact
                        candidates is "ambiguous", not a pick - this
                        agent never guesses when more than one invoice
                        could be it.
  ACTION LAYER          -> reconcile_transaction(): auto-matches ONLY
                        the unambiguous case (exactly one exact
                        candidate) - creates a real Payment row (see
                        _record_payment()) and flips the invoice to
                        paid. Every other outcome (zero candidates,
                        multiple candidates, likely-only) is left
                        unmatched for a human to confirm or ignore
                        explicitly via confirm_match() - matching money
                        to an invoice is exactly the kind of action this
                        project's approval-gate philosophy says an agent
                        should never finish alone on anything less than
                        certain.
  LLM LAYER              -> explain_unmatched_with_llm(): the ONLY LLM
                        call in this agent, and only for transactions
                        still unmatched after reasoning runs. Narrates
                        the best candidate (if any) and why it fell
                        short of auto-match - never picks the match
                        itself, and is never called from
                        run_reconciliation()'s bulk scan (that would mean
                        one page load could trigger a pile of LLM calls -
                        the same mistake already made once and fixed for
                        the reasoning-model second opinion). It's called
                        on demand, one transaction at a time, from a
                        button click - same discipline as
                        fraud_risk_agent.explain_risk_with_llm(), and the
                        result is cached so it only ever runs once per
                        transaction.
  FEEDBACK LAYER         -> a human either confirms the suggested match
                        (confirm_match() - creates the same Payment row
                        the auto-match path does) or ignores it; both are
                        AuditLog entries, same as every other human
                        financial decision in this app.
"""

import logging

from sqlalchemy.orm import Session

from app.models.models import BankTransaction, Invoice, InvoiceDirection, PaymentStatus, Payment
from app.services.llm_client import LLMUnavailableError, chat

logger = logging.getLogger(__name__)

# A payment can land well before or (more often) after an invoice's due
# date - 45 days covers slow payers without being so wide it starts
# swallowing up genuinely unrelated invoices of the same amount.
MATCH_WINDOW_DAYS = 45


def _party_name(invoice: Invoice) -> str | None:
    if invoice.vendor is not None:
        return invoice.vendor.name
    if invoice.customer is not None:
        return invoice.customer.name
    return None


def _candidate_invoices(db: Session, org_id: int, txn: BankTransaction) -> list[Invoice]:
    direction = InvoiceDirection.incoming if txn.amount < 0 else InvoiceDirection.outgoing
    return (
        db.query(Invoice)
        .filter(
            Invoice.organization_id == org_id,
            Invoice.direction == direction,
            Invoice.payment_status != PaymentStatus.paid,
            Invoice.currency == txn.currency,
            Invoice.total == abs(txn.amount),
        )
        .all()
    )


def score_match(txn: BankTransaction, invoice: Invoice) -> str:
    """Pure, deterministic, and the only function here that decides
    match quality - always called with an invoice whose amount already
    equals the transaction's (see _candidate_invoices()), so this is
    purely a question of how well the date and description corroborate
    it."""
    days_from_due = abs((txn.transaction_date - invoice.due_date).days)
    if days_from_due <= MATCH_WINDOW_DAYS:
        return "exact"
    return "likely"


def find_best_match(db: Session, org_id: int, txn: BankTransaction) -> tuple[Invoice | None, str | None]:
    """Returns (candidate, confidence) or (None, None/"ambiguous"). Only
    ever auto-matched by reconcile_transaction() when confidence ==
    "exact" AND there is exactly one such candidate."""
    candidates = _candidate_invoices(db, org_id, txn)
    if not candidates:
        return None, None

    scored = [(inv, score_match(txn, inv)) for inv in candidates]
    exact = [inv for inv, conf in scored if conf == "exact"]
    if len(exact) == 1:
        return exact[0], "exact"
    if len(exact) > 1:
        return None, "ambiguous"  # more than one same-amount, in-window invoice - a human must pick

    # No unambiguous exact match - surface the most recently due "likely"
    # candidate as a suggestion only; never auto-matched.
    likely = sorted(scored, key=lambda pair: pair[0].due_date, reverse=True)
    return likely[0][0], "likely"


def _record_payment(db: Session, invoice: Invoice, txn: BankTransaction) -> Payment:
    """The one place this agent writes money-movement to the database."""
    payment = Payment(
        invoice_id=invoice.id, amount=abs(txn.amount), paid_date=txn.transaction_date,
        method="bank_transfer", status="completed",
    )
    db.add(payment)
    invoice.payment_status = PaymentStatus.paid
    return payment


def reconcile_transaction(db: Session, org_id: int, txn: BankTransaction) -> None:
    """Deterministic top-level entry point for one transaction - no LLM
    involved. Safe to call repeatedly on the same still-unmatched
    transaction (run_reconciliation() does, every time the Reconciliation
    page loads, in case a new invoice now makes it an unambiguous match)."""
    invoice, confidence = find_best_match(db, org_id, txn)

    if confidence == "exact":
        _record_payment(db, invoice, txn)
        txn.status = "matched"
        txn.matched_invoice_id = invoice.id
        txn.suggested_invoice_id = None
        txn.match_confidence = "exact"
        return

    txn.match_confidence = confidence  # "likely" | "ambiguous" | None
    txn.suggested_invoice_id = invoice.id if invoice is not None else None


def confirm_match(db: Session, invoice: Invoice, txn: BankTransaction) -> None:
    """The human-confirmed counterpart to reconcile_transaction()'s
    auto-match path - same _record_payment() call, so a confirmed
    "likely" match produces exactly the same Payment row an "exact"
    auto-match would have."""
    _record_payment(db, invoice, txn)
    txn.status = "matched"
    txn.matched_invoice_id = invoice.id


def run_reconciliation(db: Session, org_id: int) -> dict:
    """Runs reconcile_transaction() over every currently-unmatched
    BankTransaction for this org. No LLM calls here at all - see this
    module's docstring for why that's deliberate."""
    unmatched = (
        db.query(BankTransaction)
        .filter(BankTransaction.organization_id == org_id, BankTransaction.status == "unmatched")
        .all()
    )
    matched_now = 0
    for txn in unmatched:
        reconcile_transaction(db, org_id, txn)
        if txn.status == "matched":
            matched_now += 1
    db.commit()
    return {"scanned": len(unmatched), "matched": matched_now, "still_unmatched": len(unmatched) - matched_now}


def _explanation_prompt(txn: BankTransaction, suggested: Invoice | None) -> str:
    if suggested is None:
        return (
            f"A bank transaction dated {txn.transaction_date} for {abs(txn.amount)} {txn.currency} "
            f"(description: \"{txn.description}\") has no invoice on file with a matching amount. "
            "In one short sentence, tell a bookkeeper what to check (e.g. an invoice not yet "
            "entered in the system, a partial payment, or an unrelated transaction)."
        )
    return (
        f"A bank transaction dated {txn.transaction_date} for {abs(txn.amount)} {txn.currency} "
        f"(description: \"{txn.description}\") is a possible but not certain match for invoice "
        f"#{suggested.invoice_number} (due {suggested.due_date}, total {suggested.total} {suggested.currency}). "
        f"Match confidence: {txn.match_confidence}. In one short sentence, explain why it wasn't "
        "auto-matched and what a bookkeeper should check before confirming it."
    )


def explain_unmatched_with_llm(txn: BankTransaction, suggested: Invoice | None) -> str:
    """The only LLM call in this agent. Never called from
    run_reconciliation()'s bulk scan - only on demand, one transaction at
    a time, from a button click on the Reconciliation page. Falls back to
    a plain templated sentence if the LLM is unavailable, same discipline
    as every other explain_*_with_llm() in this codebase."""
    try:
        return chat(messages=[{"role": "user", "content": _explanation_prompt(txn, suggested)}])
    except LLMUnavailableError:
        if suggested is None:
            return "No invoice on file matches this amount - check for an invoice not yet entered, or confirm this transaction is unrelated."
        return (
            f"Possible match to invoice #{suggested.invoice_number}, but not close enough to "
            "auto-confirm - check the date and amount before confirming."
        )
