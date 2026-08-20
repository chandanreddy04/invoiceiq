"""
Collections/AR Agent - the receivables mirror of the Payment/AP Agent.
Flagged as a real gap while exploring customer invoicing: Payment/AP
prioritizes which vendor invoices to pay, but nothing prioritized which
overdue customer invoices to chase - each one only ever got a reminder
drafted if a person happened to open that specific invoice and click
the button. This agent closes that gap by scanning every outstanding
outgoing invoice at once and ranking them.

Like Payment/AP, this agent NEVER contacts anyone itself - its only
output is a ranked recommendation. Actually drafting a reminder is
still communication_agent's job (the existing per-invoice "Draft
reminder" button), and sending is always a human action.

  INPUT     -> every unpaid outgoing invoice
  REASONING -> compute_priority(): plain arithmetic. More overdue ranks
               higher, same as Payment/AP. A genuine addition Payment/AP
               doesn't need: invoice size is a capped tie-breaker too -
               a $50 invoice 5 days late and a $50,000 invoice 5 days
               late are not equally urgent to chase. Invoices the
               Fraud/Risk Agent already flagged as higher collectability
               risk are marked for personal follow-up instead of an
               automated reminder - still in the queue (unlike Payment/
               AP's "hold back, don't pay yet", chasing a risky
               customer for money owed is exactly what you still need
               to do, just carefully, not automatically) but called out
               separately so a person knows to handle it differently.
  LLM LAYER -> one call to narrate the queue in plain English
  ACTION    -> none (read-only agent, like Payment/AP and Financial
               Analysis) - drafting/sending stays with the Communication
               Agent and a human, respectively.
"""

import logging
from datetime import date

from sqlalchemy.orm import Session

from app.models.models import Invoice, InvoiceDirection, PaymentStatus
from app.services.llm_client import LLMUnavailableError, chat

logger = logging.getLogger(__name__)

ESCALATE_RISK_THRESHOLD = 0.5
_AMOUNT_NUDGE_CAP = 20.0  # points - keeps invoice size a tie-breaker, never the dominant factor


def compute_priority(invoice: Invoice) -> tuple[float, bool]:
    """Returns (priority_score, needs_personal_follow_up). Same overdue
    scoring shape as payment_ap_agent.compute_priority(), plus an
    amount nudge (capped so a single huge invoice can't outrank
    something far more overdue - overdue invoices always score >=100,
    the nudge only ever breaks ties within that overdue band or within
    the not-yet-due band, never across them)."""
    days_overdue = (date.today() - invoice.due_date).days
    if days_overdue > 0:
        score = 100 + days_overdue
    else:
        score = max(0, 30 - abs(days_overdue))

    score += min(_AMOUNT_NUDGE_CAP, float(invoice.total) / 1000)

    needs_personal_follow_up = invoice.risk_score is not None and float(invoice.risk_score) >= ESCALATE_RISK_THRESHOLD
    return score, needs_personal_follow_up


def rank_outstanding_receivables(db: Session, org_id: int) -> tuple[list, list]:
    """Split out the same way payment_ap_agent.rank_unpaid_invoices() is,
    so a "regenerate live" streaming endpoint can re-narrate the exact
    same ranking without recomputing it."""
    outstanding = (
        db.query(Invoice)
        .filter(
            Invoice.organization_id == org_id,
            Invoice.direction == InvoiceDirection.outgoing,
            Invoice.payment_status != PaymentStatus.paid,
        )
        .all()
    )

    ranked, escalate = [], []
    for inv in outstanding:
        score, needs_follow_up = compute_priority(inv)
        (escalate if needs_follow_up else ranked).append((score, inv))
    ranked.sort(key=lambda x: x[0], reverse=True)
    escalate.sort(key=lambda x: x[0], reverse=True)
    return ranked, escalate


def prioritize_collections(db: Session, org_id: int) -> dict:
    ranked, escalate = rank_outstanding_receivables(db, org_id)
    return {
        "recommended": [inv for _, inv in ranked],
        "escalate": [inv for _, inv in escalate],
        "explanation": _explain(ranked, escalate),
    }


def _narration_prompt(ranked: list, escalate: list) -> str:
    summary_lines = [f"{inv.invoice_number} (${inv.total}, due {inv.due_date})" for _, inv in ranked[:5]]
    escalate_lines = [f"{inv.invoice_number} (${inv.total}, risk {float(inv.risk_score):.0%})" for _, inv in escalate]
    return (
        "Write one short paragraph recommending a collections follow-up order for a "
        f"small business owner. Chase these customers in this order (most urgent first): "
        f"{', '.join(summary_lines) or 'none'}. These need a personal phone call or direct "
        f"outreach rather than just an automated reminder, because they were flagged as "
        f"higher risk: {', '.join(escalate_lines) or 'none'}. Do not invent any other facts."
    )


def _explain(ranked: list, escalate: list) -> str:
    if not ranked and not escalate:
        return "No outstanding customer invoices to chase right now."

    try:
        return chat(messages=[{"role": "user", "content": _narration_prompt(ranked, escalate)}]).strip()
    except LLMUnavailableError as e:
        logger.warning("Collections explanation LLM call failed, using plain summary: %s", e)
        lines = [f"Follow up on {inv.invoice_number} (${inv.total}, due {inv.due_date})." for _, inv in ranked]
        lines += [
            f"ESCALATE {inv.invoice_number} - flagged {float(inv.risk_score):.0%} risk, contact personally."
            for _, inv in escalate
        ]
        return " ".join(lines) if lines else "No outstanding customer invoices to chase right now."


def explain_collections_stream(ranked: list, escalate: list):
    """Streaming twin of _explain(), for a "regenerate live" button -
    mirrors payment_ap_agent.explain_payments_stream()."""
    from app.services.llm_client import chat_stream
    if not ranked and not escalate:
        yield "No outstanding customer invoices to chase right now."
        return
    yield from chat_stream(messages=[{"role": "user", "content": _narration_prompt(ranked, escalate)}])
