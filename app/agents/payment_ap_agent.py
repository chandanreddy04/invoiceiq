"""
Payment/Accounts Payable Agent - prioritizes unpaid invoices. It NEVER
executes a payment; per Section 12's permission matrix, that always
requires a human, regardless of confidence. This agent's only output
is a ranked recommendation a person reads and acts on manually (by
marking an invoice paid on its own page) - there is deliberately no
"pay now" button anywhere connected to this agent.

  INPUT     -> every unpaid incoming invoice
  REASONING -> compute_priority(): plain arithmetic. Overdue invoices
               rank first (more overdue = higher); invoices already
               flagged risky by the Fraud/Risk Agent are held back
               instead of recommended, regardless of how overdue they
               are - paying a suspicious invoice quickly is exactly
               the wrong move.
  LLM LAYER -> one call to narrate the overall recommendation in
               plain English for whoever reviews it
  ACTION    -> none (read-only agent, like Financial Analysis)
"""

import json
import logging
from datetime import date
from decimal import Decimal

from sqlalchemy.orm import Session

from app.models.models import Invoice, InvoiceDirection, PaymentStatus
from app.services.llm_extraction_service import MODEL_NAME, LLMUnavailableError

logger = logging.getLogger(__name__)

RISK_HOLD_THRESHOLD = 0.5


def compute_priority(invoice: Invoice) -> tuple[float, bool]:
    """Returns (priority_score, held_for_risk)."""
    if invoice.risk_score and float(invoice.risk_score) >= RISK_HOLD_THRESHOLD:
        return -1.0, True

    days_overdue = (date.today() - invoice.due_date).days
    if days_overdue > 0:
        score = 100 + days_overdue
    else:
        score = max(0, 30 - abs(days_overdue))
    return float(score), False


def prioritize_payments(db: Session, org_id: int) -> dict:
    unpaid = (
        db.query(Invoice)
        .filter(
            Invoice.organization_id == org_id,
            Invoice.direction == InvoiceDirection.incoming,
            Invoice.payment_status != PaymentStatus.paid,
        )
        .all()
    )

    ranked, held = [], []
    for inv in unpaid:
        score, is_held = compute_priority(inv)
        (held if is_held else ranked).append((score, inv))
    ranked.sort(key=lambda x: x[0], reverse=True)

    return {
        "recommended": [inv for _, inv in ranked],
        "held_for_review": [inv for _, inv in held],
        "explanation": _explain(ranked, held),
    }


def _explain(ranked: list, held: list) -> str:
    if not ranked and not held:
        return "No unpaid vendor invoices right now."

    lines = []
    try:
        import ollama
        summary_lines = [f"{inv.invoice_number} (${inv.total}, due {inv.due_date})" for _, inv in ranked[:5]]
        held_lines = [f"{inv.invoice_number} (${inv.total}, risk {float(inv.risk_score):.0%})" for _, inv in held]
        prompt = (
            "Write one short paragraph recommending a payment order for a small business "
            f"owner. Pay these in this order (most urgent first): {', '.join(summary_lines) or 'none'}. "
            f"Hold these back and do not pay yet, pending fraud review: {', '.join(held_lines) or 'none'}. "
            "Do not invent any other facts."
        )
        response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], options={"temperature": 0})
        return response["message"]["content"].strip()
    except Exception as e:
        logger.warning("Payment explanation LLM call failed, using plain summary: %s", e)
        for _, inv in ranked:
            lines.append(f"Pay {inv.invoice_number} (${inv.total}, due {inv.due_date}).")
        for _, inv in held:
            lines.append(f"HOLD {inv.invoice_number} - flagged {float(inv.risk_score):.0%} risk, needs review first.")
        return " ".join(lines) if lines else "No unpaid vendor invoices right now."
