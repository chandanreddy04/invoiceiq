"""
Communication Agent - drafts messages, never sends them. Sending is a
separate, explicit human action (see the /web/communications page) -
this agent's only allowed action is writing a draft row with
status="draft" (Section 12 permission matrix: REQUIRES HUMAN APPROVAL
to send, FORBIDDEN to send unilaterally).

  INPUT     -> an invoice + how overdue it is + its risk score
  REASONING -> plain code decides the PURPOSE and tone (overdue
               payment reminder vs. a vendor follow-up flagging a risk
               concern) - the LLM does not choose whether to be firm
               or polite on its own, it's told which situation applies
  LLM LAYER -> writes the actual subject/body text for that situation
  ACTION    -> saves a draft Communication row
  FEEDBACK  -> caller can see the row was written; nothing external
               ever happens here
"""

import json
import logging
from datetime import date

from sqlalchemy.orm import Session

from app.models.models import Invoice, Communication, InvoiceDirection, ApprovalRequest
from app.schemas.communication import DraftedReminder
from app.services.llm_extraction_service import MODEL_NAME, LLMUnavailableError

logger = logging.getLogger(__name__)


def _situation_for(invoice: Invoice) -> tuple[str, str]:
    """Returns (purpose, instructions) - deterministic, not LLM-decided."""
    days_overdue = (date.today() - invoice.due_date).days

    if invoice.direction == InvoiceDirection.outgoing:
        if days_overdue > 0:
            return (
                "overdue payment reminder to a customer",
                f"This invoice is {days_overdue} day(s) overdue. Be polite but clear that payment "
                "is due. Do not threaten legal action or late fees unless told to.",
            )
        return (
            "friendly upcoming payment reminder to a customer",
            f"This invoice is due on {invoice.due_date} (not yet overdue). Send a brief, friendly heads-up.",
        )

    # incoming: a vendor invoice we received
    if invoice.risk_score and float(invoice.risk_score) >= 0.5:
        return (
            "a cautious inquiry to a vendor about a flagged invoice",
            "This invoice was flagged as higher risk by our internal review. Politely ask the "
            "vendor to confirm the invoice details (amount and invoice number) before we process payment. "
            "Do not accuse them of anything.",
        )
    return (
        "a routine acknowledgement to a vendor",
        "Simply confirm receipt of the invoice and that it will be paid per the stated terms.",
    )


def draft_reminder(db: Session, invoice: Invoice) -> Communication | None:
    party = invoice.vendor if invoice.direction == InvoiceDirection.incoming else invoice.customer
    if party is None or not party.email:
        return None

    purpose, instructions = _situation_for(invoice)
    prompt = (
        f"Write {purpose} for invoice #{invoice.invoice_number}, amount {invoice.total} {invoice.currency}, "
        f"due date {invoice.due_date}, addressed to {party.name}.\n\n{instructions}\n\n"
        "Keep it under 120 words. Sign off as 'Maple Street Bakery Supply Co.'"
    )

    try:
        import ollama
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            format=DraftedReminder.model_json_schema(),
            options={"temperature": 0.3},
        )
        drafted = DraftedReminder.model_validate(json.loads(response["message"]["content"]))
    except Exception as e:
        logger.warning("Communication draft LLM call failed: %s", e)
        raise LLMUnavailableError(str(e)) from e

    comm = Communication(
        invoice_id=invoice.id,
        recipient=party.email,
        subject=drafted.subject,
        body=drafted.body,
        status="draft",
    )
    db.add(comm)
    db.flush()  # assigns comm.id so the approval request can reference it

    db.add(ApprovalRequest(
        type="send_communication",
        related_id=comm.id,
        requested_by_agent="communication_agent",
        reason=f"Draft ready for {party.name}: \"{drafted.subject}\" ({purpose}).",
    ))
    db.commit()
    db.refresh(comm)
    return comm
