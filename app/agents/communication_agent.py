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

from app.core.config import APP_BASE_URL
from app.models.models import Invoice, Communication, InvoiceDirection, ApprovalRequest
from app.schemas.communication import DraftedReminder
from app.services.llm_client import chat

logger = logging.getLogger(__name__)


def _pay_link(db: Session, invoice: Invoice) -> str | None:
    """Only for outgoing invoices, and only once APP_BASE_URL is set -
    otherwise the link would be relative/broken, which is worse than
    just not offering one. Local import to avoid a circular import at
    module load time (invoice_service.create_invoice() imports agents
    lazily for the same reason)."""
    if invoice.direction != InvoiceDirection.outgoing or not APP_BASE_URL:
        return None
    from app.services.invoice_service import get_or_create_public_token
    token = get_or_create_public_token(db, invoice)
    return f"{APP_BASE_URL.rstrip('/')}/pay/{token}" if token else None


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


def _save_draft(db: Session, invoice: Invoice, party, drafted: DraftedReminder, reason: str) -> Communication:
    """Shared by draft_reminder() and draft_invoice_email(): both write
    the same shape of Communication + ApprovalRequest, only the LLM
    prompt/fallback content that produced `drafted` differs between them."""
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
        reason=reason,
    ))
    db.commit()
    db.refresh(comm)
    return comm


def draft_reminder(db: Session, invoice: Invoice) -> Communication | None:
    party = invoice.vendor if invoice.direction == InvoiceDirection.incoming else invoice.customer
    if party is None or not party.email:
        return None

    purpose, instructions = _situation_for(invoice)
    pay_link = _pay_link(db, invoice)
    if pay_link:
        instructions += f" Include this link so they can view and pay online: {pay_link}"
    prompt = (
        f"Write {purpose} for invoice #{invoice.invoice_number}, amount {invoice.total} {invoice.currency}, "
        f"due date {invoice.due_date}, addressed to {party.name}.\n\n{instructions}\n\n"
        "Keep it under 120 words. Sign off as 'Maple Street Bakery Supply Co.'"
    )

    try:
        content = chat(
            messages=[{"role": "user", "content": prompt}],
            schema=DraftedReminder.model_json_schema(),
            temperature=0.3,
        )
        drafted = DraftedReminder.model_validate(json.loads(content))
    except Exception as e:
        # Every other agent in this app has a fallback for this exact
        # case (fraud/classification/extraction all degrade instead of
        # doing nothing) - this one didn't, until a live test on the
        # cloud deployment (no LLM reachable there) surfaced that it
        # silently created no draft at all. A template with no LLM
        # involvement is a better failure mode than silence for
        # something a human is about to review and approve anyway.
        logger.warning("Communication draft LLM call failed, using template fallback: %s", e)
        drafted = DraftedReminder(
            subject=f"[Template - LLM unavailable] {purpose.capitalize()} - Invoice #{invoice.invoice_number}",
            body=(
                f"Dear {party.name},\n\n"
                f"This is regarding invoice #{invoice.invoice_number} for {invoice.total} {invoice.currency}, "
                f"due {invoice.due_date}.\n\n{instructions}"
                + (f"\n\nView and pay online: {pay_link}" if pay_link else "") +
                "\n\n(This message was generated from a template because the AI drafting service was "
                "unavailable - review and edit before sending.)\n\n"
                "Sincerely,\nMaple Street Bakery Supply Co."
            ),
        )

    return _save_draft(db, invoice, party, drafted, reason=f"Draft ready for {party.name}: \"{drafted.subject}\" ({purpose}).")


def draft_invoice_email(db: Session, invoice: Invoice) -> Communication | None:
    """Drafts the initial "here is your invoice" email - distinct from
    draft_reminder(), which assumes the customer already knows about
    the invoice (an upcoming/overdue nudge). This is the message that
    delivers the invoice itself for the first time, and references the
    real PDF generated by invoice_pdf_service.generate_invoice_pdf() -
    downloadable from the invoice's own page - since this app's email
    sending is simulated (see Communication's docstring), the PDF is
    what a person actually attaches if they send this for real. Only
    meaningful for outgoing/customer invoices."""
    if invoice.direction != InvoiceDirection.outgoing:
        return None
    customer = invoice.customer
    if customer is None or not customer.email:
        return None

    purpose = "sending a new invoice to a customer for the first time"
    pay_link = _pay_link(db, invoice)
    instructions = (
        f"The invoice PDF is attached. State the amount due ({invoice.total} {invoice.currency}) and the "
        f"due date ({invoice.due_date}). This is not a reminder and nothing is overdue - the customer is "
        "seeing this invoice for the first time, so keep the tone professional and welcoming."
    )
    if pay_link:
        instructions += f" Include this link so they can view and pay it online: {pay_link}"
    prompt = (
        f"Write an email {purpose} for invoice #{invoice.invoice_number}, amount {invoice.total} "
        f"{invoice.currency}, due date {invoice.due_date}, addressed to {customer.name}.\n\n{instructions}\n\n"
        "Keep it under 120 words. Sign off as 'Maple Street Bakery Supply Co.'"
    )

    try:
        content = chat(
            messages=[{"role": "user", "content": prompt}],
            schema=DraftedReminder.model_json_schema(),
            temperature=0.3,
        )
        drafted = DraftedReminder.model_validate(json.loads(content))
    except Exception as e:
        logger.warning("Invoice email LLM call failed, using template fallback: %s", e)
        drafted = DraftedReminder(
            subject=f"[Template - LLM unavailable] Invoice #{invoice.invoice_number}",
            body=(
                f"Dear {customer.name},\n\n"
                f"Please find attached invoice #{invoice.invoice_number} for {invoice.total} {invoice.currency}, "
                f"due {invoice.due_date}."
                + (f"\n\nView and pay online: {pay_link}" if pay_link else "") +
                "\n\n(This message was generated from a template because the AI drafting service was "
                "unavailable - review and edit before sending.)\n\n"
                "Sincerely,\nMaple Street Bakery Supply Co."
            ),
        )

    return _save_draft(
        db, invoice, customer, drafted,
        reason=f"Invoice email ready for {customer.name}: \"{drafted.subject}\" - download the PDF from "
               f"the invoice page to attach before sending for real.",
    )
