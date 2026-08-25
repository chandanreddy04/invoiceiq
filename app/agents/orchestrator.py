"""
The first real Orchestrator. Before this, invoice_service.create_invoice()
called the Fraud/Risk Agent directly - a hardcoded shortcut, flagged as
temporary in Phase 4. This replaces that with actual routing:

  1. receive a task (here: "a new invoice was created")
  2. run the relevant agents in a defined order
  3. log every step - agent name, status, timing, errors - to AgentLog
  4. one agent failing does not stop the others (Section 38)
  5. return a summary of what happened

This is still deliberately simple: a fixed sequence, not a dynamically
planned one (no LLM decides the order here - see Section 7, "agent
routing = orchestrator/framework logic", not an LLM call). A real
planning step (the LLM deciding *which* agents a request needs) comes
in Phase 6 once there's more than one kind of task to route between.
"""

import logging
import time
import uuid

from sqlalchemy.orm import Session

from app.models.models import Invoice, AgentLog
from app.agents import fraud_risk_agent, classification_agent, financial_analysis_agent, collections_agent, payment_ap_agent, reconciliation_agent

logger = logging.getLogger(__name__)


def _log_step(db: Session, task_id: str, invoice_id: int, agent_name: str, fn, input_summary: str, summarize=str):
    """summarize turns fn()'s return value into the human-readable text
    stored in AgentLog.output_summary - str() by default (fine for the
    small dicts/strings every other agent here returns), overridable
    for an agent like collections_agent whose result dict holds raw
    Invoice ORM objects that str() would render as unreadable reprs."""
    start = time.monotonic()
    try:
        result = fn()
        duration_ms = int((time.monotonic() - start) * 1000)
        db.add(AgentLog(
            task_id=task_id, invoice_id=invoice_id, agent_name=agent_name, status="success",
            input_summary=input_summary, output_summary=summarize(result)[:500], duration_ms=duration_ms,
        ))
        db.commit()
        return result
    except Exception as e:
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.exception("Agent %s failed on invoice %s", agent_name, invoice_id)
        db.add(AgentLog(
            task_id=task_id, invoice_id=invoice_id, agent_name=agent_name, status="failed",
            input_summary=input_summary, output_summary=None, duration_ms=duration_ms, error=str(e)[:1000],
        ))
        db.commit()
        return None


def run_invoice_pipeline(db: Session, invoice: Invoice) -> str:
    """Runs every agent relevant to a newly created invoice. Returns the
    task_id so callers can look up the full trace in AgentLog."""
    task_id = str(uuid.uuid4())[:8]

    # A real gap found while exploring customer invoicing: this used to
    # gate on invoice.vendor_id, so outgoing/customer invoices got no
    # risk assessment at all. fraud_risk_agent now reasons over
    # whichever party an invoice actually has (see get_party()), so
    # nothing needs to be decided here anymore - just always run it.
    _log_step(
        db, task_id, invoice.id, "fraud_risk_agent",
        lambda: fraud_risk_agent.run_fraud_check(db, invoice) and "flagged",
        input_summary=f"invoice #{invoice.invoice_number}, total={invoice.total}",
    )

    _log_step(
        db, task_id, invoice.id, "classification_agent",
        lambda: classification_agent.run_classification(db, invoice),
        input_summary=f"{len(invoice.items)} line item(s)",
    )

    return task_id


def route_query(db: Session, org_id: int, question: str) -> dict:
    """The second kind of task the Orchestrator handles: an open-ended
    English question instead of a new-invoice event. Routing here is
    still trivial (there's only one agent that answers questions), but
    the *intent parsing inside* financial_analysis_agent is genuinely
    LLM-decided - see that agent's docstring for why that split matters."""
    task_id = str(uuid.uuid4())[:8]
    result = _log_step(
        db, task_id, None, "financial_analysis_agent",
        lambda: financial_analysis_agent.answer_question(db, org_id, question),
        input_summary=question[:200],
    )
    return result or {"question": question, "answer": "The assistant is unavailable right now.", "result_count": 0}


def run_collections_scan(db: Session, org_id: int) -> dict:
    """The third kind of task the Orchestrator handles: scanning every
    outstanding customer invoice and prioritizing which to chase -
    triggered by visiting the Collections page. Logged to AgentLog like
    run_payment_scan() below, so it shows up on Agent Activity."""
    task_id = str(uuid.uuid4())[:8]
    result = _log_step(
        db, task_id, None, "collections_agent",
        lambda: collections_agent.prioritize_collections(db, org_id),
        input_summary=f"org #{org_id}: scan outstanding customer invoices",
        summarize=lambda r: f"{len(r['recommended'])} recommended, {len(r['escalate'])} to escalate",
    )
    return result or {"recommended": [], "escalate": [], "explanation": "The assistant is unavailable right now."}


def run_payment_scan(db: Session, org_id: int) -> dict:
    """The fourth kind of task the Orchestrator handles: scanning every
    unpaid vendor invoice and prioritizing which to pay - triggered by
    visiting the Payments page. Payment/AP's own docstring predates the
    Orchestrator being wired to log page-triggered scans at all (only
    the invoice pipeline and AI Assistant queries were logged, until
    Collections/AR added the pattern for its own page) - this brings
    Payment/AP's runs onto Agent Activity too, the same way."""
    task_id = str(uuid.uuid4())[:8]
    result = _log_step(
        db, task_id, None, "payment_ap_agent",
        lambda: payment_ap_agent.prioritize_payments(db, org_id),
        input_summary=f"org #{org_id}: scan unpaid vendor invoices",
        summarize=lambda r: f"{len(r['recommended'])} recommended, {len(r['held_for_review'])} held for review",
    )
    return result or {"recommended": [], "held_for_review": [], "explanation": "The assistant is unavailable right now."}


def run_reconciliation_scan(db: Session, org_id: int) -> dict:
    """The fifth kind of task the Orchestrator handles: matching
    imported bank transactions against open invoices - triggered by a
    CSV upload and again every time the Reconciliation page loads (a
    newly created invoice might now make a previously-unmatched
    transaction an unambiguous match). No LLM call happens here - see
    reconciliation_agent's own docstring for why that's deliberate."""
    task_id = str(uuid.uuid4())[:8]
    result = _log_step(
        db, task_id, None, "reconciliation_agent",
        lambda: reconciliation_agent.run_reconciliation(db, org_id),
        input_summary=f"org #{org_id}: reconcile unmatched bank transactions",
        summarize=lambda r: f"{r['matched']} matched, {r['still_unmatched']} still unmatched",
    )
    return result or {"scanned": 0, "matched": 0, "still_unmatched": 0}
