"""
Financial Analysis Agent - the natural-language assistant from
Section 17. This is where "agentic routing" finally means something
more than a fixed sequence (Phase 5's Orchestrator): the LLM decides
*which filters* answer an open-ended English question. But note what
it does NOT decide:

  - it never writes SQL (Section 17's hard requirement)
  - it never computes the numbers - execute_query() does that in plain
    Python/SQLAlchemy
  - it never phrases the final answer either - format_answer() is
    deterministic, on purpose: restating already-known real numbers is
    not a language-understanding problem, and letting the LLM
    "helpfully" reword a total risks it silently changing a digit
    (Section 34 hallucination risk in a FinTech app is not
    acceptable). The LLM's entire job here is a single, narrow,
    genuinely-hard-without-language-understanding task: turning fuzzy
    English into structured filters.

  INPUT     -> the user's question (English)
  LLM LAYER -> parse_intent(): question -> QueryIntent (structured output)
  TOOL LAYER -> execute_query(): calls the fixed tools in app/tools/invoice_tools.py
  ACTION    -> read-only; this agent never writes to the database
  FEEDBACK  -> if parsing fails, falls back to a summary rather than erroring
"""

import json
import logging

from sqlalchemy.orm import Session

from app.schemas.query_intent import QueryIntent
from app.services.llm_client import LLMUnavailableError, chat
from app.tools import invoice_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You translate a small business owner's question about their invoices into "
    "structured filters. Only use the fields provided. Leave a field null/false if "
    "the question doesn't mention it - do not guess values that weren't asked for.\n\n"
    "wants_summary=true ONLY for a request for totals/aggregate numbers with no list "
    "of individual invoices implied (e.g. 'what do I owe', 'how much am I owed').\n"
    "wants_summary=false whenever the question says 'show', 'list', or 'find' "
    "invoices, even if it also mentions overdue or a dollar amount.\n"
    "party_name: the vendor or customer name mentioned, as free text (e.g. 'golden grain', "
    "not the full formal name) - do not guess a party if none is named.\n"
    "invoice_number: only if a specific invoice number or fragment of one is mentioned.\n"
    "category: only if an expense/spending category is explicitly named (e.g. 'utilities', "
    "'raw ingredients', 'shipping').\n"
    "risky_only=true for words like 'suspicious', 'risky', 'flagged', or 'fraud'.\n\n"
    "Examples:\n"
    "Q: What do I owe?\n"
    "A: {\"wants_summary\": true, \"overdue_only\": false, \"payment_status\": null, \"min_total\": null, \"max_total\": null, \"direction\": null, \"party_name\": null, \"invoice_number\": null, \"category\": null, \"risky_only\": false}\n"
    "Q: Show overdue invoices\n"
    "A: {\"wants_summary\": false, \"overdue_only\": true, \"payment_status\": null, \"min_total\": null, \"max_total\": null, \"direction\": null, \"party_name\": null, \"invoice_number\": null, \"category\": null, \"risky_only\": false}\n"
    "Q: Show unpaid invoices over 500 dollars\n"
    "A: {\"wants_summary\": false, \"overdue_only\": false, \"payment_status\": \"unpaid\", \"min_total\": 500, \"max_total\": null, \"direction\": null, \"party_name\": null, \"invoice_number\": null, \"category\": null, \"risky_only\": false}\n"
    "Q: Show paid outgoing invoices\n"
    "A: {\"wants_summary\": false, \"overdue_only\": false, \"payment_status\": \"paid\", \"min_total\": null, \"max_total\": null, \"direction\": \"outgoing\", \"party_name\": null, \"invoice_number\": null, \"category\": null, \"risky_only\": false}\n"
    "Q: List all Golden Grain invoices\n"
    "A: {\"wants_summary\": false, \"overdue_only\": false, \"payment_status\": null, \"min_total\": null, \"max_total\": null, \"direction\": null, \"party_name\": \"Golden Grain\", \"invoice_number\": null, \"category\": null, \"risky_only\": false}\n"
    "Q: Show suspicious invoices\n"
    "A: {\"wants_summary\": false, \"overdue_only\": false, \"payment_status\": null, \"min_total\": null, \"max_total\": null, \"direction\": null, \"party_name\": null, \"invoice_number\": null, \"category\": null, \"risky_only\": true}\n"
    "Q: Show utilities expenses\n"
    "A: {\"wants_summary\": false, \"overdue_only\": false, \"payment_status\": null, \"min_total\": null, \"max_total\": null, \"direction\": null, \"party_name\": null, \"invoice_number\": null, \"category\": \"utilities\", \"risky_only\": false}"
)


def parse_intent(question: str) -> QueryIntent:
    try:
        content = chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            schema=QueryIntent.model_json_schema(),
        )
        return QueryIntent.model_validate(json.loads(content))
    except LLMUnavailableError as e:
        logger.warning("Intent parsing failed, defaulting to summary: %s", e)
        raise
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Intent parsing failed, defaulting to summary: %s", e)
        raise LLMUnavailableError(str(e)) from e


def format_answer(intent: QueryIntent, results) -> str:
    if intent.wants_summary:
        s = results
        lines = [
            f"You have {s['count_invoices']} invoice(s) total.",
            f"Outstanding payable (you owe): ${s['total_payable_outstanding']:.2f}.",
            f"Outstanding receivable (owed to you): ${s['total_receivable_outstanding']:.2f}.",
        ]
        if s["count_overdue"]:
            lines.append(f"{s['count_overdue']} invoice(s) are overdue, totaling ${s['overdue_total']:.2f}.")
        else:
            lines.append("Nothing is currently overdue.")
        return " ".join(lines)

    invoices = results
    if not invoices:
        return "No invoices matched that."
    lines = [f"Found {len(invoices)} matching invoice(s):"]
    for inv in invoices:
        party = inv.vendor.name if inv.vendor else (inv.customer.name if inv.customer else "unknown")
        lines.append(f"- #{inv.invoice_number} ({party}): ${inv.total} {inv.currency}, due {inv.due_date}, {inv.payment_status.value}")
    return "\n".join(lines)


def answer_question(db: Session, org_id: int, question: str) -> dict:
    try:
        intent = parse_intent(question)
    except LLMUnavailableError:
        intent = QueryIntent(wants_summary=True)  # graceful fallback: at least give something useful

    if intent.wants_summary:
        results = invoice_tools.generate_financial_summary(db, org_id)
    else:
        results = invoice_tools.search_invoices(
            db, org_id,
            direction=intent.direction,
            payment_status=intent.payment_status,
            min_total=intent.min_total,
            max_total=intent.max_total,
            overdue_only=intent.overdue_only,
            party_name=intent.party_name,
            invoice_number=intent.invoice_number,
            category=intent.category,
            risky_only=intent.risky_only,
        )

    return {
        "question": question,
        "intent": intent.model_dump(),
        "answer": format_answer(intent, results),
        "result_count": results["count_invoices"] if intent.wants_summary else len(results),
    }
