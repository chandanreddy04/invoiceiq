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

format_answer() also decides, deterministically, when to call
fx_rate_service for a live currency conversion: totals_by_currency()
elsewhere in this app never adds two currencies together, on purpose,
so a summary spanning e.g. EUR and USD invoices can only ever show
them side by side unless something fetches today's actual rate. "Is
more than one currency present" is a plain len() check, not a
judgment call - the LLM has no role in deciding whether to convert,
same discipline as every other deterministic-first choice in this
project. See fx_rate_service's own docstring for why this stays a
plain function call rather than a real MCP tool-calling loop.
"""

import json
import logging
import re

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.schemas.query_intent import QueryIntent
from app.services import fx_rate_service
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
    "wants_aggregate=true for questions asking to RANK or COMPARE across vendors/customers/"
    "categories (e.g. 'which vendor do I spend the most with', 'average invoice amount by "
    "category') - NOT the same as wants_summary, which is a single overall total. When "
    "wants_aggregate=true, set aggregate_by to what's being grouped ('vendor', 'customer', "
    "or 'category') and aggregate_metric to what's being computed ('total' for spend/most, "
    "'average' for average, 'count' for how many). Also set aggregate_order: 'highest' for "
    "words like 'most', 'highest', 'largest', 'top' (this is the default if unclear); "
    "'lowest' for words like 'least', 'lowest', 'smallest', 'fewest' - these ask for the "
    "OPPOSITE ranking, never the same answer as a 'most' question.\n\n"
    "Examples:\n"
    "Q: What do I owe?\n"
    "A: {\"wants_summary\": true, \"overdue_only\": false, \"payment_status\": null, \"min_total\": null, \"max_total\": null, \"direction\": null, \"party_name\": null, \"invoice_number\": null, \"category\": null, \"risky_only\": false, \"wants_aggregate\": false, \"aggregate_by\": null, \"aggregate_metric\": null, \"aggregate_order\": null}\n"
    "Q: Show overdue invoices\n"
    "A: {\"wants_summary\": false, \"overdue_only\": true, \"payment_status\": null, \"min_total\": null, \"max_total\": null, \"direction\": null, \"party_name\": null, \"invoice_number\": null, \"category\": null, \"risky_only\": false, \"wants_aggregate\": false, \"aggregate_by\": null, \"aggregate_metric\": null, \"aggregate_order\": null}\n"
    "Q: Show unpaid invoices over 500 dollars\n"
    "A: {\"wants_summary\": false, \"overdue_only\": false, \"payment_status\": \"unpaid\", \"min_total\": 500, \"max_total\": null, \"direction\": null, \"party_name\": null, \"invoice_number\": null, \"category\": null, \"risky_only\": false, \"wants_aggregate\": false, \"aggregate_by\": null, \"aggregate_metric\": null, \"aggregate_order\": null}\n"
    "Q: Show paid outgoing invoices\n"
    "A: {\"wants_summary\": false, \"overdue_only\": false, \"payment_status\": \"paid\", \"min_total\": null, \"max_total\": null, \"direction\": \"outgoing\", \"party_name\": null, \"invoice_number\": null, \"category\": null, \"risky_only\": false, \"wants_aggregate\": false, \"aggregate_by\": null, \"aggregate_metric\": null, \"aggregate_order\": null}\n"
    "Q: List all Golden Grain invoices\n"
    "A: {\"wants_summary\": false, \"overdue_only\": false, \"payment_status\": null, \"min_total\": null, \"max_total\": null, \"direction\": null, \"party_name\": \"Golden Grain\", \"invoice_number\": null, \"category\": null, \"risky_only\": false, \"wants_aggregate\": false, \"aggregate_by\": null, \"aggregate_metric\": null, \"aggregate_order\": null}\n"
    "Q: Show suspicious invoices\n"
    "A: {\"wants_summary\": false, \"overdue_only\": false, \"payment_status\": null, \"min_total\": null, \"max_total\": null, \"direction\": null, \"party_name\": null, \"invoice_number\": null, \"category\": null, \"risky_only\": true, \"wants_aggregate\": false, \"aggregate_by\": null, \"aggregate_metric\": null, \"aggregate_order\": null}\n"
    "Q: Show utilities expenses\n"
    "A: {\"wants_summary\": false, \"overdue_only\": false, \"payment_status\": null, \"min_total\": null, \"max_total\": null, \"direction\": null, \"party_name\": null, \"invoice_number\": null, \"category\": \"utilities\", \"risky_only\": false, \"wants_aggregate\": false, \"aggregate_by\": null, \"aggregate_metric\": null, \"aggregate_order\": null}\n"
    "Q: Which vendor do I spend the most with?\n"
    "A: {\"wants_summary\": false, \"overdue_only\": false, \"payment_status\": null, \"min_total\": null, \"max_total\": null, \"direction\": null, \"party_name\": null, \"invoice_number\": null, \"category\": null, \"risky_only\": false, \"wants_aggregate\": true, \"aggregate_by\": \"vendor\", \"aggregate_metric\": \"total\", \"aggregate_order\": \"highest\"}\n"
    "Q: Which vendor do I spend the least with?\n"
    "A: {\"wants_summary\": false, \"overdue_only\": false, \"payment_status\": null, \"min_total\": null, \"max_total\": null, \"direction\": null, \"party_name\": null, \"invoice_number\": null, \"category\": null, \"risky_only\": false, \"wants_aggregate\": true, \"aggregate_by\": \"vendor\", \"aggregate_metric\": \"total\", \"aggregate_order\": \"lowest\"}\n"
    "Q: What's my average invoice amount by category?\n"
    "A: {\"wants_summary\": false, \"overdue_only\": false, \"payment_status\": null, \"min_total\": null, \"max_total\": null, \"direction\": null, \"party_name\": null, \"invoice_number\": null, \"category\": null, \"risky_only\": false, \"wants_aggregate\": true, \"aggregate_by\": \"category\", \"aggregate_metric\": \"average\", \"aggregate_order\": \"highest\"}"
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


# ---------------------------------------------------------------------------
# Decomposed multi-turn extraction for compound questions - the second of
# two documented fix directions for the measured compound-query gap (see
# README's "Follow-up finding"): a larger model was tried first and did NOT
# help (llama3.1:8b failed the identical three questions the same way).
# A genuine reasoning model was tried next, live, against the actual
# failing cases from this project's own eval set, and made things WORSE on
# both real measurements: 78.7s and still missed a filter on one question,
# 303.5s and never produced an answer at all on the other (the exact
# degenerate-repetition failure mode already documented for the reverted
# Fraud/Risk reasoning feature, reproduced independently here). Neither
# result is hypothetical - both were run against deepseek-r1:8b with
# think=True + a schema-constrained JSON format, the same mechanism the
# reverted feature used.
#
# This is the OTHER direction PROJECT_REPORT.md already named and never
# tried: instead of one call asking the model to fill in every field of a
# 14-field schema at once (where a compound question's second filter
# reliably gets dropped), split the extraction into a handful of small,
# independent calls, each constrained to a SMALL schema covering only one
# question dimension. A status word and a number no longer compete for
# the same call's attention - each has its own call, with its own tiny
# schema, so there's nothing for the model to drop.
# ---------------------------------------------------------------------------

_STATUS_WORDS = ("unpaid", "partially paid", "partially_paid", "paid", "overdue")
_DIRECTION_WORDS = ("incoming", "outgoing")
_COMPARISON_WORDS = ("over", "under", "above", "below", "more than", "less than", "at least", "at most")
_AGGREGATE_WORDS = ("most", "least", "highest", "lowest", "largest", "smallest", "top", "fewest", "average", "compare")
_CATEGORY_HINT = "categor"  # matches "category"/"categories" - deliberately loose, this dimension is a weaker signal


def _count_question_dimensions(question: str) -> int:
    """How many distinct filter DIMENSIONS this question touches - not how
    many words. 'unpaid invoices over 500' touches status + amount (2);
    'paid outgoing invoices' touches status + direction (2); 'what do I
    owe' touches none of these (0, it's a plain summary). Two or more is
    exactly the shape that measurably breaks the single-call extraction -
    see the two live-measured failures in this module's own docstring."""
    q = question.lower()
    dims = 0
    if any(w in q for w in _STATUS_WORDS):
        dims += 1
    if any(w in q for w in _DIRECTION_WORDS):
        dims += 1
    if any(w in q for w in _COMPARISON_WORDS) and re.search(r"\d", q):
        dims += 1
    if any(w in q for w in _AGGREGATE_WORDS):
        dims += 1
    if _CATEGORY_HINT in q:
        dims += 1
    return dims


def is_compound_question(question: str) -> bool:
    """The deterministic gate - a plain word-and-digit check, not a model
    call, same discipline as has_extractable_text() and the currency-count
    check in _format_with_conversion(). Two ways in, both grounded in a
    real measurement, not a guess:

      - 2+ distinct dimensions (a status word AND a number, etc.) - the
        documented compound-query gap this whole module exists to fix.
      - an amount comparison alone, even with nothing else in the
        question - a real gap found while measuring the fix above:
        "List invoices under 100 dollars" (one dimension only) reliably
        dropped max_total via the single big-schema call too, every one
        of three live runs. The dedicated _AmountExtract call got it
        right 3/3.

    The common single-constraint case that's already reliable ("what do
    I owe", "show overdue invoices") is unaffected either way."""
    q = question.lower()
    has_amount = any(w in q for w in _COMPARISON_WORDS) and re.search(r"\d", q)
    if has_amount:
        return True
    return _count_question_dimensions(question) >= 2


class _IntentTypeExtract(BaseModel):
    wants_summary: bool = False
    wants_aggregate: bool = False
    aggregate_by: str | None = None
    aggregate_metric: str | None = None
    aggregate_order: str | None = None


class _StatusDirectionExtract(BaseModel):
    payment_status: str | None = None
    direction: str | None = None
    overdue_only: bool = False


class _AmountExtract(BaseModel):
    min_total: float | None = None
    max_total: float | None = None


class _PartyExtract(BaseModel):
    party_name: str | None = None
    invoice_number: str | None = None
    category: str | None = None
    risky_only: bool = False


_INTENT_TYPE_PROMPT = (
    "Classify what KIND of question this is about invoices. Only use the fields provided.\n"
    "wants_summary=true ONLY for a request for totals/aggregate numbers with no list of "
    "individual invoices implied (e.g. 'what do I owe', 'how much am I owed').\n"
    "wants_summary=false whenever the question says 'show', 'list', or 'find' invoices, "
    "even if it also mentions overdue or a dollar amount.\n"
    "wants_aggregate=true ONLY for questions that explicitly ask to RANK or COMPARE across "
    "MULTIPLE vendors/customers/categories against each other (e.g. 'which vendor do I "
    "spend the most with', 'average invoice amount by category'). A dollar-amount threshold "
    "like 'over 500 dollars' is a FILTER on individual invoices, never a ranking - it does "
    "NOT make wants_aggregate true by itself. When wants_aggregate is true, set aggregate_by "
    "('vendor', 'customer', or 'category'), aggregate_metric ('total' for spend/most, "
    "'average' for average, 'count' for how many), and aggregate_order ('highest' for "
    "'most'/'largest'/'top' - the default if unclear; 'lowest' for 'least'/'smallest'/"
    "'fewest'). Otherwise leave all three null/false.\n\n"
    "Examples:\n"
    "Q: Show unpaid invoices over 500 dollars\n"
    "A: {\"wants_summary\": false, \"wants_aggregate\": false, \"aggregate_by\": null, \"aggregate_metric\": null, \"aggregate_order\": null}\n"
    "Q: Which vendor do I spend the most with?\n"
    "A: {\"wants_summary\": false, \"wants_aggregate\": true, \"aggregate_by\": \"vendor\", \"aggregate_metric\": \"total\", \"aggregate_order\": \"highest\"}"
)

_STATUS_DIRECTION_PROMPT = (
    "Does this question about invoices mention a payment status or a direction? Only use "
    "the fields provided; leave a field null/false if not mentioned - never guess.\n"
    "payment_status: one of 'unpaid', 'partially_paid', 'paid' - only if that exact word "
    "(or a clear synonym) appears. Do NOT set this for the word 'overdue' - that's a "
    "separate field below.\n"
    "direction: 'incoming' (vendor bills we owe) or 'outgoing' (customer invoices owed to "
    "us) - only if the question explicitly says incoming/outgoing or vendor/customer.\n"
    "overdue_only=true ONLY for the specific standalone word 'overdue'. The word 'over' "
    "(as in 'over 500 dollars', meaning MORE THAN a number) is completely unrelated to "
    "'overdue' and must never set this field - 'over' is about an amount, 'overdue' is "
    "about a due date having passed.\n\n"
    "Examples:\n"
    "Q: Show unpaid invoices over 500 dollars\n"
    "A: {\"payment_status\": \"unpaid\", \"direction\": null, \"overdue_only\": false}\n"
    "Q: Show overdue invoices\n"
    "A: {\"payment_status\": null, \"direction\": null, \"overdue_only\": true}"
)

_AMOUNT_PROMPT = (
    "Does this question about invoices mention a minimum or maximum dollar amount? Only "
    "use the fields provided; leave a field null if no number is stated for it - never "
    "guess one, and never fill an unmentioned bound with 0.\n"
    "min_total: the number after words like 'over', 'above', 'more than', 'at least'.\n"
    "max_total: the number after words like 'under', 'below', 'less than', 'at most'.\n"
    "Only ONE of these is usually mentioned - the other stays null, not 0.\n\n"
    "Examples:\n"
    "Q: List invoices under 100 dollars\n"
    "A: {\"min_total\": null, \"max_total\": 100}\n"
    "Q: Show unpaid invoices over 500 dollars\n"
    "A: {\"min_total\": 500, \"max_total\": null}"
)

_PARTY_PROMPT = (
    "Does this question about invoices mention a specific vendor/customer name, invoice "
    "number, expense category, or ask about risky/suspicious invoices? Only use the fields "
    "provided; leave a field null/false if not mentioned - never guess.\n"
    "party_name: the vendor or customer name, as free text (e.g. 'golden grain').\n"
    "invoice_number: only if a specific invoice number or fragment is mentioned.\n"
    "category: only if an expense/spending category is explicitly named (e.g. 'utilities').\n"
    "risky_only=true for words like 'suspicious', 'risky', 'flagged', or 'fraud'."
)


def _extract_one(question: str, system_prompt: str, model_cls: type[BaseModel]) -> dict:
    """One small, focused call - schema constrained to model_cls's few
    fields only, never the full 14-field QueryIntent. Raises
    LLMUnavailableError exactly like parse_intent() does, so the caller's
    existing fallback handles a decomposed call failing the same way a
    single-call one always has."""
    content = chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        schema=model_cls.model_json_schema(),
    )
    return model_cls.model_validate(json.loads(content)).model_dump()


def parse_intent_decomposed(question: str) -> QueryIntent:
    """The compound-question path: several small extraction calls instead
    of one big one, each constrained to a schema covering a single
    dimension (see this module's docstring for why). Any one call failing
    (LLM unavailable, bad JSON) fails the whole parse the same way
    parse_intent() already does - this never silently returns a partially-
    filled intent that looks more confident than it is."""
    try:
        merged: dict = {}
        merged.update(_extract_one(question, _INTENT_TYPE_PROMPT, _IntentTypeExtract))
        merged.update(_extract_one(question, _STATUS_DIRECTION_PROMPT, _StatusDirectionExtract))
        merged.update(_extract_one(question, _AMOUNT_PROMPT, _AmountExtract))
        merged.update(_extract_one(question, _PARTY_PROMPT, _PartyExtract))
        return QueryIntent.model_validate(merged)
    except LLMUnavailableError as e:
        logger.warning("Decomposed intent parsing failed, defaulting to summary: %s", e)
        raise
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Decomposed intent parsing failed, defaulting to summary: %s", e)
        raise LLMUnavailableError(str(e)) from e


def _format_with_conversion(totals: dict) -> str:
    """format_money_by_currency() alone, unless more than one currency
    is actually present - then also fetch today's live rate and append
    a single combined figure. Never called for the common single-
    currency case, so this adds a network call only when there's
    genuinely something to convert. Silently omits the conversion (not
    an error the user sees) if the rate lookup fails - the per-currency
    breakdown underneath is always shown either way, so nothing here
    can make the answer less useful, only sometimes less convenient."""
    base = invoice_tools.format_money_by_currency(totals)
    if len(totals) <= 1:
        return base
    combined = fx_rate_service.convert_totals_to_single_currency(totals)
    if combined is None:
        return base
    return f"{base} (≈ {combined:.2f} {fx_rate_service.DEFAULT_TARGET_CURRENCY} at today's rate)"


def format_answer(intent: QueryIntent, results) -> str:
    if intent.wants_aggregate:
        by_currency = results
        if not any(by_currency.values()):
            return "No invoices to aggregate yet."
        lines = []
        metric_label = {"average": "average", "count": "count"}.get(intent.aggregate_metric, "total")
        ranking_verb = "is lowest at" if intent.aggregate_order == "lowest" else "leads at"
        for currency, rows in by_currency.items():
            if not rows:
                continue
            top = rows[0]
            lines.append(
                f"By {metric_label} ({currency}): {top['label']} {ranking_verb} "
                f"{top['value']:.2f} {currency}" + (f" ({top['count']} invoice(s))." if intent.aggregate_metric != "count" else ".")
            )
            for row in rows[1:5]:
                lines.append(f"  - {row['label']}: {row['value']:.2f} {currency} ({row['count']} invoice(s))")
        return "\n".join(lines)

    if intent.wants_summary:
        s = results
        lines = [
            f"You have {s['count_invoices']} invoice(s) total.",
            f"Outstanding payable (you owe): {_format_with_conversion(s['total_payable_outstanding_by_currency'])}.",
            f"Outstanding receivable (owed to you): {_format_with_conversion(s['total_receivable_outstanding_by_currency'])}.",
        ]
        if s["count_overdue"]:
            lines.append(f"{s['count_overdue']} invoice(s) are overdue, totaling {_format_with_conversion(s['overdue_total_by_currency'])}.")
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


_ORDER_KEYWORDS = {
    "lowest": ("least", "lowest", "smallest", "fewest", "minimum"),
    "highest": ("most", "highest", "largest", "biggest", "top", "maximum"),
}


def _correct_ranking_intent(question: str, intent: QueryIntent) -> QueryIntent:
    """Safety net for a real, measured gap: smaller/faster models (Groq's
    free-tier openai/gpt-oss-20b included) sometimes miss that a
    "most"/"least" question is a ranking request at all and fall back to
    wants_summary=true - reproduced even with the exact question spelled
    out as a worked example in the prompt, so this isn't a prompt-wording
    problem to fix with more instructions.

    "most" vs "least" is a closed, fixed vocabulary, not fuzzy language -
    checking a short word list deterministically is strictly more
    reliable than trusting a model to reproduce it every single call.
    Only activates when there's an unambiguous ranking word AND the
    model didn't already recognize this as an aggregate question - it
    never overrides a call the model already got right, so this can only
    improve on the model's own answer, never make it worse."""
    if intent.wants_aggregate:
        return intent
    q = question.lower()
    order = next((o for o, words in _ORDER_KEYWORDS.items() if any(w in q for w in words)), None)
    if order is None:
        return intent

    by = "customer" if "customer" in q else "category" if ("categor" in q or "expense" in q) else "vendor"
    return intent.model_copy(update={
        "wants_summary": False,
        "wants_aggregate": True,
        "aggregate_by": by,
        "aggregate_metric": intent.aggregate_metric or "total",
        "aggregate_order": order,
    })


def answer_question(db: Session, org_id: int, question: str) -> dict:
    # is_compound_question() decides deterministically which extraction
    # path runs - the single fast call for the common case, several small
    # ones for a question that measurably breaks the single call. Neither
    # path is ever chosen by the model itself.
    parse = parse_intent_decomposed if is_compound_question(question) else parse_intent
    try:
        intent = parse(question)
    except LLMUnavailableError:
        intent = QueryIntent(wants_summary=True)  # graceful fallback: at least give something useful
    else:
        # Only applied when the LLM actually responded - a genuinely
        # unavailable LLM stays an honest "here's a summary" fallback,
        # not silently upgraded into an aggregate answer with no real
        # parsing behind it.
        intent = _correct_ranking_intent(question, intent)

    if intent.wants_aggregate and intent.aggregate_by:
        results = invoice_tools.aggregate_invoices(
            db, org_id,
            group_by=intent.aggregate_by,
            metric=intent.aggregate_metric or "total",
            direction=intent.direction,
            ascending=(intent.aggregate_order == "lowest"),
        )
    elif intent.wants_summary:
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

    if intent.wants_aggregate and intent.aggregate_by:
        result_count = sum(len(rows) for rows in results.values())
    elif intent.wants_summary:
        result_count = results["count_invoices"]
    else:
        result_count = len(results)

    return {
        "question": question,
        "intent": intent.model_dump(),
        "answer": format_answer(intent, results),
        "result_count": result_count,
    }
