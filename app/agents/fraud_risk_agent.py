"""
The Fraud/Risk Agent - the clearest example in this project of why
"LLM is one component, not the whole agent" (Section 1/5/36).

Originally vendor-only ("only incoming invoices are assessed" was a
hardcoded gate). A real gap found while exploring customer invoicing:
outgoing/customer invoices got NO risk assessment at all, even though
the same underlying pattern is a legitimate real-world concern on that
side too - a brand-new customer suddenly being extended a large amount
of credit (an unpaid outgoing invoice IS credit extended) is exactly as
worth flagging as a brand-new vendor billing an unusually large amount,
just for a different underlying reason (collectability risk, not
"someone is defrauding me"). Vendor and Customer are structurally
identical for this purpose (name, email, address, created_at), so this
whole module now reasons over whichever "party" an invoice actually has,
instead of duplicating vendor-only logic into a parallel customer copy.

Walking through the layers from the Section 5 design:

  INPUT LAYER       -> an Invoice that was just created (either direction)
  CONTEXT/MEMORY     -> this party's (vendor or customer) prior invoices, pulled from the
                        database - and other PARTY RECORDS of the same kind in the same
                        organization too, to catch a pattern no single-party history check
                        can see on its own: two "different" parties secretly sharing an
                        email or address
  REASONING LAYER    -> compute_risk_signals() + score_risk(): PLAIN PYTHON.
                        Ratios, date math, string similarity - no LLM call
                        anywhere in here. This is the part that actually
                        decides how risky the invoice is.
  LLM LAYER           -> explain_risk_with_llm(): turns the already-computed
                        score/reasons into a readable sentence - never
                        decides anything. If this call fails, the agent
                        still works - it just falls back to the raw reasons
                        list (Section 38 failure handling).
                          A second, separate LLM call exists only for
                        genuinely borderline scores (BORDERLINE_BAND):
                        deliberate_on_borderline_case_stream() asks a real
                        reasoning model (extended chain-of-thought, not a
                        one-shot chat model) to weigh the same signals and
                        hand a human approver its deliberation as a second
                        opinion. This is still advisory, not a decision -
                        it can only ever run AFTER risk_score/pending_review
                        are already fixed (nothing it produces is read back
                        into this function) and is never called from here
                        at all: it's on-demand only, from a "Get AI's second
                        opinion" button, because a real measured run against
                        deepseek-r1:8b on a CPU-only laptop took 3-6+
                        minutes - an unacceptable block on saving an
                        invoice for a step that's advisory in the first
                        place. The distinction this project draws isn't
                        "LLM never touches judgment calls" - it's "an LLM's
                        judgment is surfaced to a human, never substituted
                        for one, on anything financially consequential."
  ACTION LAYER        -> run_fraud_check(): writes a FraudFlag row and, if
                        risk is high, forces the invoice into pending_review
                        instead of validated.
  FEEDBACK LAYER      -> the caller (invoice_service) gets the FraudFlag
                        back and can confirm the write succeeded.

Nothing here has a goal it chooses for itself, a persistent memory across
invoices, or genuine multi-step planning - by our own Section 36 test this
is a narrow, single-purpose agent, not a general autonomous one. That's
intentional: it does one job well.
"""

import difflib
import json
import logging
from decimal import Decimal

from sqlalchemy.orm import Session

from app.models.models import Invoice, Vendor, Customer, FraudFlag, InvoiceStatus, ApprovalRequest
from app.utils.time import utcnow_naive
from app.services.llm_client import LLMUnavailableError, chat, reason, reason_stream

logger = logging.getLogger(__name__)

HIGH_RISK_THRESHOLD = 0.7

# A genuinely ambiguous band around the threshold - not "anything below
# 0.7 is fine, anything above isn't." PROJECT_REPORT.md documents a real
# failure mode: two moderate signals summing to just under 0.7 scores as
# "no anomalies" with no nuance, the same as a truly clean invoice. That's
# the actual gap a reasoning model is useful for - not replacing score_risk()
# (still the sole, deterministic decider of risk_score and the pending_review
# gate below), but giving a human approver a second, deliberative read on
# the cases where the fixed threshold is least informative. Symmetric around
# HIGH_RISK_THRESHOLD: a hair under it is exactly as ambiguous as a hair over.
BORDERLINE_BAND = (0.55, 0.85)


def get_party(invoice: Invoice) -> Vendor | Customer | None:
    """The one thing that actually differs between an incoming and an
    outgoing invoice: which relationship to follow. Everything else in
    this module works identically either way once it has this."""
    return invoice.vendor if invoice.vendor_id is not None else invoice.customer


def get_party_history(db: Session, invoice: Invoice, exclude_invoice_id: int | None = None) -> list[Invoice]:
    """Tool: every other invoice this project has to/from the same party
    (whichever kind this invoice actually has)."""
    if invoice.vendor_id is not None:
        query = db.query(Invoice).filter(Invoice.vendor_id == invoice.vendor_id)
    else:
        query = db.query(Invoice).filter(Invoice.customer_id == invoice.customer_id)
    if exclude_invoice_id is not None:
        query = query.filter(Invoice.id != exclude_invoice_id)
    return query.all()


def find_parties_sharing_contact_info(db: Session, org_id: int, party: Vendor | Customer) -> list[Vendor | Customer]:
    """Tool: other records of the SAME kind (vendor-with-vendor,
    customer-with-customer - a vendor never gets compared against a
    customer) in this organization that share this party's email or
    mailing address. A real gap in every signal here - each one only
    ever looks at a single party's own history in isolation, so two
    "different" records secretly sharing contact details (a classic
    impersonation technique for quietly rerouting payments, and just as
    often an innocent duplicate-entry mistake) was invisible until now.
    Comparison is case/whitespace-normalized; blank fields on either
    side never count as a match, so two records both missing an address
    don't falsely "share" one."""
    model = type(party)
    others = db.query(model).filter(model.organization_id == org_id, model.id != party.id).all()
    p_email = (party.email or "").strip().lower()
    p_address = (party.address or "").strip().lower()
    return [
        other for other in others
        if (p_email and p_email == (other.email or "").strip().lower())
        or (p_address and p_address == (other.address or "").strip().lower())
    ]


def compute_risk_signals(db: Session, invoice: Invoice, party: Vendor | Customer) -> dict:
    """Gathers the raw numbers score_risk() will reason over. No verdict yet."""
    history = get_party_history(db, invoice, exclude_invoice_id=invoice.id)
    party_label = "vendor" if invoice.vendor_id is not None else "customer"

    signals = {
        "party_label": party_label,
        "is_new_party": False,
        "party_age_days": None,
        "amount_ratio": None,
        "amount_ratio_basis": None,  # "party" | "org" - what the ratio was computed against
        "max_invoice_number_similarity": 0.0,
        "similar_invoice_number": None,
        "shared_contact_parties": [],  # names of other same-kind parties sharing this one's email/address
    }

    signals["shared_contact_parties"] = [
        p.name for p in find_parties_sharing_contact_info(db, invoice.organization_id, party)
    ]

    if party.created_at:
        signals["party_age_days"] = (utcnow_naive() - party.created_at).days
        signals["is_new_party"] = signals["party_age_days"] < 7

    if history:
        avg_total = sum(Decimal(h.total) for h in history) / len(history)
        if avg_total > 0:
            signals["amount_ratio"] = float(Decimal(invoice.total) / avg_total)
            signals["amount_ratio_basis"] = "party"

        for h in history:
            similarity = difflib.SequenceMatcher(None, invoice.invoice_number, h.invoice_number).ratio()
            if similarity > signals["max_invoice_number_similarity"]:
                signals["max_invoice_number_similarity"] = similarity
                signals["similar_invoice_number"] = h.invoice_number
    else:
        # A brand-new party has no history of its own to compare against -
        # without this fallback, a first invoice of any size scores the same
        # as a first invoice for $10, which misses exactly the "new party +
        # suspiciously large invoice" pattern this agent exists to catch.
        # Compare against this organization's overall average invoice OF
        # THE SAME DIRECTION instead, as the next-best baseline - incoming
        # and outgoing invoices are typically very different sizes (what
        # you pay for supplies vs. what you bill for finished goods), so
        # mixing them would make for a misleading baseline either way.
        # Excludes invoices already flagged as risky - including them would
        # let a known-suspicious invoice quietly inflate the "normal"
        # baseline and dilute the very signal meant to catch the next one.
        same_direction_filter = (
            Invoice.vendor_id.isnot(None) if invoice.vendor_id is not None else Invoice.customer_id.isnot(None)
        )
        org_invoices = (
            db.query(Invoice)
            .filter(
                Invoice.organization_id == invoice.organization_id,
                Invoice.id != invoice.id,
                Invoice.direction == invoice.direction,
                same_direction_filter,
                (Invoice.risk_score.is_(None)) | (Invoice.risk_score < 0.5),
            )
            .all()
        )
        if org_invoices:
            org_avg = sum(Decimal(i.total) for i in org_invoices) / len(org_invoices)
            if org_avg > 0:
                signals["amount_ratio"] = float(Decimal(invoice.total) / org_avg)
                signals["amount_ratio_basis"] = "org"

    return signals


def score_risk(signals: dict) -> tuple[float, list[str]]:
    """The actual decision logic. Deterministic and explainable on purpose -
    see Section 16: rule-based thresholds before any ML model, so every
    score can be traced back to a specific, readable reason."""
    risk = 0.0
    reasons = []
    party_label = signals.get("party_label", "vendor")

    ratio = signals["amount_ratio"]
    basis = f"this {party_label}'s average invoice" if signals["amount_ratio_basis"] == "party" else "this business's typical invoice size"
    if ratio is not None:
        if ratio >= 3:
            risk += 0.45
            reasons.append(f"Amount is {ratio:.1f}x {basis}.")
        elif ratio >= 1.5:
            risk += 0.15
            reasons.append(f"Amount is {ratio:.1f}x {basis} (moderately high).")

    if signals["is_new_party"]:
        risk += 0.30
        reasons.append(f"{party_label.capitalize()} was added only {signals['party_age_days']} day(s) ago.")

    if signals["max_invoice_number_similarity"] >= 0.85:
        risk += 0.30
        reasons.append(
            f"Invoice number closely resembles a previous invoice "
            f"('{signals['similar_invoice_number']}', {signals['max_invoice_number_similarity']:.0%} similar)."
        )

    if signals["shared_contact_parties"]:
        risk += 0.35
        names = ", ".join(signals["shared_contact_parties"])
        reasons.append(
            f"This {party_label} shares an email or address with another {party_label} on file ({names}) - "
            f"could be a duplicate entry, or a look-alike {party_label} set up to redirect payments."
        )

    if not reasons:
        reasons.append(f"No anomalies detected against this {party_label}'s history.")

    return min(risk, 1.0), reasons


def _borderline_review_prompt(risk_score: float, reasons: list[str]) -> str:
    return (
        f"An invoice scored {risk_score:.0%} risk from these deterministic signals:\n"
        + "\n".join(f"- {r}" for r in reasons)
        + f"\n\nThis fell inside a genuinely ambiguous band around our {HIGH_RISK_THRESHOLD:.0%} review "
        "threshold - not a clean pass, not a clear fail. Reason step by step about whether this "
        "particular COMBINATION of signals looks more like innocent coincidence or a real, worth-a-look "
        "problem (consider: do the signals corroborate each other, or are they independent quirks that "
        "just happen to co-occur? is there a mundane explanation for each one on its own?). Do not invent "
        "any facts beyond what's listed above. End with exactly one line: "
        "'RECOMMENDATION: closer look warranted' or 'RECOMMENDATION: standard handling is fine'."
    )


def deliberate_on_borderline_case(risk_score: float, reasons: list[str]) -> dict | None:
    """The one and only place a reasoning model is used in this agent -
    and it is advisory ONLY. Its purpose is narrower and more honest than
    "detect fraud better": give a human approver a second, deliberative
    read specifically on the cases where a fixed numeric threshold is
    least informative (see BORDERLINE_BAND). Returns None (skips the LLM
    call entirely) for scores outside that band - reasoning models are
    slow, and there is no ambiguity to deliberate over on a clear-cut
    score. Returns None on LLM-unavailable too, same fallback discipline
    as explain_risk_with_llm(): this step is a bonus, not a dependency.

    NOT called from run_fraud_check() - a real gap found by actually
    running this against deepseek-r1:8b on a CPU-only laptop rather than
    assuming it would be fast: a single deliberation call measured
    3-6+ minutes, and even num_predict=1536 wasn't always enough for the
    model to reach a concluding line (see reasoning_model_practice.py's
    own measured output). Blocking invoice creation on that for a review
    step that's advisory in the first place would be a real regression,
    not a reasonable tradeoff - so this runs on-demand instead, from the
    "Get AI's second opinion" button (deliberate_on_borderline_case_stream
    below), never automatically. This function itself is kept as the
    non-streaming building block reason_stream() is layered on, and for
    tests/scripts that want the simpler blocking call."""
    if not (BORDERLINE_BAND[0] <= risk_score <= BORDERLINE_BAND[1]):
        return None
    try:
        return reason(messages=[{"role": "user", "content": _borderline_review_prompt(risk_score, reasons)}])
    except LLMUnavailableError as e:
        logger.warning("Borderline-case reasoning call failed, proceeding without a second opinion: %s", e)
        return None


def deliberate_on_borderline_case_stream(risk_score: float, reasons: list[str]):
    """Streaming counterpart to deliberate_on_borderline_case(), and the
    one actually wired to the UI (see the web route's own docstring for
    why: multi-minute latency needs live progress, not a blank page).
    Still gated to BORDERLINE_BAND; still advisory only. Yields nothing
    at all (not even an LLMUnavailableError) for an out-of-band score -
    the caller is expected to check BORDERLINE_BAND itself before
    offering the button in the first place, same as this function does
    before making any LLM call."""
    if not (BORDERLINE_BAND[0] <= risk_score <= BORDERLINE_BAND[1]):
        return
    yield from reason_stream(messages=[{"role": "user", "content": _borderline_review_prompt(risk_score, reasons)}])


def _risk_explanation_prompt(risk_score: float, reasons: list[str]) -> str:
    return (
        f"An invoice was scored {risk_score:.0%} risk based on these factors:\n"
        + "\n".join(f"- {r}" for r in reasons)
        + "\n\nWrite one short, plain-English paragraph explaining this risk assessment "
        "to a small business owner. Do not invent any facts not listed above."
    )


def explain_risk_with_llm(invoice: Invoice, risk_score: float, reasons: list[str]) -> str:
    """The one and only LLM call in this agent. Turns the already-decided
    score + reasons into a natural sentence. If the LLM is unavailable,
    the caller falls back to joining `reasons` directly - the agent's
    verdict does not depend on this call succeeding."""
    prompt = _risk_explanation_prompt(risk_score, reasons)
    try:
        return chat(messages=[{"role": "user", "content": prompt}]).strip()
    except LLMUnavailableError as e:
        logger.warning("Fraud explanation LLM call failed, falling back to raw reasons: %s", e)
        raise


def explain_risk_with_llm_stream(risk_score: float, reasons: list[str]):
    """Same prompt as explain_risk_with_llm(), but yields the explanation
    incrementally - used by the "regenerate live" button on the invoice
    detail page so a person watching doesn't stare at a blank space for
    the 15-30s a real local model call takes. The stored FraudFlag.explanation
    from run_fraud_check() is unaffected either way; this only powers the
    optional live re-generation, it never overwrites the saved verdict."""
    from app.services.llm_client import chat_stream
    yield from chat_stream(messages=[{"role": "user", "content": _risk_explanation_prompt(risk_score, reasons)}])


def run_fraud_check(db: Session, invoice: Invoice) -> FraudFlag | None:
    """The agent's entry point: input -> reasoning -> LLM -> action -> feedback.
    Assesses both directions now - an invoice always has a vendor or a
    customer, so the only real "nothing to assess" case is neither
    being set at all (shouldn't happen in practice, but not this
    agent's job to enforce that - just to not crash on it)."""
    party = get_party(invoice)
    if party is None:
        return None

    signals = compute_risk_signals(db, invoice, party)
    risk_score, reasons = score_risk(signals)  # <-- the only thing that decides risk_score/pending_review, below

    try:
        explanation = explain_risk_with_llm(invoice, risk_score, reasons)
    except LLMUnavailableError:
        explanation = " ".join(reasons)

    # No borderline-case deliberation call here - see
    # deliberate_on_borderline_case()'s docstring for why that runs
    # on-demand (a "Get AI's second opinion" button) instead of blocking
    # invoice creation on a multi-minute reasoning-model call for a
    # review step that's advisory only in the first place.
    flag = FraudFlag(
        invoice_id=invoice.id,
        risk_score=Decimal(str(round(risk_score, 3))),
        reasons_json=json.dumps(reasons),
        explanation=explanation,
    )
    db.add(flag)

    invoice.risk_score = Decimal(str(round(risk_score, 3)))
    if risk_score >= HIGH_RISK_THRESHOLD:
        invoice.invoice_status = InvoiceStatus.pending_review
        db.add(ApprovalRequest(
            type="high_risk_invoice",
            related_id=invoice.id,
            requested_by_agent="fraud_risk_agent",
            reason=explanation,
        ))

    db.commit()
    db.refresh(flag)
    return flag
