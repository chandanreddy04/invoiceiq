"""
The Fraud/Risk Agent - the clearest example in this project of why
"LLM is one component, not the whole agent" (Section 1/5/36).

Walking through the layers from the Section 5 design:

  INPUT LAYER       -> an Invoice that was just created
  CONTEXT/MEMORY     -> this vendor's prior invoices, pulled from the database
  REASONING LAYER    -> compute_risk_signals() + score_risk(): PLAIN PYTHON.
                        Ratios, date math, string similarity - no LLM call
                        anywhere in here. This is the part that actually
                        decides how risky the invoice is.
  LLM LAYER           -> explain_risk_with_llm(): the ONLY place an LLM is
                        used, and only to turn already-computed numbers into
                        a readable sentence. If this call fails, the agent
                        still works - it just falls back to the raw reasons
                        list (Section 38 failure handling).
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

from app.models.models import Invoice, Vendor, FraudFlag, InvoiceStatus, ApprovalRequest
from app.utils.time import utcnow_naive
from app.services.llm_client import LLMUnavailableError, chat

logger = logging.getLogger(__name__)

HIGH_RISK_THRESHOLD = 0.7


def get_vendor_history(db: Session, vendor_id: int, exclude_invoice_id: int | None = None) -> list[Invoice]:
    """Tool: every other invoice this project has ever received from this vendor."""
    query = db.query(Invoice).filter(Invoice.vendor_id == vendor_id)
    if exclude_invoice_id is not None:
        query = query.filter(Invoice.id != exclude_invoice_id)
    return query.all()


def compute_risk_signals(db: Session, invoice: Invoice, vendor: Vendor) -> dict:
    """Gathers the raw numbers score_risk() will reason over. No verdict yet."""
    history = get_vendor_history(db, vendor.id, exclude_invoice_id=invoice.id)

    signals = {
        "is_new_vendor": False,
        "vendor_age_days": None,
        "amount_ratio": None,
        "amount_ratio_basis": None,  # "vendor" | "org" - what the ratio was computed against
        "max_invoice_number_similarity": 0.0,
        "similar_invoice_number": None,
    }

    if vendor.created_at:
        signals["vendor_age_days"] = (utcnow_naive() - vendor.created_at).days
        signals["is_new_vendor"] = signals["vendor_age_days"] < 7

    if history:
        avg_total = sum(Decimal(h.total) for h in history) / len(history)
        if avg_total > 0:
            signals["amount_ratio"] = float(Decimal(invoice.total) / avg_total)
            signals["amount_ratio_basis"] = "vendor"

        for h in history:
            similarity = difflib.SequenceMatcher(None, invoice.invoice_number, h.invoice_number).ratio()
            if similarity > signals["max_invoice_number_similarity"]:
                signals["max_invoice_number_similarity"] = similarity
                signals["similar_invoice_number"] = h.invoice_number
    else:
        # A brand-new vendor has no history of its own to compare against -
        # without this fallback, a first invoice of any size scores the same
        # as a first invoice for $10, which misses exactly the "new vendor +
        # suspiciously large invoice" pattern this agent exists to catch.
        # Compare against this organization's overall average invoice
        # instead, as the next-best baseline for "is this unusually large."
        # Excludes invoices already flagged as risky - including them would
        # let a known-suspicious invoice quietly inflate the "normal"
        # baseline and dilute the very signal meant to catch the next one.
        org_invoices = (
            db.query(Invoice)
            .filter(
                Invoice.organization_id == invoice.organization_id,
                Invoice.id != invoice.id,
                Invoice.vendor_id.isnot(None),
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

    ratio = signals["amount_ratio"]
    basis = "this vendor's average invoice" if signals["amount_ratio_basis"] == "vendor" else "this business's typical invoice size"
    if ratio is not None:
        if ratio >= 3:
            risk += 0.45
            reasons.append(f"Amount is {ratio:.1f}x {basis}.")
        elif ratio >= 1.5:
            risk += 0.15
            reasons.append(f"Amount is {ratio:.1f}x {basis} (moderately high).")

    if signals["is_new_vendor"]:
        risk += 0.30
        reasons.append(f"Vendor was added only {signals['vendor_age_days']} day(s) ago.")

    if signals["max_invoice_number_similarity"] >= 0.85:
        risk += 0.30
        reasons.append(
            f"Invoice number closely resembles a previous invoice "
            f"('{signals['similar_invoice_number']}', {signals['max_invoice_number_similarity']:.0%} similar)."
        )

    if not reasons:
        reasons.append("No anomalies detected against this vendor's history.")

    return min(risk, 1.0), reasons


def explain_risk_with_llm(invoice: Invoice, risk_score: float, reasons: list[str]) -> str:
    """The one and only LLM call in this agent. Turns the already-decided
    score + reasons into a natural sentence. If the LLM is unavailable,
    the caller falls back to joining `reasons` directly - the agent's
    verdict does not depend on this call succeeding."""
    prompt = (
        f"An invoice was scored {risk_score:.0%} risk based on these factors:\n"
        + "\n".join(f"- {r}" for r in reasons)
        + "\n\nWrite one short, plain-English paragraph explaining this risk assessment "
        "to a small business owner. Do not invent any facts not listed above."
    )
    try:
        return chat(messages=[{"role": "user", "content": prompt}]).strip()
    except LLMUnavailableError as e:
        logger.warning("Fraud explanation LLM call failed, falling back to raw reasons: %s", e)
        raise


def run_fraud_check(db: Session, invoice: Invoice) -> FraudFlag | None:
    """The agent's entry point: input -> reasoning -> LLM -> action -> feedback."""
    if invoice.vendor_id is None:
        return None  # only incoming (vendor) invoices are assessed for now

    vendor = db.get(Vendor, invoice.vendor_id)
    signals = compute_risk_signals(db, invoice, vendor)
    risk_score, reasons = score_risk(signals)

    try:
        explanation = explain_risk_with_llm(invoice, risk_score, reasons)
    except LLMUnavailableError:
        explanation = " ".join(reasons)

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
