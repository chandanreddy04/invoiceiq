"""
Agent Evaluation Script (Section 33/34). Runs real agents - including
real calls to the local phi3.5 model, no mocking - against small
hand-labeled test sets and reports actual measured results.

This is NOT the same thing as the pytest suite: pytest asserts
pass/fail on cases already confirmed reliable. This script measures
accuracy/reliability as a NUMBER, including on cases known to be
unreliable (the compound-query intent parsing) - because the honest
percentage is the actual research finding, not something to hide by
only testing the easy cases.

Results will vary somewhat run to run - that variance is itself part
of the finding for a 3.8B CPU-only local model, not something to
average away or rerun until it looks better.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz

from app.agents.fraud_risk_agent import score_risk, HIGH_RISK_THRESHOLD
from app.agents.financial_analysis_agent import parse_intent, parse_intent_decomposed, is_compound_question
from app.services.extraction_service import extract_text_from_pdf
from app.services.llm_extraction_service import extract_invoice_with_llm


# ---------------------------------------------------------------------------
# Fraud/Risk Agent: deterministic scoring logic, no LLM involved (the LLM
# only narrates the score - see the agent's docstring). Evaluated as a
# binary classifier: should this invoice be held for review or not.
# ---------------------------------------------------------------------------

FRAUD_CASES = [
    {"name": "clean invoice, established vendor", "expected_flag": False,
     "signals": {"is_new_vendor": False, "vendor_age_days": 400, "amount_ratio": 1.1, "amount_ratio_basis": "vendor",
                 "max_invoice_number_similarity": 0.2, "similar_invoice_number": None}},
    {"name": "moderately high amount only", "expected_flag": False,
     "signals": {"is_new_vendor": False, "vendor_age_days": 400, "amount_ratio": 2.0, "amount_ratio_basis": "vendor",
                 "max_invoice_number_similarity": 0.1, "similar_invoice_number": None}},
    {"name": "new vendor only, normal amount", "expected_flag": False,
     "signals": {"is_new_vendor": True, "vendor_age_days": 2, "amount_ratio": 1.0, "amount_ratio_basis": "vendor",
                 "max_invoice_number_similarity": 0.1, "similar_invoice_number": None}},
    {"name": "extreme amount + new vendor", "expected_flag": True,
     "signals": {"is_new_vendor": True, "vendor_age_days": 1, "amount_ratio": 12.0, "amount_ratio_basis": "org",
                 "max_invoice_number_similarity": 0.1, "similar_invoice_number": None}},
    # Genuine borderline case, kept as a real (not fabricated) miss: two
    # moderate signals (new-vendor 0.30 + similarity 0.30) sum to 0.60,
    # just under the 0.7 threshold. Whether that SHOULD cross is a
    # legitimate threshold-tuning question, not a labeling mistake - left
    # as-is rather than adjusted to make the accuracy number look better.
    {"name": "invoice number near-duplicate + new vendor (borderline)", "expected_flag": True,
     "signals": {"is_new_vendor": True, "vendor_age_days": 0, "amount_ratio": None, "amount_ratio_basis": None,
                 "max_invoice_number_similarity": 0.92, "similar_invoice_number": "GGM-0847"}},
    # NOT flagged, by design: a single strong signal alone (even 9.3x) is
    # deliberately not enough to cross HIGH_RISK_THRESHOLD without a second
    # corroborating signal - see test_high_amount_ratio_crosses_high_risk_
    # threshold_alone_is_not_enough in tests/test_fraud_risk_agent.py, which
    # pins down this exact behavior. An earlier version of this eval script
    # mislabeled this case as expected_flag=True, contradicting the agent's
    # own documented design - caught and fixed while reviewing eval results,
    # not silently "corrected" to make the accuracy number look better.
    {"name": "extreme amount alone, established vendor (no corroborating signal)", "expected_flag": False,
     "signals": {"is_new_vendor": False, "vendor_age_days": 400, "amount_ratio": 9.3, "amount_ratio_basis": "vendor",
                 "max_invoice_number_similarity": 0.1, "similar_invoice_number": None}},
    {"name": "all three signals combined", "expected_flag": True,
     "signals": {"is_new_vendor": True, "vendor_age_days": 1, "amount_ratio": 4.0, "amount_ratio_basis": "vendor",
                 "max_invoice_number_similarity": 0.9, "similar_invoice_number": "ABC-1"}},
]


def evaluate_fraud_agent():
    results = []
    for case in FRAUD_CASES:
        score, reasons = score_risk(case["signals"])
        predicted_flag = score >= HIGH_RISK_THRESHOLD
        results.append({
            "name": case["name"], "score": round(score, 3),
            "predicted_flag": predicted_flag, "expected_flag": case["expected_flag"],
            "correct": predicted_flag == case["expected_flag"],
        })
    accuracy = sum(r["correct"] for r in results) / len(results)
    return results, accuracy


# ---------------------------------------------------------------------------
# Financial Analysis Agent: intent parsing IS an LLM task, evaluated with
# real inference. Cases 1-2 (single-constraint) were confirmed reliable in
# Phase 6 manual testing. Cases 3-5 were confirmed UNRELIABLE via the single
# big-schema call (50% exact-match / 44% field-level - see README's
# "Follow-up finding"); a larger model (llama3.1:8b) did not help, and a
# genuine reasoning model made it WORSE (measured live: 78.7s and still
# wrong on one case, 303.5s and no answer at all on another - the same
# degenerate-repetition failure already documented for the reverted
# Fraud/Risk reasoning feature). is_compound_question() now routes these
# through parse_intent_decomposed() - several small schema-constrained
# calls, one per question dimension, instead of one call asking for all 14
# fields at once - which measured 6/6 exact-match across two independent
# runs. All cases are still included here, unmodified, so this script
# keeps measuring the real, current, production routing behavior - not a
# best-case subset.
# ---------------------------------------------------------------------------

INTENT_CASES = [
    {"question": "What do I owe?", "expected": {"wants_summary": True}},
    {"question": "How much am I owed?", "expected": {"wants_summary": True}},
    {"question": "Show overdue invoices", "expected": {"overdue_only": True, "wants_summary": False}},
    {"question": "Show paid outgoing invoices", "expected": {"payment_status": "paid", "direction": "outgoing"}},
    # overdue_only and wants_aggregate are asserted False here on purpose,
    # not just the fields that should be set - a real bug found live while
    # measuring the decomposed-extraction fix: the isolated intent-type
    # call mistook "over 500" for a ranking question, and the isolated
    # status/direction call mistook "over" for "overdue". Both fixed with
    # explicit negative examples in their prompts; these two extra checks
    # are what would have caught the regression if this eval script had
    # been run as part of fixing it, so they stay as real assertions
    # rather than being narrowed back down once the fix looked done.
    {"question": "Show unpaid invoices over 500 dollars", "expected": {
        "payment_status": "unpaid", "min_total": 500, "overdue_only": False, "wants_aggregate": False,
    }},
    {"question": "List invoices under 100 dollars", "expected": {"max_total": 100}},
]


def evaluate_financial_analysis_agent():
    results = []
    for case in INTENT_CASES:
        question = case["question"]
        parse = parse_intent_decomposed if is_compound_question(question) else parse_intent
        intent = parse(question).model_dump()
        field_matches = {k: (intent.get(k) == v) for k, v in case["expected"].items()}
        all_correct = all(field_matches.values())
        results.append({
            "question": question, "compound": is_compound_question(question), "expected": case["expected"], "got": intent,
            "field_matches": field_matches, "all_fields_correct": all_correct,
        })
    exact_match_rate = sum(r["all_fields_correct"] for r in results) / len(results)
    all_field_checks = [v for r in results for v in r["field_matches"].values()]
    field_level_accuracy = sum(all_field_checks) / len(all_field_checks)
    return results, exact_match_rate, field_level_accuracy


# ---------------------------------------------------------------------------
# Extraction Agent: structured extraction from PDF text, real inference.
# Three synthetic invoices with known ground truth, varying line-item counts.
# ---------------------------------------------------------------------------

EXTRACTION_CASES = [
    {
        "text": "INVOICE\nInvoice Number: EVAL-001\nInvoice Date: 2026-03-01\nDue Date: 2026-03-31\n"
                "Widget A   Qty 5   Unit Price 10.00\nWidget B   Qty 2   Unit Price 25.00\nTotal Due: $100.00",
        "expected": {"invoice_number": "EVAL-001", "invoice_date": "2026-03-01", "due_date": "2026-03-31", "line_item_count": 2},
    },
    {
        "text": "INVOICE\nInvoice Number: EVAL-002\nInvoice Date: 2026-04-15\nDue Date: 2026-05-15\n"
                "Consulting hours   Qty 10   Unit Price 75.00\nTotal Due: $750.00",
        "expected": {"invoice_number": "EVAL-002", "invoice_date": "2026-04-15", "due_date": "2026-05-15", "line_item_count": 1},
    },
    {
        "text": "INVOICE\nInvoice Number: EVAL-003\nInvoice Date: 2026-06-10\nDue Date: 2026-07-10\n"
                "Paper (case)   Qty 20   Unit Price 8.50\nToner cartridge   Qty 4   Unit Price 45.00\n"
                "Delivery fee   Qty 1   Unit Price 15.00\nTotal Due: $370.00",
        "expected": {"invoice_number": "EVAL-003", "invoice_date": "2026-06-10", "due_date": "2026-07-10", "line_item_count": 3},
    },
]


def evaluate_extraction_agent():
    results = []
    for case in EXTRACTION_CASES:
        result = extract_invoice_with_llm(case["text"])
        exp = case["expected"]
        checks = {
            "invoice_number": result.invoice_number == exp["invoice_number"],
            "invoice_date": result.invoice_date == exp["invoice_date"],
            "due_date": result.due_date == exp["due_date"],
            "line_item_count": len(result.line_items) == exp["line_item_count"],
        }
        results.append({"expected": exp, "got": result.model_dump(), "checks": checks, "all_correct": all(checks.values())})
    all_checks = [v for r in results for v in r["checks"].values()]
    field_level_accuracy = sum(all_checks) / len(all_checks)
    exact_match_rate = sum(r["all_correct"] for r in results) / len(results)
    return results, exact_match_rate, field_level_accuracy


def main():
    from app.services.llm_client import MODEL_NAME

    print("=" * 70)
    print(f"AGENT EVALUATION — real measured results, model: {MODEL_NAME}")
    print("=" * 70)

    start = time.time()

    print("\n[1/3] Fraud/Risk Agent (deterministic scoring, no LLM)")
    fraud_results, fraud_accuracy = evaluate_fraud_agent()
    for r in fraud_results:
        mark = "PASS" if r["correct"] else "FAIL"
        print(f"  [{mark}] {r['name']}: score={r['score']} predicted_flag={r['predicted_flag']} expected={r['expected_flag']}")
    print(f"  -> Accuracy: {fraud_accuracy:.0%} ({sum(r['correct'] for r in fraud_results)}/{len(fraud_results)})")

    print("\n[2/3] Financial Analysis Agent - intent parsing (real LLM calls)")
    intent_results, intent_exact, intent_field_acc = evaluate_financial_analysis_agent()
    for r in intent_results:
        mark = "PASS" if r["all_fields_correct"] else "FAIL"
        print(f"  [{mark}] \"{r['question']}\"")
        for field, matched in r["field_matches"].items():
            print(f"         {field}: expected={r['expected'][field]!r} got={r['got'].get(field)!r} {'OK' if matched else 'MISMATCH'}")
    print(f"  -> Exact-match rate: {intent_exact:.0%}  |  Field-level accuracy: {intent_field_acc:.0%}")

    print("\n[3/3] Extraction Agent - structured field extraction (real LLM calls)")
    extraction_results, extraction_exact, extraction_field_acc = evaluate_extraction_agent()
    for i, r in enumerate(extraction_results):
        mark = "PASS" if r["all_correct"] else "FAIL"
        print(f"  [{mark}] case {i+1}: {r['checks']}")
    print(f"  -> Exact-match rate: {extraction_exact:.0%}  |  Field-level accuracy: {extraction_field_acc:.0%}")

    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Agent':<35}{'Metric':<25}{'Result'}")
    print(f"{'Fraud/Risk Agent':<35}{'classification accuracy':<25}{fraud_accuracy:.0%}")
    print(f"{'Financial Analysis Agent':<35}{'exact-match rate':<25}{intent_exact:.0%}")
    print(f"{'Financial Analysis Agent':<35}{'field-level accuracy':<25}{intent_field_acc:.0%}")
    print(f"{'Extraction Agent':<35}{'exact-match rate':<25}{extraction_exact:.0%}")
    print(f"{'Extraction Agent':<35}{'field-level accuracy':<25}{extraction_field_acc:.0%}")
    print(f"\nTotal evaluation time: {elapsed:.1f}s")

    out_path = Path(__file__).resolve().parent.parent / "logs" / "evaluation_results.json"
    out_path.write_text(json.dumps({
        "fraud_agent": {"accuracy": fraud_accuracy, "results": fraud_results},
        "financial_analysis_agent": {"exact_match_rate": intent_exact, "field_level_accuracy": intent_field_acc, "results": intent_results},
        "extraction_agent": {"exact_match_rate": extraction_exact, "field_level_accuracy": extraction_field_acc, "results": extraction_results},
        "elapsed_seconds": elapsed,
    }, indent=2, default=str))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
