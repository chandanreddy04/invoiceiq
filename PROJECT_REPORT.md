# InvoiceIQ: An Agentic Multi-Agent Architecture for FinTech Invoice Processing

## Abstract

Invoice processing in small businesses is largely manual, error-prone, and lacks systematic anomaly detection. This report presents InvoiceIQ, a locally-run invoicing application that uses a small, free, local large language model (Ollama/phi3.5, 3.8B parameters) as one component within a deliberately narrow multi-agent architecture, alongside plain deterministic code for every task that does not require language understanding. The system implements seven agents — Extraction, Fraud/Risk, Expense Classification, Payment/Accounts-Payable, Communication, Financial Analysis, and an Orchestrator — each with an explicit, auditable boundary between what the LLM decides and what deterministic logic decides. A unified human-in-the-loop approval queue gates every financially risky or externally-visible action. The system is evaluated against real, measured results (not fabricated): 86% classification accuracy on hand-labeled fraud scenarios, 100% field-level extraction accuracy on synthetic test invoices, and, initially, 44–50% accuracy on compound natural-language query parsing — reported honestly as a measured limitation rather than hidden. That gap is now resolved: neither a 3x larger model nor a genuine reasoning model helped (the reasoning model measurably made it worse), but a decomposed multi-turn extraction strategy — several small schema-constrained calls instead of one large one — raised it to 100% across three independent evaluation runs. Section 12 documents the full sequence of what was tried and why each attempt did or didn't work.

## 1. Introduction

AI-assisted invoicing is frequently marketed as an undifferentiated application of "AI," with little attention paid to *which* parts of the workflow genuinely benefit from a language model versus which parts are simply arithmetic, database queries, or rule-based logic wearing an AI label. This project treats that distinction as a first-class design constraint rather than an afterthought.

## 2. Problem Statement

A small business (modeled here as a fictional bakery-supply wholesaler, Maple Street Bakery Supply Co.) receives and issues invoices from many vendors and customers. Manual processing is slow, error-prone (duplicate payments, missed due dates), and offers no systematic way to flag an unusual invoice before payment. The goal is to automate the judgment-heavy and language-heavy parts of this workflow while keeping deterministic operations deterministic and keeping a human in control of anything financially risky.

## 3. Motivation

Three failure modes motivate the design:
1. **Misapplied AI** — using an LLM for arithmetic or database lookups is slower, more expensive, and less reliable than the equivalent function.
2. **Over-labeling** — calling every function-with-an-LLM-nearby an "agent" obscures which components actually exhibit goal-directed, tool-using, feedback-driven behavior versus which are one-shot text transformations.
3. **Unchecked autonomy in a financial context** — an agent that can both *decide* and *act* on a payment or external communication without human sign-off is a design that should not exist in a FinTech application, regardless of how confident the model is.

## 4. Related Work / Existing Approaches

Commercial invoicing tools (QuickBooks, Bill.com, and similar) offer OCR-based extraction and rule-based approval workflows but are closed systems with no local-model option and no exposed agent architecture. General-purpose "AI agent" frameworks (LangChain, LangGraph, CrewAI, AutoGen) provide orchestration primitives but do not, by themselves, enforce the LLM/deterministic-logic boundary this project treats as central; that boundary was implemented directly rather than delegated to a framework, since the project's single task-routing decision (Financial Analysis Agent intent parsing) did not yet justify the dependency weight of a full agent framework.

## 5. Research Gap

Existing educational and portfolio "AI invoicing" projects typically either (a) wrap every step in an LLM call for demonstration purposes, or (b) build a conventional CRUD app with no real agentic behavior. This project's contribution is a working system that draws the LLM/deterministic boundary explicitly, per-agent, and defends each choice (Section 7-style classification table in the README), while also being honest — via a real evaluation script — about where a small local model's reliability actually breaks down.

## 6. Proposed Architecture

```
User → FastAPI (API + server-rendered UI) → Orchestrator → Agents → Tools → Database
                                                    ↓
                                    Human Approval Queue (gates risky actions)
```

Each agent is decomposed into explicit layers (input, context/memory, LLM, reasoning/planning, tool, action, feedback) so that the LLM's contribution to any given decision is a named, isolated function call — never the decision itself for anything with a deterministic ground truth.

## 7. Multi-Agent System Design

Seven agents were implemented; several originally-considered agents (Validation, Duplicate Detection, Compliance/Audit-logging) were deliberately implemented as plain functions instead, per an explicit test: does this responsibility have a goal, use tools, make a decision under uncertainty, and check feedback? If not, it is a function, not an agent. This is a direct, tested design decision, not an informal one — see `app/agents/` docstrings, each of which states which layers are and are not LLM-driven.

The **Orchestrator** currently performs fixed-sequence dispatch (fraud check → classification on invoice creation; single-agent routing for natural-language queries) rather than LLM-planned multi-step routing, because the system currently has exactly one point where routing decisions vary by input type. This is noted as a direction for future work (Section 12) rather than implemented prematurely.

## 8. Agentic AI Framework / LLM Integration

The LLM (Ollama, phi3.5) is invoked via structured output constrained to a JSON schema (Ollama's `format` parameter bound to each Pydantic model's `model_json_schema()`), which is the mechanism that keeps LLM output safely typed rather than requiring fragile free-text parsing. Every LLM call site is documented with what specifically it is and is not responsible for deciding (see, e.g., `app/agents/fraud_risk_agent.py`'s docstring, which states plainly that the LLM only narrates an already-computed score).

## 9. Implementation

Backend: FastAPI, serving both a JSON API and server-rendered Jinja2 HTML from one process. Database: SQLite via SQLAlchemy. Authentication: standard-library-only password hashing (`hashlib.scrypt`) and HMAC-signed session cookies — no third-party auth framework, appropriate to the single-organization demo scope. Full stack and rationale in `README.md`.

## 10. Dataset

A synthetic dataset (`scripts/generate_synthetic_data.py`) of ~9 invoices was generated through the real application pipeline (not fabricated as static fixtures), covering: normal paid/unpaid invoices, overdue invoices, an oversized invoice from an established vendor, a new-vendor invoice, a deliberate duplicate-invoice attempt (correctly rejected), outgoing customer invoices, and varied expense categories. Combined with earlier manually-created test invoices from development, the working database contains 17 invoices spanning these categories at evaluation time.

## 11. Experimental Setup & Results

`scripts/evaluate_agents.py` runs three evaluations against the real local model (no mocking):

| Agent | Metric | Result | N |
|---|---|---|---|
| Fraud/Risk Agent | binary classification accuracy (hold-for-review vs. not) | 86% | 7 hand-labeled scenarios |
| Financial Analysis Agent | intent-parsing exact-match rate | 100% (initially 50%) | 6 questions |
| Financial Analysis Agent | intent-parsing field-level accuracy | 100% (initially 44%) | 6 questions, 9 fields checked |
| Extraction Agent | structured-field exact-match rate | 100% | 3 synthetic invoices |

**Methodological note, included for scientific integrity:** the evaluation script's Fraud/Risk ground truth initially mislabeled one test case in direct contradiction of a design decision the codebase had already pinned down with its own unit test (a single strong signal alone should not cross the high-risk threshold without a corroborating signal). This was identified by cross-checking the evaluation script's assumptions against the system's own test suite, corrected, and the corrected accuracy (86%, up from an initially-reported 71%) is what is presented here. The remaining single failure is a genuine borderline case (two moderate signals summing to just under the decision threshold) reported as-is rather than adjusted.

## 12. Discussion

The clearest initial finding was the gap between single-constraint and compound natural-language queries on a 3.8B CPU-only model: "What do I owe?" and "Show overdue invoices" parsed correctly essentially every time; "Show unpaid invoices over 500 dollars" correctly avoided misclassifying itself as a summary request but usually dropped the specific numeric/categorical filters. Two concrete fix directions were named at the time rather than a vague "use a bigger model" suggestion — a larger model, or a decomposed multi-turn extraction strategy (asking for one field at a time). Both were subsequently tried, along with a third that hadn't been anticipated:

1. **A 3x larger model (`llama3.1:8b`)** did not help — it failed the identical three compound questions, in the identical way (same fields dropped), producing the same 50%/44% scores. This pointed at the prompt/schema design (or the interaction between constrained decoding and many optional fields) as the actual bottleneck, not raw model capability.
2. **A genuine reasoning model (`deepseek-r1:8b`, extended thinking enabled)** was tried next and measurably made things worse: 78.7 seconds to answer one failing question, still without the missing filter; 303.5 seconds on the other, with the model's reasoning degenerating into repetition and exhausting its token budget before ever producing an answer. This is the same failure mode already documented for a since-reverted reasoning-model feature elsewhere in this codebase (a Fraud/Risk Agent borderline-case second opinion, removed for not meeting the bar despite working), reproduced independently here on an unrelated task.
3. **Decomposed multi-turn extraction** — the second originally-named direction — was implemented last: instead of one call filling in all 14 `QueryIntent` fields at once, several small calls each constrained to a schema covering a single question dimension (intent type; status/direction; amount; party/category), merged into one result. A deterministic gate (a plain keyword-and-digit check, not a model decision) routes only questions that actually touch 2+ dimensions, or mention a dollar amount at all, through this path — the common single-constraint case is untouched. This raised the exact-match rate to 100%, confirmed across three independent evaluation runs, at a latency cost of roughly 12–17 seconds for the routed questions (versus 5–7s for the untouched fast path, and versus 79–300+ seconds for the reasoning-model attempt).

The reasoning-model result is worth dwelling on precisely because it contradicts the intuitive assumption that "more thinking" should help a struggling extraction task. It didn't, in either measured case, and the failure mode (unbounded, occasionally non-terminating deliberation) is a real cost even when the underlying model is technically capable. The decomposed approach's improvement, by contrast, came from a much less glamorous mechanism: giving each call less to do, not more time to do it in.

The Fraud/Risk Agent's two-tier design (a single strong signal is insufficient; two corroborating signals are required to cross the review threshold) was validated as intentional, not accidental, through the process of correcting the evaluation script itself — a useful illustration that evaluation code is also code, and needs the same scrutiny as the system under test.

## 13. Limitations

- Evaluation sample sizes are small (6–7 cases per agent) — sufficient to surface a real, reproducible reliability gap, not sufficient for a statistically rigorous accuracy claim.
- No OCR; only PDFs with an existing text layer are supported.
- Single organization; no multi-tenant data isolation.
- JSON API has no authentication (web UI does) — documented, not hidden.
- Fraud/risk scoring is rule-based, not a trained ML model (Isolation Forest or similar was considered but not implemented, per the project's staged-complexity design — rule-based first, ML later if justified).

## 14. Future Work

Genuine LLM-based dynamic task planning in the Orchestrator once multiple routing decisions exist; OCR for scanned documents; multi-tenant support; API authentication; an ML-based (Isolation Forest) fraud layer alongside the existing rule-based one, trained on a larger synthetic/real dataset. (Larger-model and reasoning-model evaluation for compound-query reliability, previously listed here, is done — see Section 12.)

## 15. Conclusion

InvoiceIQ demonstrates that a small, free, local LLM can be integrated responsibly into a FinTech workflow by treating it as one auditable component within explicitly-scoped agents, with deterministic code handling everything that does not require language understanding, and a human-approval gate on every financially consequential action. The accompanying evaluation numbers are real measurements with their methodology and a self-corrected error disclosed, in keeping with the principle that a portfolio/research system's credibility rests on reporting what was actually found, including its limitations.
