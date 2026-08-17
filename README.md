# InvoiceIQ — AI-Based FinTech Invoicing with Multi-Agent, Agentic AI

A locally-run invoicing application for a small business (Maple Street Bakery Supply Co., a fictional bakery-supply wholesaler) that demonstrates where LLMs, agents, and agentic AI genuinely add value in a FinTech workflow — and, just as deliberately, where they don't. Every claim in this document is backed by something actually run and checked during development (see [Evaluation Results](#evaluation-results)).

## Table of Contents

- [Overview & Problem Statement](#problem-statement)
- [Features](#features)
- [Architecture](#architecture) (full diagrams in [`docs/architecture.md`](docs/architecture.md))
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Prerequisites & Hardware](#prerequisites--hardware)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Public Cloud Deployment](#public-cloud-deployment)
- [Demo Workflows](#demo-workflows)
- [Testing](#testing)
- [Evaluation Results](#evaluation-results)
- [Security](#security)
- [Troubleshooting](#troubleshooting)
- [Known Issues & Limitations](#known-issues--limitations)
- [Future Work](#future-work)
- [License](#license)

## Problem Statement

Small businesses processing vendor and customer invoices by hand face slow manual data entry, missed due dates, undetected duplicate payments, and no systematic way to spot an unusual or suspicious invoice before it's paid. Generic "add AI" solutions often misuse LLMs for tasks (arithmetic, database queries, deterministic decisions) that plain code already does better, cheaper, and more reliably. This project treats that distinction as a design constraint, not an afterthought — see [`docs/architecture.md`](docs/architecture.md) for the explicit LLM-vs-agent-vs-tool breakdown.

## Features

- Create, upload, view, edit, and delete invoices (incoming/vendor and outgoing/customer)
- PDF upload with automatic field + line-item extraction (local LLM, with a regex fallback if the model is unavailable)
- Automatic fraud/risk scoring with plain-English, signal-by-signal explanations
- Automatic expense categorization
- Automatic duplicate-invoice detection
- Payment prioritization that holds back invoices already flagged as risky
- AI-drafted reminder/follow-up emails (never sent without explicit approval)
- Natural-language financial assistant ("What do I owe?", "Show overdue invoices")
- Unified human-approval queue for every agent-flagged action
- Dashboard, Agent Activity log, and full audit trail
- Login with role-based access control (owner vs. bookkeeper)

## Architecture

```
User (browser)
  → FastAPI (JSON API + server-rendered HTML, one process)
  → Orchestrator (routes to the relevant agent(s), logs every step)
  → Agents (each: deterministic reasoning + ONE narrow LLM call + fixed tools)
  → Tools (plain SQLAlchemy queries - the only way an agent touches the database)
  → Database (SQLite)
  → Human Approval Queue (gates anything risky or irreversible)
```

Full Mermaid diagrams (system flow, ER diagram, approval sequence) are in [`docs/architecture.md`](docs/architecture.md).

### Agents (7, deliberately not more)

| Agent | LLM does | Plain code does |
|---|---|---|
| **Extraction** | Reads unstructured PDF text → structured JSON (invoice fields + line items) | PDF text extraction (PyMuPDF), regex fallback, DB write |
| **Fraud/Risk** | Turns already-computed risk signals into a readable explanation | Computes the actual risk score (amount ratios, vendor age, invoice-number similarity) |
| **Expense Classification** | Maps free-text line-item descriptions to a fixed category taxonomy | Confidence-threshold fallback to "Other" |
| **Payment/AP** | Narrates the recommended payment order in plain English | Priority ranking, holding back risky invoices |
| **Communication** | Writes the actual email text | Decides *which* situation applies (reminder vs. acknowledgement vs. cautious inquiry) — never the LLM's call |
| **Financial Analysis** | Parses an open-ended English question into structured filters | Executes the query, formats the final answer (never lets the LLM restate a dollar figure, to avoid hallucination risk) |
| **Orchestrator** | (No direct LLM call) | Routes tasks, sequences agents, logs every step, dispatches to the approval queue |

Deliberately **not** separate agents: Validation (pure arithmetic), Duplicate Detection (a tool/query), Compliance/Audit (structured logging).

## Technology Stack

| Layer | Choice | Why |
|---|---|---|
| Language | **Python 3.12** (exactly — see [Known Issues](#known-issues--limitations)) | |
| Backend + Frontend | FastAPI + Jinja2 (one process) | Server-rendered HTML, no separate frontend process or heavy JS-framework dependency chain |
| Database | SQLite (SQLAlchemy ORM) | Zero setup; swapping to PostgreSQL is a `DATABASE_URL` change, no code changes |
| LLM | Ollama running **phi3.5** (3.8B, local) | No paid API key required; runs on a CPU-only laptop; ~2.2GB |
| Document processing | PyMuPDF | Pure Python, reads real PDF text layers; no OCR dependency (see Limitations) |
| Auth | stdlib only (`hashlib.scrypt`, `hmac`) | No third-party auth framework needed at this scale |
| Testing | pytest | 72 tests: fast deterministic unit tests + real (non-mocked) LLM integration tests |
| Config | `python-dotenv` + `app/core/config.py` | One centralized settings module; every other module imports from it |
| Containerization (optional) | Docker + docker-compose | Not required — see [Installation](#installation) for the non-Docker path, which is the primary/tested one |

No vector database, no agent framework (LangGraph/CrewAI/AutoGen) — the Orchestrator is plain Python because there's currently one task-routing decision to make; a framework would add a dependency without adding capability at this scope.

## Repository Structure

```
app/
├── main.py               FastAPI entrypoint: logging, startup, health check, auth exception handler
├── core/
│   └── config.py          Centralized settings (loads .env once; everything else imports from here)
├── agents/                 the 7 agents
├── tools/                    fixed, named functions agents are allowed to call - no raw SQL ever
├── services/                  deterministic logic: validation, extraction, invoice CRUD
├── models/                     SQLAlchemy ORM models
├── schemas/                     Pydantic request/response + LLM structured-output schemas
├── database/                     engine/session setup
├── api/                            JSON API routers (no auth - see Security)
├── web/                             server-rendered HTML routes (auth-gated)
├── templates/                        Jinja2 templates
├── security/                          auth (password hashing + signed sessions), RBAC deps
└── utils/                              small shared helpers (e.g. timezone-safe utcnow)
scripts/
├── setup.py                Bootstrap: .env, directories, DB init, demo seed - one command
├── init_db.py                Idempotent DB table creation
├── seed_demo_data.py           Organization, customers, vendors, demo login users
├── generate_synthetic_data.py    ~9 varied invoices exercising every agent
├── generate_sample_invoices.py     3 sample PDFs for manually testing upload
├── smoke_test.py                     Post-install environment verification
└── evaluate_agents.py                 Real measured agent accuracy numbers
tests/                    72 tests (68 fast + 4 real-LLM)
docs/architecture.md      Full Mermaid diagrams + LLM/agent/tool/workflow distinction
data/                      SQLite database, uploaded/generated invoices (gitignored except placeholders)
logs/                       runtime + evaluation logs (gitignored except placeholder)
run.py                    Single-command startup: `python run.py`
Dockerfile, docker-compose.yml   Optional containerized path (see Installation)
.github/workflows/test.yml       CI: fast test suite on every push/PR, no LLM required
```

## Prerequisites & Hardware

- **Python 3.12** (not 3.13/3.14 — see Known Issues)
- **[Ollama](https://ollama.com)** installed
- ~3GB free disk space (Python packages + the phi3.5 model)
- **RAM:** 8GB minimum, 16GB comfortable. No GPU required — phi3.5 runs acceptably on CPU (a few seconds per AI response); it will automatically use a GPU if one is available.
- Docker + Docker Compose, **only** if using the optional containerized path

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd invoiceiq
```

### 2. Create a virtual environment

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env          # macOS/Linux
copy .env.example .env        # Windows
```

Every variable the application actually reads is declared in `.env.example` with a safe local-dev default — see the file for the full, accurate list (`DATABASE_URL`, `OLLAMA_HOST`, `OLLAMA_MODEL`, `SECRET_KEY`, `UPLOAD_DIR`, `MAX_UPLOAD_SIZE_MB`, `LOG_LEVEL`, `APP_NAME`, `APP_ENV`, `DATA_DIR`, `LOG_DIR`). Set a real `SECRET_KEY` before deploying anywhere beyond your own machine:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 5. Install the local LLM

```bash
ollama pull phi3.5
ollama serve
```
(`ollama serve` may already be running in the background/system tray after install — check before starting a duplicate.)

### 6. Initialize everything in one step

```bash
python scripts/setup.py
```
This creates `.env` (if step 4 was skipped), creates runtime directories, initializes the database, and seeds a demo organization + two login users. It prints the demo credentials — **change these before any real deployment.**

### 7. Verify the install

```bash
python scripts/smoke_test.py
```
Should report all checks passed (Ollama shows as a warning, not a failure, if it isn't running yet — start it and re-run).

### 8. Generate demo data (optional but recommended)

```bash
python scripts/generate_synthetic_data.py   # ~9 varied invoices; takes a few minutes (real LLM calls)
python scripts/generate_sample_invoices.py  # 3 sample PDFs for testing the upload page
```

### Docker alternative (optional)

```bash
docker compose up -d
docker compose exec ollama ollama pull phi3.5
docker compose exec app python scripts/setup.py
```
Then visit `http://localhost:8000`. This path has **not been build-tested locally** (Docker isn't installed on the machine this project was developed on) — the Dockerfile/compose file follow standard patterns but treat this as unverified until you confirm it on your machine. The non-Docker path above is the tested, primary installation method.

## Running the Application

```bash
python run.py
```
or equivalently:
```bash
uvicorn app.main:app --reload
```

Visit **http://localhost:8000** — you'll be redirected to log in, then to the Dashboard.

| Role | Email | Password |
|---|---|---|
| Owner (can approve/reject) | `owner@maplestreet.example` | `owner-demo-pass123` |
| Bookkeeper (cannot approve) | `bookkeeper@maplestreet.example` | `bookkeeper-demo-pass123` |

To stop: `Ctrl+C`.

## Public Cloud Deployment

The app can be deployed publicly (a real URL, no `localhost`, cloud-hosted Postgres instead of local SQLite) via the included `render.yaml` Blueprint, using [Render](https://render.com)'s free tier for both the web service and the database.

**Important, stated plainly:** free-tier cloud hosting cannot run Ollama (phi3.5 needs ~6-8GB RAM; free tiers cap around 512MB-1GB). The deployed version runs the full application — invoice CRUD, dashboard, approvals, audit log, RBAC, everything except AI inference — with AI agent features falling back to their already-tested "LLM unavailable" behavior (e.g., extraction falls back to the regex parser; fraud scoring still computes correctly, just without the LLM-written explanation sentence). This is a deliberate, documented tradeoff, not a bug.

**Deploy steps:**
1. Go to the [Render Dashboard](https://dashboard.render.com) → **New +** → **Blueprint**
2. Connect your GitHub account and select this repository (Render will prompt you to authorize access — this step requires your own GitHub login, same as the push itself did)
3. Render reads `render.yaml` automatically and provisions both the web service (built from `Dockerfile`) and a free Postgres database, wiring `DATABASE_URL` between them automatically
4. First boot runs the same startup sequence as local dev: creates tables, then seeds the demo organization/vendors/customers/login users automatically (no manual script execution needed — there's no interactive shell on a fresh cloud deploy)
5. Visit the URL Render assigns (`https://<your-service-name>.onrender.com`) and log in with the same demo credentials as above

**Honestly flagged:** `render.yaml` was written against Render's documented Blueprint spec but has not been build-verified against a live deploy (no Render account was available while building this). If any field name has drifted since, check [render.com/docs/blueprint-spec](https://render.com/docs/blueprint-spec) and adjust — the fix is a one-line YAML edit, not an architecture problem.

## Demo Workflows

All five reproducible end-to-end demonstrations, each of which was actually run and verified during development (not just described):

**1. Normal invoice** — Upload a PDF from `data/invoices/samples/` (or your own) on `/web/upload` → verify the LLM-extracted fields → save. Result: invoice appears `validated`, both Fraud/Risk and Classification agents ran automatically (see Agent Activity).

**2. Duplicate invoice** — Create/upload an invoice with a vendor + invoice number that already exists → the request is rejected with `"Invoice number '...' already exists for this vendor"` before anything is saved.

**3. Suspicious invoice** — Create an invoice for a brand-new vendor with an amount far above your typical invoice size → Fraud/Risk Agent scores it above 70%, forces `pending_review`, and creates an entry in the Approvals queue with a plain-English explanation citing the actual signals.

**4. Natural-language query** — On `/web/assistant`, ask *"Show overdue invoices over $5,000"* or *"What do I owe?"* → the Financial Analysis Agent parses intent, queries the database, and answers with real computed numbers.

**5. Multi-agent workflow** — Open an unpaid invoice's page and click "Draft a reminder" → the Communication Agent drafts a message (tone depends on direction/risk) → it appears in the Approvals queue → an **owner** (not a bookkeeper — RBAC-enforced) approves it → it flips to `sent` in Communications, and the action is recorded in the Audit Log.

## Testing

```bash
pytest                    # all 72 tests (fast + real-LLM), ~70s
pytest -m "not llm"       # just the 68 fast/deterministic ones, ~3-10s
```

GitHub Actions (`.github/workflows/test.yml`) runs the fast suite only — CI does not require a local LLM.

## Evaluation Results

Real numbers from `python scripts/evaluate_agents.py`, run against the actual local phi3.5 model — not fabricated (the script's own docstring documents a case where I caught and fixed a mislabeling bug in the evaluation script itself rather than the agent it was measuring):

| Agent | Metric | Result |
|---|---|---|
| Fraud/Risk Agent | classification accuracy | 86% (6/7 hand-labeled cases) |
| Financial Analysis Agent | exact-match rate (intent parsing) | 50% |
| Financial Analysis Agent | field-level accuracy | 44% |
| Extraction Agent | exact-match rate | 100% (3/3 synthetic invoices) |

**Honest finding:** the local 3.8B model reliably parses single-constraint natural-language questions ("What do I owe?", "Show overdue invoices") but is unreliable on compound, multi-field questions ("unpaid invoices over $500") — it correctly identifies these aren't summary requests but frequently drops the specific filter values. Reported as a measured limitation, not hidden. Full methodology and a fuller academic write-up: [`PROJECT_REPORT.md`](PROJECT_REPORT.md).

## Security

- Passwords: `hashlib.scrypt` (memory-hard, stdlib), never stored or compared in plain text
- Sessions: HMAC-SHA256-signed cookies, tamper-evident (verified by test: a modified user_id or extended expiry fails signature verification)
- RBAC: only the `owner` role can approve/reject agent-flagged actions
- No agent can execute a payment, change bank details, or send an external message without a human clicking approve
- Prompt injection: extracted document text is only ever passed as *data* into a fixed prompt template constrained to a JSON schema — never as instructions
- Tool misuse: agents can only call a fixed, named set of functions (`app/tools/`) — never raw SQL, never arbitrary code
- Upload validation: file type (PDF-only) and size (`MAX_UPLOAD_SIZE_MB`, default 10MB) enforced before any processing; uploaded files are saved under a generated filename (never the client-supplied name), which closes off path traversal entirely rather than trying to sanitize it
- Audit trail: every human-initiated create/update/delete/approve/reject action is recorded in `AuditLog`

**Documented limitation:** the JSON API (`app/api/*`) has no authentication of its own — only the web UI is gated. Tested explicitly (`test_json_api_is_not_gated_by_login`) so this stays visible rather than silently drifting. A real deployment would need API-key or OAuth2 authentication added to those routes.

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip install` hangs on `pydantic-core` / a `.tar.gz` build | Your Python version is too new for prebuilt wheels — use Python 3.12, not 3.13/3.14 |
| `ollama: command not found` | Reopen your terminal after installing Ollama, or check it's on your PATH |
| Agent features fail / pages show "LLM unavailable" | Run `ollama serve`, then `ollama pull phi3.5` if the model isn't present. `python scripts/smoke_test.py` will confirm this specifically |
| `Address already in use` on port 8000 | Another process is using it — `uvicorn app.main:app --port 8001`, or find/kill the other process |
| Redirected to `/web/login` unexpectedly | Your session cookie expired (8 hour lifetime) or you're not logged in yet — log in again |
| `.env` changes don't seem to take effect | Restart the server — `.env` is only read at process startup |
| Fresh clone: "no such table" database error | Run `python scripts/init_db.py` (or `scripts/setup.py`, which includes it) |

## Known Issues & Limitations

- Your system's default Python (3.14 at the time of writing) doesn't yet have prebuilt wheels for some dependencies (`pydantic-core`) — this project targets **Python 3.12** exactly, declared consistently in `pyproject.toml`, `.github/workflows/test.yml`, and this README.
- OCR (scanned-image invoices with no text layer) is not implemented — only PDFs with a real text layer are supported. Deliberate simplification, not an oversight.
- Docker path is written but not build-verified locally (no Docker on the development machine) — flagged explicitly above, not silently assumed to work.
- Single-organization only; multi-tenant support would need `organization_id` scoping added to a few tables (e.g. `ApprovalRequest`) that currently assume one org.
- JSON API has no authentication (web UI does) — see Security.
- `datetime.utcnow()` deprecation was fixed project-wide (`app/utils/time.py`); two cosmetic library-level warnings remain (FastAPI's `on_event`, Starlette's `TemplateResponse` argument order) — functional on current versions, left as-is to avoid unnecessary framework-API churn.

## Future Work

- OCR support for scanned invoices (Tesseract/EasyOCR)
- Multi-tenant organization scoping
- API authentication
- Replace the fixed-sequence Orchestrator with genuine LLM-based dynamic planning once there's more than one task-routing decision to make
- Larger local model (7B+) to reduce the compound-query intent-parsing gap measured above
- Real payment-gateway integration (currently simulated by design — see [`PROJECT_REPORT.md`](PROJECT_REPORT.md) for why)
- Docker path build-verification on a machine with Docker installed

## License

[MIT](LICENSE) for this project's own source code. It depends on separately-licensed third-party software and models **not included in this repository** — see the LICENSE file's "Third-Party Notices" section, in particular regarding the phi3.5 model (downloaded separately via `ollama pull`, not bundled).
