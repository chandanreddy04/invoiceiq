"""
Centralized configuration - the single place environment variables are
read from. Previously (Phases 0-11) DATABASE_URL and SECRET_KEY were
each read directly via os.getenv() in their own modules, and
OLLAMA_MODEL/UPLOAD_DIR/MAX_UPLOAD_SIZE_MB were documented in
.env.example but never actually wired to any code - a real gap found
while finalizing this repository, not a deliberate design choice.

load_dotenv() is called here, at import time, before anything else
reads an environment variable. Every other module that needs a
setting imports it from here rather than calling os.getenv() itself,
so there is exactly one source of truth and no risk of two modules
disagreeing about a default value.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # populates os.environ from a .env file if one exists; a no-op otherwise

APP_NAME = os.getenv("APP_NAME", "InvoiceIQ")
APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/invoiceiq.db")

# Render (and Heroku-style) managed Postgres hand out connection
# strings starting "postgres://" - the legacy scheme. SQLAlchemy 1.4+
# requires "postgresql://" and raises on the old one. Rewriting it
# here means the exact connection string a cloud provider gives you
# can be pasted into DATABASE_URL verbatim, no manual editing needed.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# The `ollama` Python client reads OLLAMA_HOST from the process
# environment itself when constructing its default client - since
# load_dotenv() above already populated os.environ, no extra wiring
# is needed here for OLLAMA_HOST specifically. OLLAMA_MODEL has no
# such automatic behavior, so it's read explicitly.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3.5")

# Render's free tier has no CPU/RAM budget to run even a 3.8B local model,
# so the cloud deployment needs a different LLM backend than local dev
# uses. Setting GROQ_API_KEY switches every agent (via app/services/
# llm_client.py) from local Ollama to Groq's free-tier hosted API - same
# prompts, same structured-output contract, just a different backend.
# Unset locally, so local dev behavior is completely unchanged.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# llama-3.3-70b-versatile (the original default here) was removed from
# Groq's production model lineup after this was first wired up - every
# request failed regardless of API key validity, which looked
# indistinguishable from "LLM unavailable" until checked against Groq's
# current model list directly. openai/gpt-oss-20b is their current
# free-tier production model - fast, and this app's structured-output
# calls don't need the larger 120b variant's extra capacity.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-insecure-default-change-in-env")

# Communication.status was "draft | sent" with "sent" always simulated -
# see that model's own docstring for why (demonstrating the human-in-
# the-loop approval gate, not real delivery). Setting SMTP_HOST switches
# app/services/email_service.py from the simulated log-only send to a
# real SMTP send, the same config-gated pattern GROQ_API_KEY above uses
# for the LLM backend: unset by default, so nothing about existing
# behavior changes until these are deliberately provided (e.g. a Gmail
# App Password, or any real SMTP provider's credentials) via the
# platform's own env var settings - never pasted into this codebase.
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", SMTP_USER)

# Same config-gated pattern again: unset by default (payment links stay
# "not configured" in the UI), a real Stripe secret key switches
# app/services/payment_service.py to actually creating live Checkout
# Sessions. Get one at https://dashboard.stripe.com/apikeys - test mode
# keys (sk_test_...) work identically to live ones for this integration.
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
# Verifies that a POST to /stripe/webhook genuinely came from Stripe
# (HMAC signature check) rather than trusting the request outright -
# marking an invoice paid is exactly the kind of action that must never
# be spoofable by anyone who can guess the URL. Get this from the
# webhook's own settings page after creating it in the Stripe dashboard
# (Developers -> Webhooks -> Add endpoint -> pointed at /stripe/webhook).
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

# The public origin this app is reachable at (e.g.
# https://invoiceiq-owo3.onrender.com, or http://localhost:8000 for
# local dev) - needed to build an absolute /pay/<token> link inside a
# drafted email body and to tell Stripe where to redirect a customer
# back to after paying. Empty by default; the pay-link line is simply
# omitted from drafted emails until this is set, rather than shipping a
# broken relative link.
APP_BASE_URL = os.getenv("APP_BASE_URL", "")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./data/invoices"))
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))


def ensure_runtime_directories() -> None:
    """Section 56: a fresh clone has no data/, data/invoices/, or logs/
    directories (git doesn't track empty folders) - the app must not
    crash just because nobody committed an empty directory. Called
    once at application startup."""
    for directory in (DATA_DIR, UPLOAD_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
