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

# The `ollama` Python client reads OLLAMA_HOST from the process
# environment itself when constructing its default client - since
# load_dotenv() above already populated os.environ, no extra wiring
# is needed here for OLLAMA_HOST specifically. OLLAMA_MODEL has no
# such automatic behavior, so it's read explicitly.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3.5")

SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-insecure-default-change-in-env")

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
