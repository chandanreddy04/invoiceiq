"""
Backend entrypoint. One FastAPI process now serves both the JSON API
(app/api/) and the server-rendered HTML pages (app/web/) - no
separate frontend process needed, which is a deliberate simplification
over the earlier Streamlit setup.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

from app.core.config import APP_NAME, APP_ENV, LOG_LEVEL, ensure_runtime_directories
from app.database.session import init_db, SessionLocal
from app.api import customers, vendors, invoices
from app.web import routes as web_routes
from app.web import public_routes
from app.security.deps import NotAuthenticated

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title=APP_NAME, version="1.0.0")


@app.on_event("startup")
def on_startup() -> None:
    ensure_runtime_directories()
    init_db()

    # Auto-seed demo data on boot - idempotent (both functions check
    # for existing data before creating anything), so this is safe to
    # run on every restart. Needed for cloud deployments where nobody
    # has an interactive shell to run scripts/setup.py manually: a
    # fresh deploy should be immediately reviewable, not an empty
    # database with no login.
    try:
        import scripts.seed_demo_data as seed_demo_data
        db = SessionLocal()
        try:
            org = seed_demo_data.seed_org_and_parties(db)
            seed_demo_data.seed_users(db, org)
        finally:
            db.close()
    except Exception:
        logger.exception("Demo data auto-seed failed - app will still start, but may have no login users")

    logger.info("%s started (env=%s)", APP_NAME, APP_ENV)


@app.exception_handler(NotAuthenticated)
def handle_not_authenticated(request: Request, exc: NotAuthenticated):
    return RedirectResponse(f"/web/login?next={exc.next_path}", status_code=303)


app.include_router(customers.router)
app.include_router(vendors.router)
app.include_router(invoices.router)
app.include_router(web_routes.router)
app.include_router(public_routes.router)


@app.get("/health")
def health_check() -> dict:
    """Section 35: reports application, database, and LLM service
    readiness. The LLM being unreachable is reported, not fatal - the
    app itself is still 'up' (invoice CRUD, dashboard, etc. all work
    without it; only agent features degrade, several with documented
    fallback behavior)."""
    from app.database.session import engine
    from sqlalchemy import text

    db_status = "ok"
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unavailable: {e}"

    from app.services.llm_client import is_available as llm_is_available

    llm_status = "ok" if llm_is_available() else "unavailable"

    overall = "ok" if db_status == "ok" else "degraded"
    return {
        "status": overall,
        "service": APP_NAME,
        "environment": APP_ENV,
        "database": db_status,
        "llm": llm_status,
    }


@app.get("/")
def root():
    return RedirectResponse("/web/dashboard")
