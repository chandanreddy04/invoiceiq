"""
Database connection setup. Everything else in the app imports from
here rather than creating its own connection - one engine, one
source of truth for how we talk to the database.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.core.config import DATABASE_URL

# check_same_thread=False is SQLite-specific: it allows the FastAPI
# dev server (which handles requests on different threads) to reuse
# the same connection safely. Not needed once we move to PostgreSQL.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency: gives each request its own DB session and
    always closes it afterward, even if the request raised an error."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables that don't exist yet. Safe to call every
    startup - it never touches tables that already exist."""
    from app.models import models  # noqa: F401  (import so models register with Base)
    Base.metadata.create_all(bind=engine)
    _add_missing_columns()


def _add_missing_columns() -> None:
    """create_all() only creates tables that don't exist yet - it never
    alters a table that's already there, so a new column added to a
    model needs a matching migration here or it silently never applies
    to a database (local or cloud) that was created before the column
    existed. This project has no migration framework (Alembic would be
    overkill at this scale), so this is a small, explicit, idempotent
    substitute: check what's actually in the database before running
    ALTER TABLE, safe to call on every startup. ADD COLUMN syntax here
    works unchanged on both SQLite and Postgres."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    existing = {col["name"] for col in inspector.get_columns("invoices")}
    if "source_pdf_filename" not in existing:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE invoices ADD COLUMN source_pdf_filename VARCHAR(255)"))
