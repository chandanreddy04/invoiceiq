"""
Explicit, idempotent database initialization: `python scripts/init_db.py`.

This is a thin wrapper around app.database.session.init_db() (which
is also called automatically on application startup) - it exists as
its own script because a new developer following a README should be
able to initialize the database as a distinct, verifiable step before
starting the server, per this repository's setup walkthrough.

Safe to run any number of times: it only creates tables that don't
already exist and never drops or alters existing data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import ensure_runtime_directories, DATABASE_URL
from app.database.session import init_db


def main() -> None:
    ensure_runtime_directories()
    init_db()
    print("Database initialized successfully.")
    print(f"  Connection: {DATABASE_URL}")


if __name__ == "__main__":
    main()
