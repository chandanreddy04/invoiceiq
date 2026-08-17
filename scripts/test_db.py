"""
Phase 0 sanity check: proves SQLAlchemy can create a SQLite file,
create a table, write a row, and read it back. No app models yet -
that's Phase 1. This script only tests the plumbing.
"""

from sqlalchemy import create_engine, text

DATABASE_URL = "sqlite:///./data/phase0_test.db"

engine = create_engine(DATABASE_URL, echo=False)

with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS greeting (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL
        )
    """))
    conn.execute(text("INSERT INTO greeting (message) VALUES (:msg)"), {"msg": "Hello from SQLite"})
    conn.commit()

    result = conn.execute(text("SELECT id, message FROM greeting"))
    rows = result.fetchall()

    print(f"Database file: {DATABASE_URL}")
    print(f"Rows in 'greeting' table: {len(rows)}")
    for row in rows:
        print(f"  id={row[0]}  message={row[1]}")

print("\nDatabase test PASSED.")
