"""
One-command bootstrap: `python scripts/setup.py`.

Does everything needed to go from "dependencies installed" to
"database ready and demo login created" in one step: creates .env
from .env.example if missing, creates runtime directories, initializes
the database, and seeds demo organization/users. Does NOT pull the
Ollama model or generate synthetic invoices (both can take several
minutes and need Ollama already running) - those stay separate,
documented steps so this script stays fast and this step's failures
are easy to isolate from the AI-dependent steps.

Every step prints a clear PASS/FAIL - it does not hide failures
behind a generic traceback.
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def step(description: str):
    print(f"\n-> {description}")


def main() -> None:
    print("InvoiceIQ setup\n" + "=" * 40)

    step("Checking Python version")
    major, minor = sys.version_info.major, sys.version_info.minor
    if (major, minor) != (3, 12):
        print(f"   [WARN] Running Python {major}.{minor} - this project targets 3.12 (see README 'Known Issues')")
    else:
        print(f"   [OK] Python {major}.{minor}")

    step("Checking for .env file")
    env_path = ROOT / ".env"
    example_path = ROOT / ".env.example"
    if env_path.exists():
        print("   [OK] .env already exists - leaving it untouched")
    elif example_path.exists():
        shutil.copy(example_path, env_path)
        print("   [OK] Created .env from .env.example (using safe local-dev defaults)")
    else:
        print("   [FAIL] .env.example not found - cannot continue")
        sys.exit(1)

    try:
        step("Creating runtime directories")
        from app.core.config import ensure_runtime_directories, DATA_DIR, UPLOAD_DIR, LOG_DIR
        ensure_runtime_directories()
        print(f"   [OK] {DATA_DIR}, {UPLOAD_DIR}, {LOG_DIR}")

        step("Initializing database")
        from app.database.session import init_db
        init_db()
        print("   [OK] Tables created (or already existed)")

        step("Seeding demo organization, vendors, customers, and login users")
        import scripts.seed_demo_data as seed_demo_data
        seed_demo_data.main()

    except Exception as e:
        print(f"\n[FAIL] Setup failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 40)
    print("Setup complete. Next steps:")
    print("  1. ollama pull phi3.5          (if not already done)")
    print("  2. ollama serve                 (if not already running)")
    print("  3. python scripts/smoke_test.py (verify everything)")
    print("  4. python run.py                (start the application)")


if __name__ == "__main__":
    main()
