"""
Post-installation smoke test: `python scripts/smoke_test.py`.

Verifies the pieces a fresh clone needs before the application will
actually work - not a substitute for the real test suite (`pytest`),
but a fast, readable "is my environment set up correctly" check with
a clear pass/fail summary. Ollama being unreachable is reported as a
warning, not a failure - the app itself still starts without it
(agent features degrade gracefully; see README).
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS: list[tuple[str, bool, str]] = []  # (check name, passed, detail)


def check(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((name, passed, detail))
    mark = "PASS" if passed else "FAIL"
    line = f"  [{mark}] {name}"
    if detail:
        line += f" - {detail}"
    print(line)


def check_python_version() -> None:
    major, minor = sys.version_info.major, sys.version_info.minor
    ok = (major, minor) == (3, 12)
    check(
        "Python version",
        ok,
        f"running {major}.{minor} (this project targets 3.12 - see README 'Known Issues')" if not ok else f"{major}.{minor}",
    )


def check_imports() -> None:
    modules = [
        "fastapi", "uvicorn", "sqlalchemy", "pydantic", "jinja2",
        "fitz", "ollama", "dotenv", "pytest",
    ]
    for mod in modules:
        try:
            importlib.import_module(mod)
            check(f"import {mod}", True)
        except ImportError as e:
            check(f"import {mod}", False, str(e))


def check_app_imports() -> None:
    try:
        from app.main import app  # noqa: F401
        check("import app.main (full application)", True)
    except Exception as e:
        check("import app.main (full application)", False, str(e))


def check_config() -> None:
    try:
        from app.core.config import DATABASE_URL, OLLAMA_MODEL, SECRET_KEY
        check("configuration loads", True, f"DATABASE_URL={DATABASE_URL}, OLLAMA_MODEL={OLLAMA_MODEL}")
        if SECRET_KEY == "dev-only-insecure-default-change-in-env":
            print("  [WARN] SECRET_KEY is still the insecure default - fine for local dev, "
                  "set a real one in .env before deploying anywhere shared")
    except Exception as e:
        check("configuration loads", False, str(e))


def check_directories() -> None:
    from app.core.config import DATA_DIR, UPLOAD_DIR, LOG_DIR, ensure_runtime_directories
    ensure_runtime_directories()
    for name, path in [("DATA_DIR", DATA_DIR), ("UPLOAD_DIR", UPLOAD_DIR), ("LOG_DIR", LOG_DIR)]:
        check(f"directory exists: {name}", Path(path).is_dir(), str(path))


def check_database() -> None:
    try:
        from app.database.session import engine, init_db
        from sqlalchemy import text
        init_db()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        check("database connects and initializes", True)
    except Exception as e:
        check("database connects and initializes", False, str(e))


def check_ollama() -> None:
    try:
        import ollama
        from app.core.config import OLLAMA_MODEL
        models = ollama.list()
        names = [m.get("model", m.get("name", "")) for m in models.get("models", [])]
        model_present = any(OLLAMA_MODEL in n for n in names)
        if model_present:
            print(f"  [PASS] Ollama reachable, model '{OLLAMA_MODEL}' is pulled")
        else:
            print(f"  [WARN] Ollama reachable, but model '{OLLAMA_MODEL}' not found - run: ollama pull {OLLAMA_MODEL}")
    except Exception as e:
        print(f"  [WARN] Ollama not reachable ({e}) - agent features will fail until `ollama serve` is running")


def main() -> None:
    print("InvoiceIQ smoke test\n" + "=" * 40)

    print("\nPython & dependencies:")
    check_python_version()
    check_imports()

    print("\nApplication:")
    check_app_imports()
    check_config()

    print("\nRuntime directories:")
    check_directories()

    print("\nDatabase:")
    check_database()

    print("\nLocal LLM (informational - not a pass/fail gate):")
    check_ollama()

    total = len(RESULTS)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = total - passed

    print("\n" + "=" * 40)
    print(f"SUMMARY: {passed}/{total} checks passed")
    if failed:
        print(f"{failed} check(s) FAILED:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name}: {detail}")
        sys.exit(1)
    else:
        print("All checks passed. Run `python run.py` to start the application.")
        sys.exit(0)


if __name__ == "__main__":
    main()
