"""
Real browser end-to-end tests - the one thing the rest of the suite
(TestClient + mocked ollama.chat) never does: click through the actual
rendered HTML the way a person would, in a real Chromium browser. Every
other test in this repo can pass while a template is still broken (a
missing form field, a JS error, a button that doesn't submit) - these
tests are the check for that gap.

Requires `pip install -r requirements-dev.txt` then
`python -m playwright install chromium`; skipped automatically
(pytest.importorskip) if playwright isn't installed, so the rest of the
suite is unaffected. Marked @pytest.mark.e2e and excluded from the
default fast run and CI - see pytest.ini / .github/workflows/test.yml.

Spins up a real `uvicorn` subprocess against a throwaway SQLite file
(never the real dev database) with OLLAMA_MODEL pointed at a
nonexistent model, so every agent call takes the fast, deterministic
LLM-unavailable fallback path instead of a real 15-30s inference call.
"""

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("playwright")

E2E_PORT = 8791
BASE_URL = f"http://127.0.0.1:{E2E_PORT}"
REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DB_PATH = REPO_ROOT / "data" / "test_e2e.db"

pytestmark = pytest.mark.e2e


def _wait_for_server(url: str, timeout: float = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


@pytest.fixture(scope="module")
def live_server():
    TEST_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()

    env = os.environ.copy()
    env["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH.as_posix()}"
    env["OLLAMA_MODEL"] = "e2e-test-nonexistent-model"  # forces the fast fallback path, not a real 15-30s call
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(E2E_PORT)],
        env=env, cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        if not _wait_for_server(BASE_URL):
            proc.terminate()
            pytest.fail("e2e server did not start within 30s")
        yield BASE_URL
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        # Windows can hold the SQLite file handle open for a moment after
        # the child process has already exited - a plain unlink() here
        # flakes with PermissionError often enough to be a real, observed
        # bug (found by actually running this suite), not hypothetical.
        # The next run's setup already re-deletes this file unconditionally,
        # so a failed cleanup here is harmless - retry briefly, then give up
        # quietly rather than fail the whole test run over leftover test data.
        for attempt in range(5):
            try:
                if TEST_DB_PATH.exists():
                    TEST_DB_PATH.unlink()
                break
            except PermissionError:
                if attempt == 4:
                    break
                time.sleep(0.5)


def _login(page, base_url):
    page.goto(base_url)
    page.fill('input[name="email"]', "owner@maplestreet.example")
    page.fill('input[name="password"]', "owner-demo-pass123")
    page.click('button[type="submit"]')
    page.wait_for_url(f"{base_url}/web/dashboard")


def test_login_page_loads_with_real_browser(live_server, page):
    page.goto(live_server)
    assert "InvoiceIQ" in page.title()
    assert page.locator('input[name="email"]').is_visible()
    assert page.locator('input[name="password"]').is_visible()


def test_wrong_password_shows_error_and_does_not_log_in(live_server, page):
    page.goto(live_server)
    page.fill('input[name="email"]', "owner@maplestreet.example")
    page.fill('input[name="password"]', "wrong-password")
    page.click('button[type="submit"]')
    assert page.locator("text=Invalid email or password").is_visible()
    assert "/web/login" in page.url


def test_correct_login_reaches_dashboard(live_server, page):
    _login(page, live_server)
    assert page.locator("text=Dashboard").first.is_visible()
    assert page.locator("text=Total invoices").is_visible()


def test_create_invoice_via_real_rendered_form(live_server, page):
    """The golden path this suite otherwise never actually clicks through:
    fill the real <form> on /web/invoices/new, submit it, and confirm the
    new invoice shows up - proving the template's field names actually
    match what the route expects, not just that the route works when
    called directly."""
    _login(page, live_server)

    page.goto(f"{live_server}/web/invoices/new")
    page.select_option('select[name="direction"]', "incoming")
    page.fill('input[name="invoice_number"]', "E2E-BROWSER-1")
    page.select_option('select[name="vendor_id"]', label="Golden Grain Milling")
    page.fill('input[name="invoice_date"]', "2026-01-01")
    page.fill('input[name="due_date"]', "2026-01-31")
    page.fill('input[name="item_description_0"]', "Browser test widget")
    page.fill('input[name="item_quantity_0"]', "1")
    page.fill('input[name="item_unit_price_0"]', "42")
    page.click('button[type="submit"]')

    # page.click() on a real <form> submit auto-waits for the navigation
    # the POST redirect triggers - no separate wait_for_url() needed (and
    # calling it afterward races the already-finished navigation, since
    # by then there's no new navigation event left to wait for).
    # web_create_invoice() redirects to the plain /web/invoices list on
    # success, not to the new invoice's own detail page - this test
    # originally assumed otherwise, which is exactly the kind of gap a
    # real browser test is meant to catch.
    page.wait_for_load_state("load")
    assert page.url.rstrip("/").endswith("/web/invoices")
    assert "E2E-BROWSER-1" in page.content()


def test_invoices_list_search_box_filters_live(live_server, page):
    """Depends on test_create_invoice_via_real_rendered_form having
    already created an invoice from Golden Grain Milling - runs after it
    in this module (pytest preserves file order by default)."""
    _login(page, live_server)
    page.goto(f"{live_server}/web/invoices?q=golden")
    assert "E2E-BROWSER-1" in page.content()

    page.goto(f"{live_server}/web/invoices?q=no-such-vendor-xyz")
    assert "E2E-BROWSER-1" not in page.content()
