"""
Single-command startup: `python run.py`. Equivalent to
`uvicorn app.main:app --reload`, provided so the README's "one
preferred startup method" (Section 38) is a plain `python` command
that works identically on Windows/macOS/Linux without relying on the
uvicorn CLI being on PATH.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
