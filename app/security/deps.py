"""
FastAPI dependencies for the web UI's auth gate and RBAC (Section 12).
Not applied to the JSON API (app/api/*) - see the README's security
section for why that's a documented limitation, not an oversight.
"""

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.models import User, UserRole
from app.security.auth import verify_session_token

SESSION_COOKIE_NAME = "session"


class NotAuthenticated(Exception):
    def __init__(self, next_path: str = "/"):
        self.next_path = next_path


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None
    user_id = verify_session_token(token)
    if user_id is None:
        return None
    return db.get(User, user_id)


def require_login(request: Request, db: Session = Depends(get_db)) -> User:
    user = get_current_user(request, db)
    if user is None:
        raise NotAuthenticated(next_path=request.url.path)
    return user


def require_owner(user: User = Depends(require_login)) -> User:
    """Section 12: approving/rejecting agent-flagged actions is
    reserved for the account owner - a bookkeeper can create and edit
    invoices, but cannot be the one who signs off on a high-risk one."""
    if user.role != UserRole.owner:
        raise HTTPException(status_code=403, detail="Only the account owner can approve or reject.")
    return user
