# Authentication Implementation

**Category:** Security
**Target:** JWT, session management, password hashing, OAuth flow

---

## Prompt

Build a complete authentication service in Python (FastAPI) with the following features:

**Structure:**

```
auth-service/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app, router mounting
│   ├── auth.py            # JWT creation/validation, password hashing
│   ├── models.py          # SQLAlchemy models (User, RefreshToken, APIKey, AuditLog)
│   ├── schemas.py         # Pydantic schemas (request/response)
│   ├── crud.py            # Database operations
│   ├── dependencies.py    # FastAPI dependencies (get_current_user, require_role)
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── auth.py        # /login, /register, /refresh, /logout, /forgot-password
│   │   ├── users.py       # /me, /change-password
│   │   └── admin.py       # /admin/users, /admin/keys
│   └── security/
│       ├── __init__.py
│       ├── password.py    # Password hashing (bcrypt)
│       ├── jwt.py         # JWT token management
│       ├── api_keys.py    # API key generation/validation
│       └── rate_limit.py  # Per-endpoint rate limiting
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_auth.py
│   ├── test_password.py
│   └── test_jwt.py
├── alembic.ini
├── alembic/
├── pyproject.toml
└── README.md
```

**Requirements:**

- **Registration:** Email + password, email verification via token, password strength validation
- **Login:** Email + password, returns access token (15 min) + refresh token (7 days)
- **Refresh:** Refresh token rotation (old token invalidated on refresh)
- **Logout:** Revoke refresh token, optional revoke all sessions
- **Password reset:** Time-limited token (1 hour), single-use, invalidates on use
- **API keys:** Generate/revoke API keys with scoped permissions and expiration
- **Audit logging:** Log all auth events (login, logout, password change, key generation)
- **Rate limiting:** 
  - Login: 5 attempts per minute per IP
  - Registration: 3 attempts per hour per IP
  - Password reset: 3 attempts per hour per email
- **RBAC:** Roles (admin, user, viewer), permission checks via FastAPI dependencies
- **Security headers:** HSTS, X-Content-Type-Options, X-Frame-Options, CORS configuration

**Security requirements:**

- Passwords hashed with bcrypt (cost factor 12)
- JWT with RS256 (asymmetric keys, not HS256)
- Refresh tokens stored in database with hash (never store raw token)
- Token blacklist for revoked tokens
- Account lockout after 5 failed login attempts (30 minute cooldown)
- Password policy: minimum 12 chars, uppercase, lowercase, number, special char

**Constraints:**

- Python 3.11+, FastAPI, SQLAlchemy 2.0, Pydantic v2
- SQLite for development, PostgreSQL-compatible queries
- Tests must cover: successful auth flow, token expiration, refresh rotation, account lockout, rate limiting
- `pyproject.toml` with all dependencies

Produce all files with complete working code. No placeholders, no TODOs.
