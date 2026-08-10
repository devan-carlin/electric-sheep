# Full-Stack CRUD Application

**Category:** Multi-file project
**Target:** Multi-module architecture, routing, data layer, auth, testing

---

## Prompt

Build a complete task management API with the following structure:

```
task-api/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app, middleware, lifespan
│   ├── auth.py            # JWT auth, password hashing
│   ├── models.py          # SQLAlchemy models
│   ├── schemas.py         # Pydantic request/response schemas
│   ├── crud.py            # Database operations
│   └── routes/
│       ├── __init__.py
│       ├── tasks.py       # CRUD endpoints
│       └── users.py       # Auth endpoints
├── tests/
│   ├── __init__.py
│   ├── conftest.py        # Test fixtures, test DB
│   └── test_tasks.py      # Endpoint tests
├── alembic.ini
├── requirements.txt
└── README.md
```

**Requirements:**

- SQLite database with Alembic migrations
- JWT authentication (register, login, protected routes)
- Full CRUD for tasks (title, description, status, priority, due_date)
- Input validation with Pydantic v2
- Async database operations (SQLAlchemy 2.0 + asyncpg/sqlite)
- Proper HTTP status codes (401, 403, 404, 422, 500)
- Test suite with pytest-asyncio that covers all endpoints
- README with setup instructions and curl examples

**Constraints:**

- No external dependencies beyond FastAPI, SQLAlchemy, Pydantic, pytest
- Passwords hashed with bcrypt
- JWT tokens expire after 24 hours
- All database access goes through `crud.py` — never raw queries in routes
- Tests must use a separate in-memory database

Produce all files with complete working code. No placeholders, no TODOs.
