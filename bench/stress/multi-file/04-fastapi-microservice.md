# Backend Architecture — Standalone FastAPI Microservice

**Target Capability:** Single-file production code, database schema generation, async handling, REST standards.

---

## Prompt

```
Create a single, complete, runnable Python script that implements a URL shortener microservice using FastAPI and SQLite. The entire application must be in one file — no external files, no placeholders, no TODO comments.

Requirements:
1. Database: Use sqlite3 (or SQLAlchemy) with automatic table creation on startup. Tables:
   - `short_urls`: id (integer PK), short_code (text unique), original_url (text), created_at (timestamp), click_count (integer default 0)
2. Endpoints:
   - POST /shorten — Accepts {"url": "<valid_http_url>"}, returns {"short_code": "abc123", "short_url": "http://localhost:8000/r/abc123"} with status 201. Return 422 if URL is invalid.
   - GET /r/{short_code} — Redirects (302) to the original URL and increments click_count. Return 404 if code not found.
   - GET /stats/{short_code} — Returns {"short_code": "...", "original_url": "...", "click_count": N, "created_at": "..."} with status 200. Return 404 if not found.
   - GET /health — Returns {"status": "ok"} with status 200.
3. Validation: Use Pydantic models for request/response validation. Reject non-HTTP(S) URLs with 422.
4. Short Code: Generate a random 6-character alphanumeric code. Handle collisions by regenerating.
5. Startup: Create the database table if it doesn't exist. Print a confirmation message.

The script should run with: `uvicorn main:app --reload` (assume the file is named main.py).
```
