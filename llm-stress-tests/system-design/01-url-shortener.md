# System Design: URL Shortener

Design a URL shortening service (like bit.ly). Provide:

1. **API design** — REST endpoints with request/response schemas.
2. **Database schema** — tables, indexes, and rationale.
3. **ID generation strategy** — how short codes are created and why.
4. **Scaling strategy** — how to handle 10M daily redirects and 1M daily creates.
5. **Caching strategy** — what to cache, TTL, invalidation.
6. **Edge cases** — collisions, expired links, custom aliases, analytics.

---

## Requirements

- Shorten a URL → return a short code (e.g., `abc123`).
- Redirect short URL → original URL (301 or 302).
- Track click count and referrer for each short URL.
- Support custom short codes (user-provided alias).
- Support expiration (URL stops redirecting after N days).
- API rate limiting: 100 requests/minute per API key.

## Constraints

- Assume a single-region deployment initially (no multi-region replication).
- Use PostgreSQL as the primary database.
- Redis is available for caching.
- Provide the API in Python (FastAPI) with the database schema in SQL.

## Deliverable

1. `api.py` — FastAPI application with all endpoints.
2. `schema.sql` — DDL for all tables with indexes.
3. A brief design document (markdown) covering scaling, caching, and edge cases.

## Evaluation Criteria

| Criterion | Weight |
|-----------|--------|
| API completeness (all 6 requirements covered) | 25% |
| Database schema (proper indexes, no N+1 queries) | 20% |
| ID generation (collision handling, length vs namespace) | 15% |
| Scaling strategy (read/write ratio, cache hit rate) | 20% |
| Edge case handling (custom alias conflicts, expiration cleanup) | 20% |
