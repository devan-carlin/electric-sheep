# System Design: Distributed Rate Limiter

Design and implement a distributed rate limiter that can be shared across multiple API server instances.

## Requirements

- **Token bucket algorithm** — each client gets a bucket with a max capacity and refill rate.
- **Distributed state** — multiple API servers share the same rate limit state via Redis.
- **Atomic operations** — no race conditions when two requests from the same client hit different servers simultaneously.
- **Configurable limits** — different limits per endpoint (e.g., `/api/login`: 5/min, `/api/search`: 30/min).
- **HTTP headers** — return `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset` on every response.
- **429 responses** — return `Retry-After` header when rate limited.

## Implementation

Build a Python FastAPI application with:

1. A `RateLimiter` class that wraps Redis operations using Lua scripts for atomicity.
2. A FastAPI middleware that applies rate limiting per client IP + endpoint.
3. A configuration file (YAML or dict) that maps endpoint patterns to limits.
4. A `/health` endpoint that bypasses rate limiting.

## Constraints

- Use Redis Lua scripts (EVALSHA) for atomic token bucket operations — do not use Python-level locking.
- The Lua script should implement: check current tokens → deduct if available → return remaining + reset time.
- Handle Redis connection failures gracefully (fail open: allow the request if Redis is down).
- Bucket state should persist in Redis with a TTL equal to 2× the refill period (cleanup stale buckets).

## Deliverable

1. `rate_limiter.py` — the RateLimiter class with Lua script.
2. `main.py` — FastAPI app with middleware.
3. `test_rate_limiter.py` — pytest file that verifies:
   - Requests within limit succeed (200).
   - Requests exceeding limit are rejected (429).
   - Tokens refill after the refill period.
   - Headers are present and correct.

## Evaluation Criteria

| Criterion | Weight |
|-----------|--------|
| Correct token bucket implementation | 30% |
| Atomic Redis operations (Lua script) | 25% |
| Middleware integration (headers, 429, Retry-After) | 20% |
| Error handling (Redis down = fail open) | 15% |
| Test coverage | 10% |
