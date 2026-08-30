# Output: qwen3.8-27b-2k × 07-system-design

## Prompt
# System Design — URL Shortener (Key Decisions Only)

Design a URL shortener (like bit.ly). Answer each in 1-2 bullet points:

1. **ID generation:** How do you create short codes? Base62? Hash? Why?
2. **Database schema:** What's the core table? Key indexes?
3. **Redirect flow:** 301 vs 302? Why? How do you handle cache invalidation?
4. **Caching:** What goes in Redis? TTL strategy? How do you handle cache misses?
5. **Scaling:** 10M redirects/day, 1M creates/day. Read/write ratio? Where's the bottleneck?
6. **Custom aliases:** How do you handle collisions? Race conditions on creation?
7. **Expiration:** How do you clean up expired links without scanning the whole table?

## Reasoning
We need answer user's request. Need produce final answer concise: each 1-2 bullet points. Need think through content.

User asks System Design URL Shortener Key Decisions Only. Answer each in 1-2 bullet points. Need likely bullet per question maybe sub-bullets. Need be precise.

Let's craft:
1 ID generation: Use Base62 of monotonically increasing ID from Snowflake/DB sequence or preallocated ranges; avoid raw hash due collisions and no uniqueness guarantee; if deterministic hash use 64-bit hash + uniqueness check but collisions. Maybe mention 6-7 chars enough. 1-2 bullets.
2 DB schema: core table links(id PK, short_code unique, long_url, created_by, created_at, expires_at, status). Indexes: unique short_code, maybe created_by+created_at, expires_at for cleanup. If sharded, short_code prefix.
3 Redirect flow: 302 temporary by default to preserve referrer and allow analytics/expiration; 301 only if user opts permanent. Cache invalidation: set Cache-Control/ETag/Expires, use versioned URL or purge CDN/edge cache on delete/update; for 301 hard to invalidate, so avoid or use short TTL.
4 Caching: Redis maps short_code -> long_url + metadata (expires, status). TTL: short for mutable/expiration (e.g., 5-15 min) or until expiry; hot links longer. Cache miss: read DB, populate cache, negative cache short TTL for invalid codes.
5 Scaling: 10M redirects/day ~116 rps avg, 1M creates ~12 wps; read/write ~10:1. Bottleneck: read path/DB lookups and edge latency; mitigate with CDN/Redis, read replicas, sharding by short_code; writes less critical.
6 Custom aliases: validate, check uniqueness with DB unique constraint; use atomic INSERT ... ON CONFLICT or distributed lock/Redis SETNX; if collision return error or auto-suffix; handle race by unique index as source of truth.
7 Expiration: store expires_at; use time-based partitioning or secondary index on expires_at; background job selects expired batches by expires_at < now (limit) and deletes/invalidates; or lazy check on read and mark expired; avoid full scan.

Need ensure each answer 1-2 bullet points. Could format numbered with bullets. Maybe each item has two bullets. Need not overdo. Final only. Ensure no extra.

## Response


1. **ID generation**
   - Use **Base62 encoding of a unique monotonic ID** (e.g., Snowflake, DB sequence, or preallocated ID ranges) to get short, collision-free codes.
   - Avoid plain hashing as the primary method because hashes can collide; if using hashes, still need a uniqueness check or retry logic.

2. **Database schema**
   - Core table: `links(id PK, short_code UNIQUE, long_url, created_by, created_at, expires_at, status)`.
   - Key indexes: **unique index on `short_code`**, plus indexes on `created_by` and `expires_at` for ownership queries and expiration cleanup.

3. **Redirect flow**
   - Use **302 temporary redirects** by default so analytics, expiration, and updates still work; use **301 only if the user explicitly wants permanent redirects**.
   - Handle cache invalidation by using short TTLs, versioned redirect responses, or explicit CDN/edge cache purge on delete/update; 301s are hard to invalidate, so prefer 302 unless permanence is required.

4. **Caching**
   - Cache `short_code -> long_url + metadata` in Redis, with TTL based on volatility: short TTL for expiring/mutable links, longer TTL for stable hot links.
   - On cache miss, read from DB and repopulate Redis; optionally use a short negative cache for invalid/expired codes to reduce DB load.

5. **Scaling**
   - At 10M redirects/day and 1M creates/day, the read/write ratio is roughly **10:1**, so the main bottleneck is the **redirect read path**, not creation.
   - Scale reads with Redis/CDN edge caching, read replicas, and sharding by `short_code`; writes can be handled with a smaller write path and async analytics.

6. **Custom aliases**
   - Enforce uniqueness with a **DB unique constraint on `short_code`** and use atomic operations like `INSERT ... ON CONFLICT` or Redis `SETNX` to handle races.
   - If a collision occurs, return an error or auto-generate a suffix; the database unique index is the final source of truth.

7. **Expiration**
   - Store `expires_at` and use a background job that queries only expired rows, e.g. `WHERE expires_at < now() LIMIT N`, instead of scanning the whole table.
   - Also do **lazy expiration on read**: if a cached/DB link is expired, return 410/404 and mark it for cleanup.

## Metadata
- **Model**: qwen3.8-27b-2k
- **Elapsed**: 36163ms
- **Prompt tokens**: 238
- **Completion tokens**: 1095
- **Total tokens**: 1333
- **Gen tok/s**: 30.2
