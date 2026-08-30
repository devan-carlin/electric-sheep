# Output: qwen3.8-27b-int4-16k × 07-system-design

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
We need answer user's request: design URL shortener key decisions only, answer each in 1-2 bullet points. Need produce final concise. Need think through each.

1 ID generation: Use base62 (a-z A-Z 0-9) generated from unique snowflake/sequence or hash? Need avoid collisions. Could use deterministic hash (e.g. SHA-256 truncated) but collisions possible; better use unique ID (snowflake/DB sequence) encoded base62, or hash with collision check. Why: short, URL-safe, high entropy, no sequential leakage if randomized. 1-2 bullets.

2 DB schema: core table urls(id PK, short_code unique, long_url, created_at, expires_at, status, maybe owner). Indexes: unique on short_code, maybe index on expires_at for cleanup, maybe long_url if dedup. Need key indexes.

3 Redirect flow: 302 vs 301? For analytics and ability to change target, use 302/307? bit.ly uses 301? Key decision: 302 temporary to preserve analytics and allow updates; 301 permanent if immutable and better caching. Cache invalidation: if 301, hard; use short TTL or versioned URL / ETag? For 302, no long browser cache; CDN can cache with short TTL and purge on update. Need answer 1-2 bullets.

4 Caching: Redis stores short_code -> long_url (and metadata) maybe hot redirects. TTL: short TTL for mutable (e.g. 60s-5min) or longer for immutable; cache misses go to DB and populate. Handle negative caching? Maybe store null briefly.

5 Scaling: 10M redirects/day, 1M creates/day. Read/write ratio 10:1. Bottleneck: read path (redirect lookups) and DB hot keys; use cache/CDN, read replicas, sharding by short_code. Creates less bottleneck but need unique ID.

6 Custom aliases: handle collisions: check uniqueness before insert, use unique constraint; race conditions: use DB unique index and retry with generated suffix or reject; or use distributed lock/atomic insert. 1-2 bullets.

7 Expiration: cleanup without scanning whole table: use expires_at index and batch query WHERE expires_at < now LIMIT N; or schedule per bucket / time-partitioned tables; or lazy delete on access. Need answer.

Need ensure each in 1-2 bullet points. Could format numbered with bullets. Keep concise. Need maybe mention base62 vs hash. Let's craft final with 1-2 bullets per item. Ensure not too long.

Potential final:
1. **ID generation:**
- Use a unique 64-bit ID (Snowflake/sequence) encoded in Base62, not raw hash, to guarantee uniqueness and avoid collision handling.
- If using hash (e.g., truncated SHA-256), add a uniqueness check/retry; Base62 gives URL-safe 62^N space and short codes.
But 2 bullets okay.

2. **Database schema:**
- Core table: `urls(id PK, short_code UNIQUE, long_url, created_at, expires_at, status, owner_id)`.
- Indexes: unique on `short_code` for O(1) redirect lookup; secondary on `expires_at` for cleanup; optional unique on `long_url` if dedup.

3. **Redirect flow:**
- Prefer 302/307 for mutable links so you can change target and count analytics; use 301 only for permanent immutable links.
- Cache invalidation: keep CDN/browser TTL short for 302; for 301 use versioned target or purge on update, otherwise stale.

4. **Caching:**
- Redis: `short_code -> long_url` (plus status/expires) for hot redirects; optionally cache custom alias metadata.
- TTL: 30s–5min for mutable, longer for immutable; on miss read DB, write-through, and negative-cache briefly to avoid stampedes.

5. **Scaling:**
- Read/write ratio ~10:1; bottleneck is redirect reads, not creates.
- Put cache/CDN in front, shard DB by `short_code`, use read replicas; ID generation must be globally unique and low-latency.

6. **Custom aliases:**
- Enforce uniqueness with DB unique index; pre-check for UX but rely on constraint for correctness.
- On race, catch duplicate key and either reject or auto-suffix/retry; avoid distributed locks if DB constraint suffices.

7. **Expiration:**
- Use indexed `expires_at` and batch delete/update `WHERE expires_at < now() LIMIT N` in a background job.
- Or partition by expiration bucket/time and lazy-expire on read; never full table scan.

Need maybe 1-2 bullet points each. Good. Ensure no extra. final.

## Response


1. **ID generation:**
- Use a unique 64-bit ID (Snowflake/sequence) encoded in **Base62** to get short, URL-safe codes without relying on collision-prone hashes.
- If using a hash, use a truncated strong hash plus uniqueness check/retry; Base62 is preferred for predictable length and large space.

2. **Database schema:**
- Core table: `urls(id PK, short_code UNIQUE, long_url, created_at, expires_at, status, owner_id)`.
- Key indexes: unique on `short_code` for fast redirects; index on `expires_at` for cleanup; optional unique on `long_url` if deduping.

3. **Redirect flow:**
- Use **302/307** for mutable links so you can change the target and preserve analytics; use **301** only for permanent, immutable links.
- Cache invalidation: keep CDN/browser TTL short for 302; for 301, use versioned URLs or explicit purge on update.

4. **Caching:**
- Store `short_code -> long_url` in Redis for hot redirects; optionally cache status/expiration metadata.
- TTL: short for mutable links, longer for immutable; on miss, read DB, populate cache, and briefly negative-cache to avoid stampedes.

5. **Scaling:**
- Read/write ratio is about **10:1**; the main bottleneck is redirect reads, not link creation.
- Put cache/CDN in front, shard by `short_code`, use read replicas, and ensure ID generation is globally unique and low-latency.

6. **Custom aliases:**
- Enforce uniqueness with a DB unique index; pre-check for UX, but rely on the constraint for correctness.
- On race conditions, catch duplicate-key errors and either reject or auto-suffix/retry; avoid distributed locks if the DB constraint is enough.

7. **Expiration:**
- Use an indexed `expires_at` column and batch cleanup: `WHERE expires_at < now() LIMIT N`.
- Alternatively, partition by expiration bucket or lazy-expire on read; avoid full table scans.

## Metadata
- **Model**: qwen3.8-27b-int4-16k
- **Elapsed**: 32140ms
- **Prompt tokens**: 238
- **Completion tokens**: 1527
- **Total tokens**: 1765
- **Gen tok/s**: 47.5
