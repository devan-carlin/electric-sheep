# Output: qwen3.8-27b-16k × 07-system-design

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
We need answer user's request: System Design URL Shortener Key Decisions Only. Need answer each in 1-2 bullet points. Need likely concise. Need think through best answers.

Need produce final with numbered list, each 1-2 bullets. Let's craft precise.

1 ID generation: Use unique snowflake/sequence + base62? Or hash? For URL shortener, need short, collision-free, non-enumerable? Options: generate unique ID via DB auto-increment/snowflake, encode base62. Hash (MD5) can collide and not unique; can use hash for deterministic idempotent but need collision handling. Why base62: compact, URL-safe, 6 chars ~56 billion. Could use random base62 with collision check. Key decisions: use unique ID generator (snowflake/DB sequence) encoded base62, not raw hash, to guarantee uniqueness and avoid collisions; if deterministic idempotent, use hash + collision fallback.

2 DB schema: core table links(id PK, short_code unique, long_url, created_at, expires_at, owner_id, click_count maybe). Indexes: unique index on short_code for lookup; index on owner_id/created_at for management; maybe expires_at for cleanup. If sharded, short_code prefix? Keep simple.

3 Redirect flow: 302 temporary vs 301 permanent. Use 302 to preserve analytics and allow changing target/expiring; 301 caches aggressively and makes invalidation hard. Cache invalidation: don't cache redirect response long; use short TTL or no-store for 302; if 301, use versioned URL or purge CDN. For cache invalidation: store mapping in cache with short TTL; on update/delete, delete cache key and set short TTL; use CDN purge if needed.

4 Caching: Redis stores short_code -> long_url (and metadata) hot mappings. TTL: short TTL e.g. 5-60s for redirects, longer for immutable? Since 302, short TTL. Cache misses: read DB, populate cache, maybe request coalescing/singleflight to avoid stampede; negative caching? For misses, DB lookup.

5 Scaling: 10M redirects/day, 1M creates/day. Read/write ratio ~10:1. Bottleneck: read path (redirect lookups) and cache/DB hot keys; writes less. Need cache, CDN, read replicas, sharding by short_code. Maybe DB write for creates and click counts; avoid counting every click in DB, async.

6 Custom aliases: handle collisions: validate uniqueness with unique index; check-then-insert race: use DB unique constraint and retry with generated suffix or reject; use optimistic insert; for high concurrency, use Redis SETNX or DB unique. Maybe reserve aliases.

7 Expiration: avoid scanning whole table: store expires_at and use time-based partitioning or queue. On lookup, check expires_at and return 410 if expired; for cleanup, use secondary index on expires_at and delete in batches where expires_at < now, or use TTL in Redis and scheduled jobs per shard/partition; maybe use Kafka/queue when created with expiration. Need 1-2 bullets.

Need ensure each answer 1-2 bullet points. Could be concise but enough. User asks "Answer each in 1-2 bullet points". So each numbered item should have 1 or 2 bullets. Need not overdo. Let's craft final with exactly 2 bullets per item maybe. Ensure no extra paragraphs.

Potential final:
1. **ID generation:**
- Use a unique ID generator (DB sequence/Snowflake) encoded in Base62; Base62 is URL-safe and compact (6 chars ≈ 56B), and unique IDs avoid hash collisions.
- If idempotent/deterministic short codes are needed, use a strong hash (e.g. SHA-256 truncated) plus a unique-index collision check/fallback, not raw hash as primary key.

2. **Database schema:**
- Core table: `links(id PK, short_code UNIQUE, long_url, owner_id, created_at, expires_at, click_count)`.
- Indexes: unique on `short_code` for O(1) redirect lookup; secondary indexes on `owner_id/created_at` for management and `expires_at` for cleanup.

3. **Redirect flow:**
- Prefer 302/307 over 301 so we can keep analytics, change targets, and expire links; 301 causes browsers/CDNs to cache permanently.
- Invalidation: use short TTL/no-store for redirect responses and delete/purge cache key on update/delete; if using 301, require versioned short code or CDN purge.

4. **Caching:**
- Cache `short_code -> long_url` (and expiry/status) in Redis for hot redirects; use short TTL (seconds/minutes) for mutable links, longer only for immutable/301 links.
- On miss, read DB and repopulate; use singleflight/request coalescing and optional negative cache to prevent stampedes.

5. **Scaling:**
- Read/write ratio is ~10:1 (10M reads/redirects vs 1M creates/day), so redirect lookup is the bottleneck, not creation.
- Scale with Redis/CDN in front, read replicas, and sharding by `short_code`; make click counting async to avoid write amplification.

6. **Custom aliases:**
- Enforce uniqueness with a DB unique index on `short_code`; check availability first, but rely on the constraint to make check-then-insert safe.
- On race/collision, catch duplicate-key error and either reject, auto-suffix, or retry with a generated code; use Redis `SETNX`/DB transaction for high-contention aliases.

7. **Expiration:**
- On lookup, check `expires_at` and return 410/404; use Redis TTL for hot expired entries so they stop serving quickly.
- Clean up in batches using an index on `expires_at` (or time-partitioned tables/queue) with `WHERE expires_at < now()`, avoiding full-table scans.

Need maybe "1-2 bullet points" each. This is 2 each. Good.

Double-check technical nuance: 302 vs 307? 302 is temporary, may change method historically; 307 preserves method. For URL shortener, 302 common. Could mention 302/307. Cache invalidation: 302 can be cached by browsers? Use Cache-Control: no-store or short max-age. Good.

ID generation: Base62 vs hash. Could say use random base62 with collision check? But unique ID generator better. Need answer "How do you create short codes? Base62? Hash? Why?" We can say Base62 encoding of unique ID, not hash. Good.

DB schema: core table and key indexes. Good.

Caching: What goes in Redis? TTL strategy? cache misses? Good.

Scaling: 10M redirects/day, 1M creates/day. Read/write ratio? Where bottleneck? Good.

Custom aliases: collisions? race conditions? Good.

Expiration: cleanup without scanning whole table? Good.

Need final only. Ensure no markdown table? Numbered list with bullets. Fine.

## Response


1. **ID generation:**
   - Use a unique ID generator (DB sequence/Snowflake) encoded in Base62; Base62 is URL-safe and compact, and unique IDs avoid hash collisions.
   - If deterministic/idempotent codes are needed, use a strong hash plus a unique-index collision check/fallback, not raw hash as the primary source of uniqueness.

2. **Database schema:**
   - Core table: `links(id PK, short_code UNIQUE, long_url, owner_id, created_at, expires_at, click_count)`.
   - Key indexes: unique index on `short_code` for redirect lookup; secondary indexes on `owner_id/created_at` for management and `expires_at` for cleanup.

3. **Redirect flow:**
   - Prefer 302/307 over 301 so links can expire, change targets, and preserve analytics; 301 causes aggressive browser/CDN caching.
   - Invalidation: use short TTL or `no-store` for redirect responses, and delete/purge the cache key on update/delete; if using 301, require versioned codes or CDN purge.

4. **Caching:**
   - Cache `short_code -> long_url` plus status/expiry in Redis; use short TTL for mutable links and longer TTL only for immutable/301 links.
   - On cache miss, read DB and repopulate; use request coalescing/singleflight and optional negative caching to avoid stampedes.

5. **Scaling:**
   - Read/write ratio is ~10:1 (10M redirects/day vs 1M creates/day), so redirect lookup is the main bottleneck.
   - Scale with Redis/CDN in front, read replicas, and sharding by `short_code`; make click counting async to avoid write amplification.

6. **Custom aliases:**
   - Enforce uniqueness with a DB unique index on `short_code`; check availability first, but rely on the constraint to make check-then-insert safe.
   - On race/collision, catch duplicate-key error and reject, auto-suffix, or retry with a generated code; use Redis `SETNX`/DB transaction for high-contention aliases.

7. **Expiration:**
   - On lookup, check `expires_at` and return 410/404; use Redis TTL for hot entries so expired links stop serving quickly.
   - Clean up in batches using an index on `expires_at` or time-partitioned tables/queues with `WHERE expires_at < now()`, avoiding full-table scans.

## Metadata
- **Model**: qwen3.8-27b-16k
- **Elapsed**: 70740ms
- **Prompt tokens**: 238
- **Completion tokens**: 2135
- **Total tokens**: 2373
- **Gen tok/s**: 30.1
