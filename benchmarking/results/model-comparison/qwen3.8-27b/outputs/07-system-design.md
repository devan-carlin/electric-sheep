# Output: qwen3.8-27b × 07-system-design

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

User asks System Design URL Shortener Key Decisions Only. Answer each in 1-2 bullet points. Need likely 7 sections, each 1-2 bullets. Need be precise.

Let's craft.

1 ID generation: Use base62 of monotonically increasing ID from distributed ID generator (Snowflake/DB sequence) or pre-generated batches; avoid raw hash due collisions and non-unique; if deterministic hash use 64-bit hash + uniqueness check but collisions. Maybe mention base62 6 chars ~56B enough. Why: unique, short, no collision, predictable capacity.

2 DB schema: core table links(id PK, short_code unique, long_url, created_at, expires_at, owner_id, click_count maybe). Indexes: unique index on short_code, index on owner_id/created_at for management, maybe expires_at for cleanup. If sharded, short_code prefix or consistent hash.

3 Redirect flow: Use 302 temporary redirect by default to preserve analytics and allow changing target; 301 only if user opts permanent and cacheable. Cache invalidation: set Cache-Control/ETag/Last-Modified; for 301 long TTL; for 302 no-store or short TTL; if target changes, bump version in cache key or purge CDN/edge by short_code.

4 Caching: Redis stores short_code -> long_url (+ metadata, version) hot links; TTL e.g. 1h-24h or LRU with refresh on hit; cache misses go to DB, set with TTL, use negative caching short TTL to avoid stampede; use request coalescing/singleflight.

5 Scaling: 10M redirects/day ~116 rps avg, 1M creates/day ~12 wps avg; read/write ~10:1; bottleneck is read path/edge latency and DB hot keys, not writes; scale with CDN/edge cache, Redis, read replicas, sharding by short_code.

6 Custom aliases: Validate, check uniqueness with DB unique constraint; use atomic INSERT ... ON CONFLICT or Redis SETNX/lock; if collision return 409; race handled by unique index and retry/choose different alias; maybe reserve with short TTL.

7 Expiration: Store expires_at; use delayed queue (

## Response


## Metadata
- **Model**: qwen3.8-27b
- **Elapsed**: 16835ms
- **Prompt tokens**: 238
- **Completion tokens**: 512
- **Total tokens**: 750
- **Gen tok/s**: 30.4
