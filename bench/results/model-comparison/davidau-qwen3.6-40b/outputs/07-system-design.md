# Output: davidau-qwen3.6-40b × 07-system-design

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

## Response
This is a concise system design question asking for key decisions only, with 1-2 bullet points per question. Let me answer each directly.

1. **ID generation:**
- Use base62 encoding of auto-incrementing IDs from a distributed ID generator (like Snowflake or consistent hashing of shards). Hashes risk collisions and don't scale predictably.
- Base62 (a-z, A-Z, 0-9) gives 62^7 ≈ 3.5 trillion unique codes with 7 chars.

2. **Database schema:**
- Core table: `urls(short_code PK, original_url, user_id, created_at, expires_at, click_count)`
- Index on `short_code` (primary key), secondary index on `(user_id, created_at)` for user queries.

3. **Redirect flow:**
- Use 302 (temporary) redirect so analytics can be tracked and links can be updated/revoked.
- Cache redirects in CDN with short TTL (1-5 min); invalidate on link deletion/update by purging CDN cache.

4. **Caching:**
- Cache hot short_code → original_url mappings in Redis with TTL based on access frequency (e.g., 1h-24h).
- Cache miss: fetch from DB, populate Redis, return response (write-through pattern).

5. **Scaling:**
- Read/write ratio ~100:1 (most operations are redirects, not creations).
- Bottleneck: read-heavy redirect path; solve with CDN + Redis + read replicas. Write bottleneck on ID generation solved with sharded ID generators.

6. **Custom aliases:**
- Check uniqueness with SELECT before INSERT; use database-level unique constraint on `short_code` to prevent race conditions.
- On collision, return error or suggest alternatives.

7. **Expiration:**
- Use a separate expiration table or Redis sorted set mapping (timestamp, short_code) for efficient lookup.
- Background job processes expirations in time windows (e.g., every hour) instead of scanning main table.

Let me refine these to be even more concise and impactful.
</think>

1. **ID generation:**
- Base62-encode auto-incrementing IDs from a distributed ID generator (Snowflake or sharded DB). Predictable, no collisions, scales linearly.
- Avoid hashes: collision risk, can't estimate capacity, harder to shard.

2.

## Metadata
- **Model**: davidau-qwen3.6-40b
- **Elapsed**: 24120ms
- **Prompt tokens**: 196
- **Completion tokens**: 512
- **Total tokens**: 708
- **Gen tok/s**: 21.2
