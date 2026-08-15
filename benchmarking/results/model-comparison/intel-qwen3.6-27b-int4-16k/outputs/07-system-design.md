# Output: intel-qwen3.6-27b-int4-16k × 07-system-design

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
Here's a thinking process:

1.  **Analyze User Input:**
   - **Topic:** System Design — URL Shortener (Key Decisions Only)
   - **Format:** Answer each of 7 questions in 1-2 bullet points
   - **Questions:**
     1. ID generation: Base62? Hash? Why?
     2. Database schema: Core table? Key indexes?
     3. Redirect flow: 301 vs 302? Why? Cache invalidation?
     4. Caching: What in Redis? TTL strategy? Cache misses?
     5. Scaling: 10M redirects/day, 1M creates/day. Read/write ratio? Bottleneck?
     6. Custom aliases: Collisions? Race conditions?
     7. Expiration: Cleanup without full table scan?

2.  **Deconstruct & Answer Each Question (Mental Draft & Refinement):**

   *Question 1: ID generation*
   - Need short, unique, non-sequential (to prevent enumeration), deterministic or random?
   - Base62 encoding of a monotonically increasing ID (e.g., Snowflake or DB auto-increment) is standard. Why? Predictable length, collision-free, efficient storage/lookup. Hashing (MD5/SHA) is too long; random Base62 needs collision checks.
   - *Draft:* Use Base62 encoding of a distributed unique ID (e.g., Snowflake or DB sequence). Why? Guarantees uniqueness, fixed short length (~7 chars for billions), and avoids collision checks unlike random generation.

   *Question 2: Database schema*
   - Core table: `short_urls` with columns: `id` (PK), `short_code` (unique), `original_url`, `created_by`, `created_at`, `expires_at`, `click_count`.
   - Key indexes: Primary key on `id`, unique index on `short_code` (for fast lookups), index on `expires_at` (for cleanup), index on `created_by` (for user queries).
   - *Draft:* Core table: `short_urls(id PK, short_code UNIQUE, original_url, created_by, created_at, expires_at, click_count)`. Key indexes: Unique index on `short_code` for O(1) lookups, and a secondary index on `expires_at` for efficient expiration queries.

   *Question 3: Redirect flow*
   - 301 (permanent) vs 302 (temporary). 301 caches aggressively in browsers/CDNs, making updates/deletions hard. 302 is better for URL shorteners because links can be updated, deleted, or expire.
   - Cache invalidation: Use short TTLs in CDN/Redis, or include a cache-busting query param/version in the redirect response, or rely on 302 + short TTL.
   - *Draft:* Use 302 (temporary) redirects to allow link updates/deletions without forcing browser/CDN cache purges. Handle invalidation via short CDN/Redis TTLs (e.g., 5–15 mins) and optional cache-control headers that respect expiration status.

   *Question 4: Caching*
   - What in Redis? `short_code -> original_url` mapping, plus metadata like `expires_at`, `click_count`, `is_active`.
   - TTL strategy: Set TTL slightly longer than expected link lifetime or use a fixed short TTL (e.g., 1 hour) with lazy refresh. For permanent links, use long TTL + cache-aside pattern.
   - Cache misses: Fallback to DB, then populate cache. Use write-through or cache-aside. Handle hot keys with local cache or request coalescing.
   - *Draft:* Cache `short_code → {original_url, expires_at, is_active}` in Redis using a cache-aside pattern. Set TTL based on link type (e.g., 1h for standard, ∞ for permanent) with lazy refresh on hits; on miss, fetch from DB, repopulate cache, and serve response.

   *Question 5: Scaling*
   - 10M redirects/day ≈ ~115 reads/sec. 1M creates/day ≈ ~11 writes/sec. Read/write ratio ~10:1.
   - Bottleneck: DB read latency for hot keys, or network/CDN edge if not properly cached. Actually, at this scale, the bottleneck is usually DB connection pool or hot key contention in Redis. Also, write path is light, so reads dominate.
   - *Draft:* Read/write ratio is ~10:1 (~115 reads/sec vs ~11 writes/sec). The bottleneck is typically DB/Redis hot-key contention and connection pooling; mitigated by CDN edge caching, Redis clustering, and read replicas for analytics.

   *Question 6: Custom aliases*
   - Collisions: Check uniqueness before insert. Use DB unique constraint on `short_code`/`alias`.
   - Race conditions: Use DB-level unique constraint + retry logic, or distributed lock (Redis SETNX) during creation. Optimistic concurrency with `INSERT ... ON CONFLICT DO NOTHING` + check return.
   - *Draft:* Enforce uniqueness via a DB unique constraint on `short_code` and use `INSERT ... ON CONFLICT DO NOTHING` with a retry loop. Prevent race conditions with a short-lived distributed lock (Redis SETNX) or atomic DB upsert during creation.

   *Question 7: Expiration*
   - Cleanup without full scan: Use a secondary index on `expires_at`, or a dedicated expiration queue/job. Better: Use Redis TTL for active links, and a background job that queries DB index on `expires_at < NOW()` in batches. Or use a time-series partitioning.
   - *Draft:* Maintain a secondary index on `expires_at` and run a background job that batches-deletes expired rows (`WHERE expires_at < NOW()`). For high scale, use a dedicated expiration queue (e.g., Redis sorted set by expiry time) to trigger lazy cleanup without full table scans.

3.  **Review & Refine (Check against constraints):**
   - Constraint: "Answer each in 1-2 bullet points"
   - Check each answer for length and precision.
   - Ensure technical accuracy and alignment with industry best practices.
   - Format clearly.

   *Self-Correction/Verification during drafting:*
   - Q1: Base62 of Snowflake/sequence is standard. Good. 1 bullet.
   - Q2: Core table + indexes. 1 bullet. Good.
   - Q3: 302 vs 301 + cache invalidation. 1 bullet. Good.
   - Q4: Redis content + TTL + miss handling. 1 bullet. Good.
   - Q5: Ratio + bottleneck. 1 bullet. Good.
   - Q6: Collisions + race conditions. 1 bullet. Good.
   - Q7: Expiration cleanup. 1 bullet. Good.
   - All are concise, technical, and directly answer the prompts. I'll format them exactly as requested.

   Let's tweak slightly to ensure each is strictly 1-2 bullets and hits all sub-questions.
   Q1: Base62 of distributed ID (Snowflake/DB sequence). Why? Collision-free, fixed length (~7 chars), avoids random generation retries.
   Q2: `short_urls(id PK, short_code UNIQUE, original_url, created_by, created_at, expires_at, click_count)`. Indexes: Unique on `short_code` for O(1) lookups, secondary on `expires_at` for cleanup.
   Q3: Use 302 (temporary) to allow link updates/deletions without forcing browser/CDN cache purges. Invalidation handled via short CDN/Redis TTLs (5–15m) and `Cache-Control: no-cache` or versioned query params.
   Q4: Cache `short_code → {original_url, expires_at, is_active}` in Redis via cache-aside. TTL matches link policy (e.g., 1h standard, ∞ permanent) with lazy refresh; on miss, fetch from DB, repopulate, and serve.
   Q5: ~10:1 read/write ratio (~115 reads/sec vs ~11 writes/sec). Bottleneck is DB/Redis hot-key contention and connection pooling; solved via CDN edge caching, Redis clustering, and read replicas.
   Q6: Enforce uniqueness via DB unique constraint + `INSERT ... ON CONFLICT DO NOTHING` with retry. Race conditions prevented by atomic upserts or short-lived distributed locks (Redis SETNX) during creation.
   Q7: Use a secondary index on `expires_at` with a background job that batches-deletes `WHERE expires_at < NOW()`. For high scale, maintain a Redis sorted set keyed by expiry time to trigger lazy, targeted cleanup.

   All meet the 1-2 bullet constraint. Technical depth is appropriate for a system design interview. I'll output exactly this.✅
   One minor adjustment: Q3 cache invalidation could be tighter. "302 + short TTL + Cache-Control: no-cache" is standard. I'll keep it concise.
   Q5: 10M/day = 10,000,000 / 86400 ≈ 115.7 rps. 1M/day ≈ 11.6 wps. Ratio ~10:1. Correct.
   All good. Output matches response. 
   Structure:
   1. **ID generation:** ...
   2. **Database schema:** ...
   etc.
   Each with exactly 1-2 bullets. Ready. 
   Output matches the refined version.✅
   Proceeds. 
   [Self-Correction/Verification during thought]
   - Check Q3: "301 vs 302? Why? How do you handle cache invalidation?" -> My answer covers both in 1 bullet. Good.
   - Check Q4: "What goes in Redis? TTL strategy? How do you handle cache misses?" -> Covered in 1 bullet. Good.
   - Check Q5: "10M redirects/day, 1M creates/day. Read/write ratio? Where's the bottleneck?" -> Covered. Good.
   - Check Q6: "Custom aliases: How do you handle collisions? Race conditions on creation?" -> Covered. Good.
   - Check Q7: "Expiration: How do you clean up expired links without scanning the whole table?" -> Covered. Good.
   All constraints met. Output matches. 
   Generating response. 
   [Output Generation] -> *Proceeds*
   *(Self-Correction/Refinement during output prep)*: I'll make sure each answer is strictly formatted as requested, with clear bullet points. I'll keep it tight.
   Final check of the prompt: "Answer each in 1-2 bullet points:"
   I will format exactly as:
   1. **ID generation:** 
      - ...
   2. **Database schema:**
      - ...
   etc.
   All good.✅
   Output matches the refined version. 
   [Done] 
   *Output Generation* (matches the final refined version)
   "1. **ID generation:** 
      - Use Base62 encoding of a distributed unique ID (e.g., Snowflake or DB sequence). Why? Guarantees collision-free uniqueness, fixed short length (~7 chars for billions), and avoids the retry overhead of random generation.
   2. **Database schema:** 
      - Core table: `short_urls(id PK, short_code UNIQUE, original_url, created_by, created_at, expires_at, click_count)`. Key indexes: Unique index on `short_code` for O(1) lookups, and a secondary index on `expires_at` for efficient expiration queries.
   3. **Redirect flow:** 
      - Use 302 (temporary) redirects to allow link updates/deletions without forcing browser/CDN cache purges. Handle invalidation via short CDN/Redis TTLs (5–15 mins) and `Cache-Control: no-cache` or versioned query parameters.
   4. **Caching:** 
      - Cache `short_code → {original_url, expires_at, is_active}` in Redis using a cache-aside pattern. Set TTL based on link policy (e.g., 1h standard, ∞ permanent) with lazy refresh; on miss, fetch from DB, repopulate cache, and serve.
   5. **Scaling:** 
      - Read/write ratio is ~10:1 (~115 reads/sec vs ~11 writes/sec). The bottleneck is typically DB/Redis hot-key contention and connection pooling; mitigated by CDN edge caching, Redis clustering, and read replicas for analytics.
   6. **Custom aliases:** 
      - Enforce uniqueness via a DB unique constraint + `INSERT ... ON CONFLICT DO NOTHING` with a retry loop. Prevent race conditions using atomic DB upserts or short-lived distributed locks (Redis SETNX) during creation.
   7. **Expiration:** 
      - Maintain a secondary index on `expires_at` and run a background job that batches-deletes `WHERE expires_at < NOW()`. For high scale, use a Redis sorted set keyed by expiry time to trigger lazy, targeted cleanup without full table scans."
   All constraints met. Technical accuracy verified. Ready. 
   Output matches exactly.✅
   [Proceeds]

## Response


1. **ID generation:** 
   - Use Base62 encoding of a distributed unique ID (e.g., Snowflake or DB sequence). Why? Guarantees collision-free uniqueness, fixed short length (~7 chars for billions), and avoids the retry overhead of random generation.

2. **Database schema:** 
   - Core table: `short_urls(id PK, short_code UNIQUE, original_url, created_by, created_at, expires_at, click_count)`. Key indexes: Unique index on `short_code` for O(1) lookups, and a secondary index on `expires_at` for efficient expiration queries.

3. **Redirect flow:** 
   - Use 302 (temporary) redirects to allow link updates/deletions without forcing browser/CDN cache purges. Handle invalidation via short CDN/Redis TTLs (5–15 mins) and `Cache-Control: no-cache` or versioned query parameters.

4. **Caching:** 
   - Cache `short_code → {original_url, expires_at, is_active}` in Redis using a cache-aside pattern. Set TTL based on link policy (e.g., 1h standard, ∞ permanent) with lazy refresh; on miss, fetch from DB, repopulate cache, and serve.

5. **Scaling:** 
   - Read/write ratio is ~10:1 (~115 reads/sec vs ~11 writes/sec). The bottleneck is typically DB/Redis hot-key contention and connection pooling; mitigated by CDN edge caching, Redis clustering, and read replicas for analytics.

6. **Custom aliases:** 
   - Enforce uniqueness via a DB unique constraint + `INSERT ... ON CONFLICT DO NOTHING` with a retry loop. Prevent race conditions using atomic DB upserts or short-lived distributed locks (Redis SETNX) during creation.

7. **Expiration:** 
   - Maintain a secondary index on `expires_at` and run a background job that batches-deletes `WHERE expires_at < NOW()`. For high scale, use a Redis sorted set keyed by expiry time to trigger lazy, targeted cleanup without full table scans.

## Metadata
- **Model**: intel-qwen3.6-27b-int4-16k
- **Elapsed**: 69903ms
- **Prompt tokens**: 196
- **Completion tokens**: 3367
- **Total tokens**: 3563
- **Gen tok/s**: 48.1
