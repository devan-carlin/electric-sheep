# Output: intel-qwen3.6-27b-int4-2k × 07-system-design

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
   - Need short, unique, non-sequential (to prevent enumeration), deterministic.
   - Base62 encoding of auto-incrementing ID or distributed ID (Snowflake) is standard. Hashing (MD5/SHA) is too long and collision-prone without truncation.
   - *Draft:* Use a distributed auto-incrementing ID (e.g., Snowflake or DB sequence) encoded in Base62 for compactness, uniqueness, and O(1) generation. Avoid cryptographic hashes due to length, collision risk, and non-deterministic enumeration.

   *Question 2: Database schema*
   - Core table: `short_urls` with columns: `id` (PK), `short_code` (unique), `long_url`, `created_by`, `created_at`, `expires_at`, `click_count`.
   - Indexes: Primary key on `id`, unique index on `short_code` for fast lookups, index on `expires_at` for cleanup, index on `created_by` for user queries.
   - *Draft:* Core table: `short_urls(id PK, short_code UNIQUE, long_url, created_by, created_at, expires_at, click_count)`. Key indexes: unique index on `short_code` for O(1) lookups, and a secondary index on `expires_at` for efficient expiration cleanup.

   *Question 3: Redirect flow*
   - 301 (permanent) vs 302 (temporary). 302 is better for URL shorteners because it allows updating the destination URL later without breaking cached redirects. 301 gets cached aggressively by browsers/CDNs.
   - Cache invalidation: Use short TTLs in CDN/Redis, or versioned URLs, or rely on 302 to avoid aggressive caching. For updates, invalidate Redis/CDN cache or use a cache-busting parameter.
   - *Draft:* Use 302 (temporary) redirects to allow destination URL updates without breaking browser/CDN caches. Handle invalidation via short CDN/Redis TTLs (e.g., 5–15 mins) and explicit cache purge on URL updates.

   *Question 4: Caching*
   - What in Redis: `short_code -> long_url` mapping, plus click counters.
   - TTL strategy: Short TTL (e.g., 5-15 mins) for hot links, longer for cold. Use LRU/LFU eviction.
   - Cache misses: Fallback to DB, then repopulate Redis. Use write-through or cache-aside pattern.
   - *Draft:* Cache `short_code → long_url` mappings and click counters in Redis using a cache-aside pattern. Set short TTLs (5–15 mins) with LRU eviction; on misses, fetch from DB, repopulate cache, and serve the redirect.

   *Question 5: Scaling*
   - 10M redirects/day = ~115 req/sec avg, but peak could be 10x-100x. 1M creates/day = ~11.5 req/sec. Read/write ratio ~10:1.
   - Bottleneck: DB read latency for hot keys, or network/CDN if not properly cached. Redis handles most reads. DB write bottleneck for creates/clicks. Use async click counting, read replicas, sharding by short_code prefix.
   - *Draft:* Read/write ratio is ~10:1; peak read traffic drives the bottleneck. Mitigate with CDN edge caching, Redis for hot keys, DB read replicas, and async click counting to avoid write contention.

   *Question 6: Custom aliases*
   - Collisions: Check uniqueness before insert. Use DB unique constraint + retry logic.
   - Race conditions: Use DB-level unique constraint with `INSERT ... ON CONFLICT` or distributed lock/transaction. Return error if taken.
   - *Draft:* Enforce uniqueness via a DB unique constraint on `short_code` with `INSERT ... ON CONFLICT` to atomically handle collisions. Prevent race conditions by relying on DB-level locking/transactions rather than application-level checks.

   *Question 7: Expiration*
   - Cleanup without full scan: Use a background job that queries `WHERE expires_at < NOW()` using the `expires_at` index. Alternatively, use a time-series partitioned table or lazy deletion (check on access, mark as expired, batch delete later).
   - *Draft:* Use a background worker querying `WHERE expires_at < NOW()` on the indexed `expires_at` column for batch deletion. Supplement with lazy expiration (check on access, mark stale, and purge in batches) to avoid heavy DB load.

3.  **Check Constraints:**
   - Each answer must be 1-2 bullet points.
   - Cover all 7 questions.
   - Keep it concise, focused on key decisions.

   Let's format strictly as requested. I'll ensure each answer is exactly 1-2 bullets, directly addressing the prompts.

   *Self-Correction/Refinement during drafting:*
   - Q1: Base62 vs Hash? I'll explicitly state Base62 of distributed ID, explain why over hash.
   - Q2: Core table + indexes. Keep it tight.
   - Q3: 301 vs 302 + cache invalidation. Clear.
   - Q4: Redis content + TTL + miss handling. Clear.
   - Q5: Ratio + bottleneck. Clear.
   - Q6: Collisions + race conditions. Clear.
   - Q7: Expiration cleanup. Clear.

   All fit 1-2 bullets. I'll polish the wording to be precise and professional.

   Final check of the prompt: "Answer each in 1-2 bullet points:"
   I will structure it exactly as:
   1. **ID generation:** ...
      - bullet
      - bullet (optional)
   etc.

   Let's draft carefully. 
   Q1: 
   - Use a distributed auto-incrementing ID (e.g., Snowflake or DB sequence) encoded in Base62 for compactness, guaranteed uniqueness, and O(1) generation.
   - Avoid cryptographic hashes due to excessive length, collision risk when truncated, and inability to support deterministic enumeration or easy range queries.

   Q2:
   - Core table: `short_urls(id PK, short_code UNIQUE, long_url, created_by, created_at, expires_at, click_count)`.
   - Key indexes: unique index on `short_code` for O(1) lookups, and a secondary index on `expires_at` for efficient expiration cleanup.

   Q3:
   - Use 302 (temporary) redirects to allow destination URL updates without breaking aggressive browser/CDN caching.
   - Handle invalidation via short CDN/Redis TTLs (5–15 mins) and explicit cache purge on URL updates; 302 ensures clients revalidate on each request.

   Q4:
   - Cache `short_code → long_url` mappings and click counters in Redis using a cache-aside pattern.
   - Set short TTLs (5–15 mins) with LRU eviction; on misses, fetch from DB, repopulate cache, and serve the redirect without blocking.

   Q5:
   - Read/write ratio is ~10:1; peak read traffic (hot links) is the primary bottleneck.
   - Mitigate with CDN edge caching, Redis for hot keys, DB read replicas, and async click counting to decouple writes from the redirect path.

   Q6:
   - Enforce uniqueness via a DB unique constraint on `short_code` with `INSERT ... ON CONFLICT` to atomically detect and reject collisions.
   - Prevent race conditions by relying on DB-level row/table locking or transactions rather than application-level pre-checks.

   Q7:
   - Run a background worker querying `WHERE expires_at < NOW()` on the indexed `expires_at` column for efficient batch deletion.
   - Supplement with lazy expiration (check on access, mark as stale, and purge in batches) to avoid heavy DB load during peak traffic.

   All answers are exactly 1-2 bullets. They directly address the prompts. Technical accuracy is high. Matches industry standards for URL shortener design. Ready.✅
   Output matches response. 
   One minor tweak: Ensure formatting exactly matches the prompt's numbering/style. I'll keep it clean.
   All good.✅
   Output generation. 
   [Self-Correction/Verification during thought]

## Response


## Metadata
- **Model**: intel-qwen3.6-27b-int4-2k
- **Elapsed**: 42596ms
- **Prompt tokens**: 196
- **Completion tokens**: 2048
- **Total tokens**: 2244
- **Gen tok/s**: 48.0
