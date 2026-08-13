# Output: top-8 × 07-system-design

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
Here's a thinking process:

1.  **Analyze User Input:**
   - **Topic:** System Design — URL Shortener
   - **Constraint:** Answer each in 1-2 bullet points
   - **Focus:** Key Decisions Only
   - **Questions:** 7 specific questions covering ID generation, DB schema, redirect flow, caching, scaling, custom aliases, expiration.
   - **Tone/Format:** Concise, direct, bullet-pointed per question.

2.  **Deconstruct Each Question & Formulate Answers (Mental Draft):**
   - *Q1: ID generation: Base62? Hash? Why?*
     - Base62 encoding of auto-incrementing IDs or distributed ID generators (Snowflake) is preferred over hashing because it's deterministic, avoids collisions, and produces uniformly distributed short codes. Hashing causes collision storms and non-uniform distribution.
   - *Q2: DB schema: Core table? Key indexes?*
     - Table: `id (PK), original_url, short_code (UNIQUE), created_at, expires_at, click_count`. Index on `short_code` for O(1) lookups; index on `original_url` optional for deduplication.
   - *Q3: Redirect flow: 301 vs 302? Cache invalidation?*
     - Use 302 (temporary) by default to preserve SEO and allow dynamic routing/updates; 301 only if explicitly requested. Cache invalidation via write-through or cache-aside with explicit TTL/purge on update.
   - *Q4: Caching: What in Redis? TTL? Misses?*
     - Cache the full redirect response (original URL + status) keyed by short code. TTL matches link expiration or a fixed short window (e.g., 24h). On miss, fetch from DB, populate cache, return.
   - *Q5: Scaling: 10M reads/day, 1M writes/day. Ratio? Bottleneck?*
     - Read/write ratio ~10:1. Bottleneck is the read path (DB lookups); solved by caching and read replicas. Write path is lightweight and scales horizontally.
   - *Q6: Custom aliases: Collisions? Race conditions?*
     - Check uniqueness via DB UNIQUE constraint; retry on collision. Use optimistic locking or `INSERT ... ON CONFLICT DO NOTHING` to serialize race

## Metadata
- **Top-k**: 8
- **Elapsed**: 6100ms
- **Prompt tokens**: 196
- **Completion tokens**: 512
- **Total tokens**: 708
