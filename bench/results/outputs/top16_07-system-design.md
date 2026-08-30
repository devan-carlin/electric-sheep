# Output: top-16 × 07-system-design

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
   - **Topic:** System Design - URL Shortener
   - **Constraint:** Answer each in 1-2 bullet points
   - **Questions:**
     1. ID generation: Base62? Hash? Why?
     2. Database schema: Core table? Key indexes?
     3. Redirect flow: 301 vs 302? Why? Cache invalidation?
     4. Caching: Redis content? TTL strategy? Cache misses?
     5. Scaling: 10M redirects/day, 1M creates/day. Read/write ratio? Bottleneck?
     6. Custom aliases: Collisions? Race conditions?
     7. Expiration: Cleanup without full table scan?

2.  **Deconstruct & Draft Answers (Mental Refinement):**
   *Constraint Check:* Each answer must be exactly 1-2 bullet points. I'll structure each response as a single bullet or two concise bullets.

   **1. ID generation:**
   - Base62 encoding of a monotonically increasing integer (or distributed ID generator like Snowflake) ensures uniqueness, compactness, and human-readable alphanumeric codes.
   - Avoids hash collisions and provides predictable, sequential IDs that scale well across distributed systems.

   **2. Database schema:**
   - Core table: `short_urls(id, short_code, original_url, created_at, expires_at, custom_alias, status)`.
   - Indexes: Unique index on `short_code` (or `custom_alias`), and secondary indexes on `created_at`/`expires_at` for expiration cleanup and analytics.

   **3. Redirect flow:**
   - Use 302 (temporary) for standard redirects to allow cache invalidation and dynamic routing changes; use 301 only for permanent/custom links.
   - Cache invalidation via Redis TTL expiration + a background job that updates/invalidates cached entries when the underlying URL or status changes.

   **4. Caching:**
   - Redis stores `short_code → original_url` mappings with a TTL (e.g., 24h) to reduce DB load and latency.
   - Cache misses trigger a DB lookup, then populate Redis; use a write-through or cache-aside pattern with a short TTL to balance freshness and performance.

   **5.

## Metadata
- **Top-k**: 16
- **Elapsed**: 6454ms
- **Prompt tokens**: 196
- **Completion tokens**: 512
- **Total tokens**: 708
