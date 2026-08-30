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
   - **Questions:**
     1. ID generation: How? Base62? Hash? Why?
     2. Database schema: Core table? Key indexes?
     3. Redirect flow: 301 vs 302? Why? Cache invalidation?
     4. Caching: What in Redis? TTL strategy? Cache miss handling?
     5. Scaling: 10M redirects/day, 1M creates/day. Read/write ratio? Bottleneck?
     6. Custom aliases: Collisions? Race conditions?
     7. Expiration: Clean up without full table scan?

2.  **Deconstruct & Draft Answers (Mental Refinement):**
   *Constraint Check:* Each answer must be 1-2 bullet points. I'll aim for concise, direct answers.

   **1. ID generation:**
   - Use a distributed ID generator (e.g., Twitter Snowflake) to create unique, time-sortable numeric IDs, then encode them in Base62 for URL-friendly short codes.
   - Avoids hash collisions, ensures deterministic generation, and allows easy decoding back to the original ID for DB lookups.

   **2. Database schema:**
   - Core table: `short_urls(id BIGINT PRIMARY KEY, original_url VARCHAR, short_code VARCHAR UNIQUE, created_at TIMESTAMP, expires_at TIMESTAMP, click_count INT)`.
   - Indexes: `short_code` (unique, for fast redirects), `created_at`/`expires_at` (for expiration cleanup), and optionally `original_url` (for duplicate detection).

   **3. Redirect flow:**
   - Use 302 (temporary) for standard redirects to allow cache invalidation and dynamic routing changes; use 301 (permanent) only for trusted/long-term links to leverage browser caching.
   - Cache invalidation: Set short TTLs on cached redirects, use Redis pub/sub or versioned keys to force stale cache drops when the target URL changes.

   **4. Caching:**
   - Cache the `short_code → original_url` mapping in Redis with a short TTL (e.g., 1-24 hours) to handle high read traffic.

## Metadata
- **Top-k**: 8
- **Elapsed**: 6114ms
- **Prompt tokens**: 196
- **Completion tokens**: 512
- **Total tokens**: 708
