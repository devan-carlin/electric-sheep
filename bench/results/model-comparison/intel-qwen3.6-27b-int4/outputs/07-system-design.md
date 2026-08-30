# Output: intel-qwen3.6-27b-int4 × 07-system-design

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
   - **Topic:** System Design — URL Shortener (Key Decisions Only)
   - **Format:** Answer each of 7 questions in 1-2 bullet points
   - **Questions:**
     1. ID generation: Base62? Hash? Why?
     2. Database schema: Core table? Key indexes?
     3. Redirect flow: 301 vs 302? Why? Cache invalidation?
     4. Caching: What in Redis? TTL strategy? Cache misses?
     5. Scaling: 10M redirects/day, 1M creates/day. Read/write ratio? Bottleneck?
     6. Custom aliases: Collisions? Race conditions?
     7. Expiration: Cleanup without full scan?

2.  **Deconstruct & Answer Each Question (Mental Draft):**

   *1. ID generation:*
   - Need short, unique, non-sequential (to prevent enumeration), and efficient.
   - Base62 encoding of a distributed ID (e.g., Snowflake) or hash (e.g., SHA256 truncated) + collision handling.
   - Why Base62? Maximizes character density (a-z, A-Z, 0-9) for shorter codes; deterministic mapping from long ID.
   - *Draft:* Use Base62 encoding of a distributed unique ID (e.g., Snowflake) for deterministic, collision-free short codes; avoids sequential enumeration and maximizes character density for shorter URLs.

   *2. Database schema:*
   - Core table: `short_urls` with columns: `short_code` (PK), `original_url`, `created_by`, `created_at`, `expires_at`, `click_count`.
   - Indexes: Primary key on `short_code`, index on `created_by` for user lookups, maybe index on `expires_at` for cleanup.
   - *Draft:* Core table: `short_urls(short_code PK, original_url, created_by, created_at, expires_at, click_count)`; indexes on `short_code` (PK) for O(1) lookups and `created_by` for user history.

   *3. Redirect flow:*
   - 301 (permanent) vs 302 (temporary). 302 is better for URL

## Metadata
- **Model**: intel-qwen3.6-27b-int4
- **Elapsed**: 10083ms
- **Prompt tokens**: 196
- **Completion tokens**: 512
- **Total tokens**: 708
- **Gen tok/s**: 50.7
