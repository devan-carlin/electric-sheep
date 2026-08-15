# Output: unsloth-qwen3.6-27b × 07-system-design

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
   - **Format:** Key Decisions Only, 1-2 bullet points per question
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
   - Need deterministic, collision-resistant, short codes. Base62 encoding of auto-incrementing IDs (or distributed ID generator like Snowflake) is standard.
   - Why? Predictable length, no collisions, easy to decode, avoids hash collisions and expensive lookups. Hashes (MD5/SHA) are longer, non-deterministic, and require collision handling.
   - *Draft bullets:* 
     - Use Base62 encoding of a distributed auto-incrementing ID (e.g., Snowflake or DB sequence) for deterministic, collision-free, fixed-length codes.
     - Avoid cryptographic hashes due to collision risks, longer lengths, and inability to reverse-map without storage.

   *2. Database schema:*
   - Core table: `urls(short_code PK, original_url, created_by, created_at, expires_at, click_count)`
   - Indexes: Primary key on `short_code`, secondary index on `created_by` or `expires_at` if needed. Maybe composite index for expiration queries.
   - *Draft bullets:*
     - Core table: `urls(short_code VARCHAR(7) PK, original_url TEXT, created_by UUID, created_at TIMESTAMP, expires_at TIMESTAMP, click_count BIGINT)`.
     - Indexes: Primary key on `short_code` for O(1) lookups; secondary index on `expires_at` for efficient expiration cleanup.

   *3. Redirect flow:*
   - 301 vs 

## Metadata
- **Model**: unsloth-qwen3.6-27b
- **Elapsed**: 16241ms
- **Prompt tokens**: 196
- **Completion tokens**: 512
- **Total tokens**: 708
- **Gen tok/s**: 31.5
