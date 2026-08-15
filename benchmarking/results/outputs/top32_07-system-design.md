# Output: top-32 × 07-system-design

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

1.  **Analyze User Request:**
   - Topic: System Design for URL Shortener (like bit.ly)
   - Constraint: Answer each in 1-2 bullet points
   - Questions: 7 specific points covering ID generation, DB schema, redirect flow, caching, scaling, custom aliases, expiration.
   - Goal: Provide concise, high-level, key decisions only.

2.  **Deconstruct Each Question & Formulate Answers:**

   *1. ID generation:*
   - Options: Base62, Hash, Random, Sequential, etc.
   - Key decision: Base62 (or similar alphanumeric encoding) is standard for bit.ly-like systems. Why? It's compact, human-readable, avoids collisions with standard hash outputs, and allows predictable length. Alternatively, could mention a hybrid approach (random + base62) or sequential with base62 encoding.
   - Bullet points: 
     - Use Base62 encoding (0-9, a-z, A-Z) to generate short codes from a sequential/random counter.
     - Why: Compact, human-readable, avoids hash collisions, and allows predictable URL length without truncation.

   *2. Database schema:*
   - Core table: `short_urls` or `links`
   - Columns: `id` (short code), `original_url`, `created_at`, `expires_at`, `custom_alias`, `status`, etc.
   - Key indexes: Primary key on `id`, index on `original_url` (for lookup/redirect), maybe index on `custom_alias` if allowed.
   - Bullet points:
     - Core table: `links` with columns `short_code` (PK), `original_url`, `created_at`, `expires_at`, `custom_alias`.
     - Indexes: Primary key on `short_code`, secondary index on `original_url` for reverse lookup/redirect validation.

   *3. Redirect flow:*
   - 301 vs 302: 301 is permanent, 302 is temporary. For URL shorteners, 302 is preferred because links can be updated/redirected to new URLs, and browsers/cache shouldn't permanently cache the short URL.
   - Cache invalidation: How to handle? Use HTTP headers (Cache-Control, Expires), or invalidate via Redis cache, or use dynamic routing.
   - Bullet

## Metadata
- **Top-k**: 32
- **Elapsed**: 7028ms
- **Prompt tokens**: 196
- **Completion tokens**: 512
- **Total tokens**: 708
