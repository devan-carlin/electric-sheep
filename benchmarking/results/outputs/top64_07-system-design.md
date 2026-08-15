# Output: top-64 × 07-system-design

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

1.  **Analyze the User Request:**
   - The user wants a system design for a URL shortener (like bit.ly).
   - They specifically ask for "Key Decisions Only" and want each answer in 1-2 bullet points.
   - The questions are numbered 1 to 7, covering specific aspects: ID generation, Database schema, Redirect flow, Caching, Scaling, Custom aliases, Expiration.

2.  **Deconstruct each question & formulate concise answers (1-2 bullets):**
   - **1. ID generation:** How do you create short codes? Base62? Hash? Why?
   - *Decision:* Base62 ( alphanumeric encoding) is standard for bit.ly. Why: It maximizes space efficiency (62 chars vs 16 hex), avoids hash collisions, and is deterministic/ predictable for user/custom aliases.
   - *Bullet 1:* Use Base62 encoding (0-9, A-Z, a-z) to maximize space efficiency and avoid hash collisions.
   - *Bullet 2:* Deterministic generation allows predictable custom aliases; hash-based would be unpredictable and prone to collisions.

   - **2. Database schema:** What's the core table? Key indexes?
   - *Decision:* Core table: `short_url` (id, original_url, created_at, expires_at, status). Key indexes: Primary key on `id`, secondary index on `original_url` (for lookup/verification), maybe index on `expires_at` for cleanup.
   - *Bullet 1:* Core table: `short_url` with columns `id`, `original_url`, `created_at`, `expires_at`, `status`.
   - *Bullet 2:* Primary index on `id` for fast lookup; secondary index on `original_url` for verification/anti-duplicate checks.

   - **3. Redirect flow:** 301 vs 302? Why? How do you handle cache invalidation?
   - *Decision:* 301 is permanent redirect (SEO/long-term). 302 is temporary (for testing/short-lived). Why: 301 preserves SEO value and avoids browser cache issues; 302 allows quick invalidation. Cache invalidation: Update DB status/expire flag, invalidate Redis cache via TTL or explicit purge.
   - *Bullet 1:* Use 30

## Metadata
- **Top-k**: 64
- **Elapsed**: 8258ms
- **Prompt tokens**: 196
- **Completion tokens**: 512
- **Total tokens**: 708
