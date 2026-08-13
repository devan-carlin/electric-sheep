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
Thinking Process:

1.  **Deconstruct the Prompt:**
    *   Goal: System Design URL Shortener.
    *   Constraint: Key Decisions Only, 1-2 bullet points per question.
    *   Questions: ID gen, DB schema, Redirect flow, Caching, Scaling, Custom aliases, Expiration.

2.  **Draft Answers per Question (Mental Check against constraints):**

    *   *ID generation:* Base62 vs Hash. Base62 is deterministic and space-efficient. Hash is random but might collide or be longer. Base62 is standard.
        *   *Draft:* Base62 encoding of a monotonically increasing counter. Deterministic, space-efficient, avoids random collision checks.
        *   *Check:* 1 bullet? Yes. Key decision? Yes.

    *   *Database schema:* Core table. Key indexes.
        *   *Draft:* Table `short_url` with columns `id`, `short_code`, `long_url`, `created_at`. PK on `id`. Unique index on `short_code`.
        *   *Check:* 1 bullet? Yes. Key decision? Yes.

    *   *Redirect flow:* 301 vs 302. Cache invalidation.
        *   *Draft:* 302 for analytics/ability to change target; 301 for SEO/permanent. Cache invalidation via write-through or TTL.
        *   *Check:* 1 bullet? Yes. Key decision? Yes.

    *   *Caching:* Redis. TTL. Misses.
        *   *Draft:* Cache `short_code` -> `long_url`. TTL matches link expiration. Misses go to DB, write-through cache.
        *   *Check:* 1 bullet? Yes. Key decision? Yes.

    *   *Scaling:* 10M read, 1M write. Ratio. Bottleneck.
        *   *Draft:* Read-heavy (10:1). Bottleneck is DB read on cache miss. Shard by hash of short_code or use read replicas.
        *   *Check:* 1 bullet? Yes. Key decision? Yes.

    *   *Custom aliases:* Collisions. Race conditions.
        *   *Draft:* Unique index check. Use DB transaction or optimistic locking with retry.
        *   *Check:* 1 bullet? Yes.

## Metadata
- **Top-k**: 16
- **Elapsed**: 6553ms
- **Prompt tokens**: 196
- **Completion tokens**: 512
- **Total tokens**: 708
