# Output: davidau-qwen3.6-27b × 07-system-design

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
Here's a thinking process that leads to the suggested response:

1.  **Analyze the Request:**
    *   **Topic:** System Design — URL Shortener.
    *   **Constraint:** Key Decisions Only.
    *   **Format:** Answer each of the 7 specific questions in 1-2 bullet points.
    *   **Tone:** Concise, technical, decision-oriented.

2.  **Drafting Responses (Iterative Refinement):**

    *   **1. ID generation:**
        *   *Options:* Hash (MD5/SHA), Random, Auto-increment + Base62.
        *   *Decision:* Auto-increment + Base62 is standard for predictability and collision avoidance. Random is good for security but harder to manage collisions. Hash is too long.
        *   *Draft:* Use a distributed ID generator (like Snowflake) or a centralized auto-incrementing integer converted to Base62. This ensures uniqueness, avoids collisions, and keeps codes short (6-7 chars).
        *   *Refinement (1-2 bullets):*
            *   Use a centralized auto-incrementing integer (or distributed ID generator like Snowflake) converted to Base62 to guarantee uniqueness and predictable length (6-7 chars).
            *   Avoid hashing (too long) or pure random generation (collision handling overhead) unless security against enumeration is a strict requirement.

    *   **2. Database schema:**
        *   *Core:* `short_code` -> `long_url`.
        *   *Indexes:* Primary key on `short_code`.
        *   *Draft:* Table: `urls(short_code PK, long_url, created_at, user_id)`. Index on `short_code`.
        *   *Refinement (1-2 bullets):*
            *   Core table: `urls(short_code VARCHAR(10) PRIMARY KEY, long_url TEXT, created_at TIMESTAMP, user_id INT)`.
            *   Indexing: Primary key on `short_code` for O(1) lookups; secondary index on `user_id` for analytics/history.

    *   **3. Redirect flow:**
        *   *301 vs 302:* 301 is permanent (cached aggressively), 302 is temporary.
        *   *Decision:* 301 is usually preferred for performance, but 

## Metadata
- **Model**: davidau-qwen3.6-27b
- **Elapsed**: 16232ms
- **Prompt tokens**: 196
- **Completion tokens**: 512
- **Total tokens**: 708
- **Gen tok/s**: 31.5
