# SQL Migration Script

**Target Capability:** Database schema changes, data backfill, rollback planning.

Tests whether a model can write safe, reversible migrations with proper transaction handling.

---

## Prompt

```
Write a PostgreSQL migration script that performs the following changes to an existing `users` table:

Current schema:
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

Required changes:
1. Add a `phone` column (VARCHAR(20), nullable).
2. Add a `status` column (VARCHAR(20), NOT NULL, default 'active', check constraint: 'active', 'suspended', 'deleted').
3. Create a new `user_preferences` table:
   ```sql
   CREATE TABLE user_preferences (
       user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
       theme VARCHAR(20) DEFAULT 'light',
       language VARCHAR(10) DEFAULT 'en',
       notifications_enabled BOOLEAN DEFAULT true,
       updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
   );
   ```
4. Backfill: Insert a row into `user_preferences` for every existing user (with default values).
5. Add an index on `users.status` for filtering.
6. Add an index on `users.email` (if not already present from the UNIQUE constraint).

Requirements:
- Wrap everything in a transaction (BEGIN / COMMIT).
- Use `IF NOT EXISTS` or `DO $$ ... $$` blocks to make the migration idempotent (safe to run twice).
- Include a ROLLBACK section at the bottom (commented out) that reverses all changes.
- Add comments explaining each step.
- The migration should handle the case where `user_preferences` already exists (skip creation and backfill).

Output ONLY the SQL. No explanation.
```
