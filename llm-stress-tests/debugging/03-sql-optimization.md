# SQL Query Optimization

**Category:** Debugging
**Target:** EXPLAIN output analysis, index selection, N+1 detection, query rewriting

---

## Prompt

The following Python application uses SQLAlchemy to serve a dashboard API for an e-commerce platform. The dashboard is slow — page loads take 8-12 seconds. **There are five performance problems**: three N+1 queries, one missing index, and one cartesian product.

**Schema:**

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    status TEXT NOT NULL DEFAULT 'pending',
    total_cents INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    shipped_at TIMESTAMP,
    delivered_at TIMESTAMP
);

CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price_cents INTEGER NOT NULL
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price_cents INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE reviews (
    id INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES products(id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
    comment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Slow queries (as written by the developer):**

```python
# Query 1: Dashboard — top 10 orders with user info and items
def get_top_orders(session, limit=10):
    orders = session.execute(
        text("SELECT * FROM orders WHERE status = 'delivered' ORDER BY total_cents DESC LIMIT :limit"),
        {"limit": limit}
    ).fetchall()

    result = []
    for order in orders:
        user = session.execute(
            text("SELECT * FROM users WHERE id = :id"), {"id": order.user_id}
        ).fetchone()

        items = session.execute(
            text("SELECT * FROM order_items WHERE order_id = :id"), {"id": order.id}
        ).fetchall()

        item_details = []
        for item in items:
            product = session.execute(
                text("SELECT * FROM products WHERE id = :id"), {"id": item.product_id}
            ).fetchone()
            item_details.append({
                "product": product.name,
                "quantity": item.quantity,
                "price": item.unit_price_cents
            })

        result.append({
            "order_id": order.id,
            "user": user.name,
            "total": order.total_cents,
            "items": item_details
        })

    return result

# Query 2: Product leaderboard with average ratings
def get_product_leaderboard(session, category=None, limit=20):
    if category:
        products = session.execute(
            text("SELECT * FROM products WHERE category = :cat"), {"cat": category}
        ).fetchall()
    else:
        products = session.execute(
            text("SELECT * FROM products")
        ).fetchall()

    result = []
    for product in products:
        reviews = session.execute(
            text("SELECT * FROM reviews WHERE product_id = :id"), {"id": product.id}
        ).fetchall()

        avg_rating = sum(r.rating for r in reviews) / len(reviews) if reviews else 0
        review_count = len(reviews)

        # Count orders containing this product
        order_count = session.execute(
            text("SELECT COUNT(*) FROM order_items WHERE product_id = :id"),
            {"id": product.id}
        ).fetchone()[0]

        result.append({
            "product": product.name,
            "avg_rating": round(avg_rating, 2),
            "review_count": review_count,
            "order_count": order_count
        })

    result.sort(key=lambda x: x["order_count"], reverse=True)
    return result[:limit]

# Query 3: User activity report
def get_user_activity(session, user_id):
    orders = session.execute(
        text("SELECT * FROM orders WHERE user_id = :uid"), {"uid": user_id}
    ).fetchall()

    all_items = []
    for order in orders:
        items = session.execute(
            text("SELECT oi.*, p.name, p.category FROM order_items oi JOIN products p ON oi.product_id = p.id WHERE oi.order_id = :oid"),
            {"oid": order.id}
        ).fetchall()
        all_items.extend(items)

    return {
        "total_orders": len(orders),
        "total_spent": sum(o.total_cents for o in orders),
        "items_purchased": len(all_items),
        "categories": list(set(i.category for i in all_items))
    }
```

**Requirements:**

1. Identify all five performance problems (label each: N+1, missing index, or cartesian)
2. For each problem:
   - Quote the code
   - Explain the query count vs data size (e.g., "1 + 10 + 10×N queries for top_orders")
   - Show the optimized SQL
3. Provide the corrected Python with:
   - Proper JOINs instead of N+1 loops
   - Recommended CREATE INDEX statements
   - `session.query()` or SQLAlchemy Core instead of raw `text()` where appropriate
4. Include the recommended indexes with justification

**Constraints:**

- SQLite-compatible SQL (the app uses SQLite in dev, PostgreSQL in prod)
- SQLAlchemy 2.0 style
- Must produce equivalent results to the original queries

Produce the analysis and corrected code. No placeholders, no TODOs.
