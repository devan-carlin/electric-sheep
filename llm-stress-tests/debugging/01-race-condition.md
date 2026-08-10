# Race Condition Hunt

**Category:** Debugging
**Target:** Concurrency bugs, lock ordering, thread safety

---

## Prompt

The following Python code implements a connection pool for a database client. It has **three race conditions** and **one deadlock scenario**. Find all four bugs, explain why each occurs, and provide a corrected version.

```python
import threading
import time
from collections import deque

class ConnectionPool:
    def __init__(self, max_size=10, timeout=30):
        self.max_size = max_size
        self.timeout = timeout
        self._pool = deque()
        self._in_use = set()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._stats = {
            'created': 0,
            'checked_out': 0,
            'checked_in': 0,
            'expired': 0,
            'errors': 0
        }
        self._closed = False

    def acquire(self, timeout=None):
        timeout = timeout or self.timeout
        start = time.monotonic()

        while True:
            with self._not_empty:
                # Try to get a healthy connection from the pool
                while self._pool:
                    conn = self._pool.popleft()
                    if self._is_healthy(conn):
                        self._in_use.add(conn['id'])
                        self._stats['checked_out'] += 1
                        return conn

                    # Connection is unhealthy, discard it
                    self._stats['expired'] += 1

                # Pool is empty, create new connection if under limit
                if len(self._in_use) + len(self._pool) < self.max_size:
                    conn = self._create_connection()
                    self._in_use.add(conn['id'])
                    self._stats['created'] += 1
                    self._stats['checked_out'] += 1
                    return conn

                # Wait for a connection to be returned
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    raise TimeoutError(f"Could not acquire connection within {timeout}s")

                self._not_empty.wait(timeout=timeout - elapsed)

            # Timeout from wait, retry the loop
            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                raise TimeoutError(f"Could not acquire connection within {timeout}s")

    def release(self, conn):
        conn_id = conn['id']

        # BUG AREA: Check if connection was actually in use
        if conn_id in self._in_use:
            self._in_use.remove(conn_id)

        with self._not_empty:
            if self._is_healthy(conn):
                self._pool.append(conn)
                self._stats['checked_in'] += 1
                self._not_empty.notify()
            else:
                self._stats['expired'] += 1

    def close(self):
        self._closed = True
        with self._not_empty:
            while self._pool:
                conn = self._pool.popleft()
                self._close_connection(conn)
            self._not_empty.notify_all()

    def get_stats(self):
        return self._stats.copy()

    def _is_healthy(self, conn):
        if self._closed:
            return False
        if conn.get('error'):
            return False
        if time.monotonic() - conn.get('created_at', 0) > conn.get('max_lifetime', 3600):
            return False
        return True

    def _create_connection(self):
        return {
            'id': f"conn-{self._stats['created'] + 1}",
            'created_at': time.monotonic(),
            'max_lifetime': 3600,
            'error': None
        }

    def _close_connection(self, conn):
        conn['error'] = 'closed'

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
```

**Requirements:**

1. Identify all four bugs (3 race conditions + 1 deadlock)
2. For each bug:
   - Quote the exact line(s)
   - Explain the scenario that triggers it
   - Show a thread interleaving that demonstrates the bug
3. Provide a corrected version of the entire class
4. Include a test file that uses `threading` to reproduce each bug scenario

**Constraints:**

- Python 3.10+
- Use only stdlib (threading, time, collections)
- The corrected version must pass the test file under `pytest -v`
- Tests should use `threading.Barrier` or `threading.Event` to force specific interleavings

Produce the analysis and corrected code. No placeholders, no TODOs.
