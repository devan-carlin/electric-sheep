# Memory Leak Diagnosis

**Category:** Debugging
**Target:** Profile output interpretation, reference cycles, resource cleanup

---

## Prompt

The following Python code implements an HTTP request cache with LRU eviction. Under sustained load, memory grows unbounded despite the size limit. **There are three sources of memory leaks.** Diagnose each one and provide a corrected version.

```python
import time
import threading
import traceback
from functools import wraps

class RequestCache:
    def __init__(self, max_size=1000, ttl=300):
        self.max_size = max_size
        self.ttl = ttl
        self._cache = {}
        self._access_order = []
        self._lock = threading.RLock()
        self._callbacks = {}
        self._metrics = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'errors': 0
        }

    def get(self, key):
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if time.time() - entry['created_at'] < self.ttl:
                    self._metrics['hits'] += 1
                    self._access_order.move_to_end(key)
                    return entry['value']
                else:
                    del self._cache[key]
                    self._access_order.discard(key)

            self._metrics['misses'] += 1
            return None

    def put(self, key, value, fetch_fn=None):
        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                self._evict_oldest()

            self._cache[key] = {
                'value': value,
                'created_at': time.time(),
                'fetch_fn': fetch_fn,
                'stack': traceback.format_stack()  # For debugging
            }
            self._access_order[key] = time.time()

    def on_evict(self, key, callback):
        if key not in self._callbacks:
            self._callbacks[key] = []
        self._callbacks[key].append(callback)

    def _evict_oldest(self):
        if not self._access_order:
            return

        oldest_key = min(self._access_order, key=self._access_order.get)
        entry = self._cache.pop(oldest_key, None)
        self._access_order.pop(oldest_key, None)
        self._metrics['evictions'] += 1

        # Run eviction callbacks
        for cb in self._callbacks.get(oldest_key, []):
            try:
                cb(oldest_key, entry['value'] if entry else None)
            except Exception:
                self._metrics['errors'] += 1

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._access_order.clear()

    def size(self):
        return len(self._cache)

    def get_metrics(self):
        return dict(self._metrics)


def cached(cache_instance, key_fn=lambda **kw: str(kw)):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = key_fn(*args, **kwargs)
            result = cache_instance.get(key)
            if result is not None:
                return result

            result = fn(*args, **kwargs)
            cache_instance.put(key, result, fetch_fn=fn)
            return result
        return wrapper
    return decorator
```

**Requirements:**

1. Identify all three memory leak sources
2. For each leak:
   - Explain what's accumulating and why it never gets freed
   - Show what happens under sustained load (e.g., 10K unique keys cycling through)
3. Provide a corrected version that:
   - Uses `OrderedDict` for proper LRU ordering
   - Stops storing tracebacks (or limits them)
   - Properly clears callbacks on eviction
   - Uses weak references where appropriate
4. Include a test that verifies memory stays bounded under cycling load

**Constraints:**

- Python 3.10+
- stdlib only
- The corrected version must pass a test that puts/gets 100K keys with max_size=1000 and verifies `len(cache._cache) <= 1000`

Produce the analysis and corrected code. No placeholders, no TODOs.
