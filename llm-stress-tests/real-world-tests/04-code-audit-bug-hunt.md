# Test 4: Code Auditing — Subtle Bug & Race Condition Hunting

**Target Capability:** Code comprehension, logic analysis, restraint (fixing bugs without rewriting working code).

**Why it's a great test:** Generative models tend to completely rewrite code when asked to fix small bugs. Testing whether a model can pinpoint a single subtle issue without breaking surrounding architectural patterns is a great test of true reasoning.

**The Task:** Provide a 50–100 line code snippet that *looks* correct and runs without throwing immediate syntax errors, but contains a subtle issue. Instruct the model: *"Identify the exact bug, explain why it happens, and provide ONLY the refactored function—do not rewrite the rest of the script."*

---

## Prompt

```
The following Python script implements a simple in-memory rate limiter for API requests. It looks correct and runs without syntax errors, but it contains a subtle bug that causes it to fail under concurrent load.

Identify the exact bug, explain WHY it happens, and provide ONLY the refactored function — do not rewrite the rest of the script.

```python
import time
import threading
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        timestamps = self.requests[client_id]

        # Remove expired timestamps
        while timestamps and timestamps[0] < now - self.window_seconds:
            timestamps.pop(0)

        if len(timestamps) < self.max_requests:
            timestamps.append(now)
            return True
        return False

    def get_remaining(self, client_id: str) -> int:
        now = time.time()
        timestamps = self.requests[client_id]

        while timestamps and timestamps[0] < now - self.window_seconds:
            timestamps.pop(0)

        return max(0, self.max_requests - len(timestamps))

# Usage example
limiter = RateLimiter(max_requests=5, window_seconds=60.0)

def handle_request(client_id: str):
    if limiter.is_allowed(client_id):
        print(f"Request allowed for {client_id}. Remaining: {limiter.get_remaining(client_id)}")
    else:
        print(f"Request denied for {client_id}. Rate limit exceeded.")

# Simulate concurrent requests
threads = []
for i in range(20):
    t = threading.Thread(target=handle_request, args=(f"user_{i % 3}",))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
```

Your response must include:
1. The exact bug (name it).
2. A one-sentence explanation of why it fails under concurrency.
3. ONLY the refactored `is_allowed` method (nothing else).
```
