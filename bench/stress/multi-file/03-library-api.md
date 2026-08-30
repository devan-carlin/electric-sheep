# Library with Public API

**Category:** Multi-file project
**Target:** Module boundaries, type safety, documentation, versioning

---

## Prompt

Build a Python library called `retrykit` that provides a robust retry/decorator framework. Structure:

```
retrykit/
├── src/
│   └── retrykit/
│       ├── __init__.py          # Public API surface
│       ├── core.py              # RetryEngine, execution logic
│       ├── strategies.py        # Backoff strategies (fixed, exponential, jitter)
│       ├── predicates.py        # Should-retry predicates (exception, return value)
│       ├── decorators.py        # @retry decorator with full options
│       ├── async_.py            # Async equivalents
│       ├── exceptions.py        # RetryExhausted, RetryCancelled
│       └── typing.py            # Type aliases, Protocol definitions
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   ├── test_strategies.py
│   ├── test_decorators.py
│   └── test_async.py
├── docs/
│   └── api.md                   # API reference
├── pyproject.toml
├── CHANGELOG.md
└── README.md
```

**Public API:**

```python
from retrykit import retry, ExponentialBackoff, RetryExhausted

@retry(
    max_attempts=5,
    backoff=ExponentialBackoff(base=0.1, max_delay=30),
    on=(ConnectionError, TimeoutError),
    jitter=True,
)
def fetch(url: str) -> bytes:
    ...
```

**Requirements:**

- Full type annotations with `typing` module (no `Any` unless justified)
- Docstrings on every public function/class (Google style)
- Both sync and async support
- Configurable backoff: fixed, exponential, exponential with jitter, Fibonacci
- Predicates for: exception type, return value check, custom callable
- Circuit breaker pattern (optional, after N consecutive failures)
- Comprehensive test suite with mocked time for backoff verification
- `CHANGELOG.md` with semantic versioning (start at 0.1.0)
- `pyproject.toml` with proper package configuration

**Constraints:**

- Python 3.10+ only
- Zero external dependencies (stdlib only)
- All tests must pass with `pytest`
- `mypy --strict` must pass with no errors

Produce all files with complete working code. No placeholders, no TODOs.
