# Output: top-8 × 06-code-transform

## Prompt
# Code Transform — Python to Go

Convert this Python key-value store handler to Go. Produce a complete `main.go` using `net/http` and `sync.RWMutex`.

```python
import time, threading, json
from http.server import HTTPServer, BaseHTTPRequestHandler

store = {}
lock = threading.Lock()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        key = self.path.strip('/')
        if key == 'health':
            return self._json(200, {"status": "ok"})
        with lock:
            if key not in store:
                return self._json(404, {"error": "not found"})
            val, exp = store[key]
            if time.time() > exp:
                del store[key]
                return self._json(404, {"error": "expired"})
            self._json(200, {"key": key, "value": val, "ttl": int(exp - time.time())})

    def do_PUT(self):
        key = self.path.strip('/')
        data = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
        with lock:
            store[key] = (data['value'], time.time() + data.get('ttl', 3600))
        self._json(201, {"key": key})

    def do_DELETE(self):
        key = self.path.strip('/')
        with lock:
            if key in store:
                del store[key]
                return self._json(200, {"deleted": key})
        self._json(404, {"error": "not found"})

    def _json(self, code, data):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
```

Requirements: `sync.RWMutex` (RLock for reads), proper error handling, `GET /{key}`, `PUT /{key}`, `DELETE /{key}`, `GET /health`. Produce just the Go code.

## Response
Thinking Process:

1.  **Deconstruct the Python code:**
    *   Imports: `time`, `threading`, `json`, `http.server`.
    *   Global state: `store = {}`, `lock = threading.Lock()`.
    *   Handler class: `BaseHTTPRequestHandler`.
    *   `do_GET`:
        *   Path parsing: `self.path.strip('/')`.
        *   Special case: `health` -> 200 `{"status": "ok"}`.
        *   Locking: `with lock`.
        *   Logic: Check existence, check TTL, return value or 404.
        *   Response: `_json` helper.
    *   `do_PUT`:
        *   Path parsing.
        *   Read body: `json.loads`.
        *   Locking: `with lock`.
        *   Logic: Write to store with TTL.
        *   Response: 201.
    *   `do_DELETE`:
        *   Path parsing.
        *   Locking: `with lock`.
        *   Logic: Delete if exists, return 200 or 404.
    *   `_json` helper: Send response code, headers, body.

2.  **Map to Go constructs:**
    *   `net/http` server: `http.HandleFunc`, `http.ListenAndServe`.
    *   Global state: `mu sync.RWMutex`, `store map[string]item`.
    *   Item structure: `type item struct { value string; exp int64 }`.
    *   `sync.RWMutex`: `RLock`/`RUnlock` for reads, `Lock`/`Unlock` for writes.
    *   *Requirement Check:* The prompt asks for `sync.RWMutex` and mentions "RLock for reads". The Python code uses a single `Lock` for everything. Go allows finer granularity. I should use `RLock` for GET and `Lock` for PUT/DELETE.
    *   Path parsing: `strings.TrimPrefix(r.URL.Path, "/")`.
    *   JSON: `json.Marshal`, `json.Unmarshal`.
    *   Time: `time.Now().Unix()`.
    *   Error handling: Check errors from `json.Unmarshal`, `io.ReadAll`, etc.

3.  **Drafting

## Metadata
- **Top-k**: 8
- **Elapsed**: 6085ms
- **Prompt tokens**: 474
- **Completion tokens**: 512
- **Total tokens**: 986
