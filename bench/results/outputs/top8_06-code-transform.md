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
The user wants to convert a Python HTTP key-value store handler to Go.
The Python code implements a simple in-memory key-value store with TTL (Time To Live) support.
It handles GET, PUT, and DELETE requests.
It uses a lock for thread safety.
It has a health check endpoint.

Key features to implement in Go:
1.  Use `net/http` for the server.
2.  Use `sync.RWMutex` for concurrency control. The prompt specifically asks for `sync.RWMutex` and mentions "RLock for reads".
3.  Implement the store logic:
    *   Store: `map[string]struct{value string, expiry time.Time}` (or similar).
    *   TTL logic: Check if current time > expiry time.
4.  Endpoints:
    *   `GET /health`: Returns `{"status": "ok"}`.
    *   `GET /{key}`: Returns value and remaining TTL if exists and not expired. Returns 404 if not found or expired.
    *   `PUT /{key}`: Sets value and TTL (default 3600s). Returns 201.
    *   `DELETE /{key}`: Deletes key. Returns 200 if deleted, 404 if not found.
5.  Error handling: Proper JSON responses for errors.
6.  Input parsing: Read body for PUT requests.

Let's refine the data structure.
The Python code stores `(value, expiry)`.
In Go, a struct is cleaner.
```go
type item struct {
    value  string
    expiry time.Time
}
```

The store will be `map[string]item`.

Concurrency:
The prompt asks for `sync.RWMutex`.
Read operations (GET) should use `RLock`.
Write operations (PUT, DELETE) should use `Lock`.

However, there's a nuance with the TTL check.
In the Python code:
```python
with lock:
    if key not in store:
        ...
    val, exp = store[key]
    if time.time() > exp:
        del store[key]
        ...
```
The deletion of an expired key happens inside the lock. This is important to prevent race conditions where a key expires and is read simultaneously.
So, even for GET,

## Metadata
- **Top-k**: 8
- **Elapsed**: 6249ms
- **Prompt tokens**: 474
- **Completion tokens**: 512
- **Total tokens**: 986
