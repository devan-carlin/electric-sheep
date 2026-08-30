# Output: top-32 × 06-code-transform

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
The user wants to convert a Python HTTP server handler into a Go `main.go` file.
The Python code implements a key-value store with expiration (TTL).
It uses `HTTPServer` and `BaseHTTPRequestHandler`.
It has `GET`, `PUT`, `DELETE` methods.
It uses a `threading.Lock` for synchronization.
The requirements specify using `sync.RWMutex` (RLock for reads).
Proper error handling is needed.
Endpoints: `GET /{key}`, `PUT /{key}`, `DELETE /{key}`, `GET /health`.
Output should be just the Go code.

Let's analyze the Python code:
- `store = {}`: A dictionary storing key-value pairs with expiration time.
- `lock = threading.Lock()`: A lock for synchronization.
- `Handler`: A request handler.
- `do_GET`:
  - If `key == 'health'`, return 200 with `{"status": "ok"}`.
  - Otherwise, acquire lock.
  - If `key` not in `store`, return 404 with `{"error": "not found"}`.
  - If `key` in `store`, get `val` and `exp`.
  - If `time.time() > exp`, delete `key` from `store`, return 404 with `{"error": "expired"}`.
  - Otherwise, return 200 with `{"key": key, "value": val, "ttl": int(exp - time.time())}`.
- `do_PUT`:
  - Acquire lock.
  - Read JSON data from request body.
  - Store `key` with `(data['value'], time.time() + data.get('ttl', 3600))`.
  - Return 201 with `{"key": key}`.
- `do_DELETE`:
  - Acquire lock.
  - If `key` in `store`, delete `key`, return 200 with `{"deleted": key}`.
  - Otherwise, return 404 with `{"error": "not found"}`.
- `_json`: Helper to send JSON response.

Go implementation:
- Use `net/http` for server.
- Use `sync.RWMutex` for synchronization.
- Store data in a map.
-

## Metadata
- **Top-k**: 32
- **Elapsed**: 7105ms
- **Prompt tokens**: 474
- **Completion tokens**: 512
- **Total tokens**: 986
