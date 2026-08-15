# Output: davidau-qwen3.6-27b × 06-code-transform

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
The user wants a Go implementation of a Python key-value store HTTP handler.
Key requirements:
1. Use `net/http` and `sync.RWMutex`.
2. Implement `GET /{key}`, `PUT /{key}`, `DELETE /{key}`, `GET /health`.
3. Use `RLock` for reads (GET), `Lock` for writes (PUT, DELETE).
4. Handle expiration logic (TTL).
5. Proper error handling.
6. Produce a complete `main.go`.

Let's break down the Python code:
- `store` is a dict mapping key -> (value, expiry_time).
- `lock` is a threading.Lock.
- `GET /health`: returns 200 {"status": "ok"}.
- `GET /{key}`:
  - If key not in store: 404 {"error": "not found"}.
  - If expired (time > exp): delete key, return 404 {"error": "expired"}.
  - Else: return 200 {"key": key, "value": val, "ttl": int(exp - time.time())}.
- `PUT /{key}`:
  - Read JSON body: {"value": ..., "ttl": ...} (ttl defaults to 3600).
  - Store key -> (value, time.time() + ttl).
  - Return 201 {"key": key}.
- `DELETE /{key}`:
  - If key in store: delete, return 200 {"deleted": key}.
  - Else: return 404 {"error": "not found"}.

Go implementation details:
- Use `sync.RWMutex`.
- Store type: `map[string]entry` where `entry` is a struct with `Value interface{}` and `Expiry time.Time`.
- Handler: `http.HandlerFunc` or a struct with `ServeHTTP`.
- Use `json.Marshal` for responses.
- Use `json.Unmarshal` for PUT body.
- Handle path parsing: `strings.TrimPrefix(r.URL.Path, "/")`.
- Health endpoint: check path == "health".
- For GET: use `RLock`, check existence, check expiry. If expired, need to upgrade to write lock or handle carefully. Since we need to delete on expiry, we should probably use

## Metadata
- **Model**: davidau-qwen3.6-27b
- **Elapsed**: 16355ms
- **Prompt tokens**: 474
- **Completion tokens**: 512
- **Total tokens**: 986
- **Gen tok/s**: 31.3
