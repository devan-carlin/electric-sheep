# Python → Go Migration

**Category:** Code transformation
**Target:** Cross-language reasoning, idiomatic patterns, error handling

---

## Prompt

Convert the following Python service to Go. The Python code is a simple HTTP service that manages a key-value store with TTL expiration.

**Source (Python):**

```python
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse

store = {}
lock = threading.Lock()

class KVHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        key = parsed.path.strip('/')
        
        if key == 'health':
            self._respond(200, {"status": "ok"})
            return
        
        with lock:
            if key not in store:
                self._respond(404, {"error": "key not found"})
                return
            value, expires_at = store[key]
            if time.time() > expires_at:
                del store[key]
                self._respond(404, {"error": "key expired"})
                return
            self._respond(200, {"key": key, "value": value, "ttl": int(expires_at - time.time())})
    
    def do_PUT(self):
        parsed = urllib.parse.urlparse(self.path)
        key = parsed.path.strip('/')
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._respond(400, {"error": "invalid JSON"})
            return
        
        value = data.get('value')
        ttl = data.get('ttl', 3600)
        
        if value is None:
            self._respond(400, {"error": "missing 'value'"})
            return
        
        with lock:
            store[key] = (value, time.time() + ttl)
        
        self._respond(201, {"key": key, "ttl": ttl})
    
    def do_DELETE(self):
        key = self.path.strip('/')
        with lock:
            if key in store:
                del store[key]
                self._respond(200, {"deleted": key})
            else:
                self._respond(404, {"error": "key not found"})
    
    def _respond(self, code, data):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', 8080), KVHandler)
    print("Serving on :8080")
    server.serve_forever()
```

**Requirements for the Go version:**

- Use `net/http` standard library (no external frameworks)
- Use `sync.RWMutex` for concurrent access (RLock for reads, Lock for writes)
- Proper Go error handling (no panics for client errors)
- Structured logging with `log/slog`
- Graceful shutdown on SIGTERM
- Background goroutine that periodically cleans expired keys
- Command-line flag for port (`-port=8080`)
- Equivalent API: `GET /{key}`, `PUT /{key}` with JSON body, `DELETE /{key}`, `GET /health`
- Include a `go.mod` file
- Include a `main_test.go` with table-driven tests using `httptest`

**Constraints:**

- Go 1.22+
- No external dependencies (stdlib only)
- Must compile with `go build`
- Tests must pass with `go test -v -race`

Produce all files with complete working code. No placeholders, no TODOs.
