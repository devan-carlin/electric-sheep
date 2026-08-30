# Output: unsloth-qwen3.6-27b × 06-code-transform

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
Here's a thinking process that leads to the suggested Go code:

1.  **Understand the Goal:** The objective is to translate a specific Python HTTP server (a key-value store with TTL support) into Go. The Go implementation must use `net/http`, `sync.RWMutex` (specifically using `RLock` for reads), handle specific HTTP methods (`GET`, `PUT`, `DELETE`), and include a health check endpoint.

2.  **Analyze the Python Code:**
    *   **State:** A global dictionary `store` mapping keys to tuples `(value, expiration_time)`.
    *   **Concurrency:** A global `threading.Lock()` protects the store.
    *   **Endpoints:**
        *   `GET /health`: Returns `{"status": "ok"}`.
        *   `GET /{key}`: Retrieves value. Checks expiration. If expired, deletes and returns 404. If missing, returns 404. If valid, returns value and remaining TTL.
        *   `PUT /{key}`: Stores value. Reads JSON body. Sets expiration (default 3600s). Returns 201.
        *   `DELETE /{key}`: Removes key. Returns 200 if found, 404 if not.
    *   **Helper:** `_json` sends response headers and JSON body.

3.  **Map to Go Concepts:**
    *   **State:** A `map[string]Entry` struct. The `Entry` struct needs to hold the value and the expiration timestamp.
    *   **Concurrency:** `sync.RWMutex`.
        *   *Reads (GET):* Use `mu.RLock()`.
        *   *Writes (PUT, DELETE, Expired cleanup):* Use `mu.Lock()`.
    *   **HTTP Server:** `http.HandleFunc` or `http.NewServeMux`.
    *   **JSON:** `encoding/json` package.
    *   **Time:** `time.Now()`, `time.Unix()`, etc.

4.  **Drafting the Data Structures:**

    ```go
    type Entry struct {
        Value     interface{} // Python used generic values, interface{} is Go equivalent
        ExpiresAt time.Time
    }

    var store = make(map[string]Entry)
    var mu sync.RWMutex

## Metadata
- **Model**: unsloth-qwen3.6-27b
- **Elapsed**: 16397ms
- **Prompt tokens**: 474
- **Completion tokens**: 512
- **Total tokens**: 986
- **Gen tok/s**: 31.2
