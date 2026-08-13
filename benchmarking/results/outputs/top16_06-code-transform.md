# Output: top-16 × 06-code-transform

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
    *   Methods: `do_GET`, `do_PUT`, `do_DELETE`, `_json`.
    *   Logic:
        *   `do_GET`: Path -> key. `health` check. Lock. Check existence. Check TTL expiry. Return value or error.
        *   `do_PUT`: Path -> key. Read body. Parse JSON. Lock. Insert with TTL. Return success.
        *   `do_DELETE`: Path -> key. Lock. Check existence. Delete. Return success or error.
        *   `_json`: Helper to send response with JSON body.

2.  **Map to Go constructs:**
    *   `net/http` server: `http.HandleFunc`, `http.ListenAndServe`.
    *   State: `map[string]Item`. Item struct: `{Value string, Expire time.Time}`.
    *   Lock: `sync.RWMutex`.
    *   *Crucial Requirement:* "RLock for reads". Python uses a single `Lock` for everything. Go allows `RLock` for reads. This is a semantic improvement requested by the prompt.
    *   JSON: `encoding/json`.
    *   Time: `time`.
    *   Error handling: Go returns errors, doesn't raise exceptions. Need to check `err != nil`.

3.  **Draft the Go code structure:**
    *   `package main`.
    *   Imports: `net/http`, `sync`, `time`, `encoding/json`, `log`, `fmt`.
    *   Structs: `Item`, `KVStore`.
    *   `KVStore` holds `mu sync.RWMutex` and `items map[string]Item`.
    *   Functions: `Get`, `Put`, `Delete`, `Health`.
    *   `main`: Setup mux, listen.

4.  **Implement Logic with RWMutex distinction:**
    *   *Reads (GET, Health):* Use `mu.RLock()`. `mu.RUnlock()`.
    *   *Writes (PUT, DELETE):* Use `mu.Lock()

## Metadata
- **Top-k**: 16
- **Elapsed**: 6624ms
- **Prompt tokens**: 474
- **Completion tokens**: 512
- **Total tokens**: 986
