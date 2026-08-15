# Output: intel-qwen3.6-27b-int4 × 06-code-transform

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
Here's a thinking process:

1.  **Understand the Goal**: The user wants to convert a Python HTTP key-value store handler to Go. The Go code should be a complete `main.go` file using `net/http` and `sync.RWMutex`. It needs to handle `GET /{key}`, `PUT /{key}`, `DELETE /{key}`, and `GET /health`. It must use `sync.RWMutex` with `RLock` for reads, handle errors properly, and follow the behavior of the Python code.

2.  **Analyze the Python Code**:
   - **State**: `store = {}` (dict mapping key to tuple `(value, expiration_time)`)
   - **Lock**: `threading.Lock()` (used for both reads and writes in Python, but the prompt specifically asks for `sync.RWMutex` with `RLock` for reads)
   - **GET `/health`**: Returns `{"status": "ok"}` with 200
   - **GET `/{key}`**: 
     - Strips leading/trailing slashes
     - Checks if key exists
     - If not, returns 404 `{"error": "not found"}`
     - If exists, checks expiration (`time.time() > exp`)
     - If expired, deletes key, returns 404 `{"error": "expired"}`
     - If valid, returns 200 `{"key": key, "value": val, "ttl": int(exp - time.time())}`
   - **PUT `/{key}`**:
     - Strips slashes
     - Reads JSON body: `{"value": ..., "ttl": ...}` (ttl defaults to 3600)
     - Stores `(value, time.time() + ttl)`
     - Returns 201 `{"key": key}`
   - **DELETE `/{key}`**:
     - Strips slashes
     - If key exists, deletes it, returns 200 `{"deleted": key}`
     - Else returns 404 `{"error": "not found"}`
   - **Helper `_json`**: Sets status code, content-type to `application/json`, writes JSON body.

3.  **Map to Go**:
   - **State**: `store map[string]entry` where `entry struct { Value interface{}; ExpiresAt float64

## Metadata
- **Model**: intel-qwen3.6-27b-int4
- **Elapsed**: 10210ms
- **Prompt tokens**: 474
- **Completion tokens**: 512
- **Total tokens**: 986
- **Gen tok/s**: 50.1
