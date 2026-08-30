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
The user wants to convert a Python HTTP server handler for a key-value store into a Go program.
The Python code uses:
- `http.server` for handling HTTP requests.
- `threading.Lock` for synchronization.
- A dictionary `store` for key-value storage.
- JSON for data serialization.
- Time-based expiration for keys.

The Go requirements are:
- Use `net/http` for the server.
- Use `sync.RWMutex` for synchronization (RLock for reads).
- Handle `GET /{key}`, `PUT /{key}`, `DELETE /{key}`, `GET /health`.
- Proper error handling.
- Produce a complete `main.go`.

Let's break down the conversion:

1. **Server Setup**:
   - Create a `http.Server` or use `http.HandleFunc` and `http.ListenAndServe`.
   - Define a struct to hold the store and the mutex.

2. **Store and Mutex**:
   - `store` can be a `map[string]struct{value string, expiration time.Time}`.
   - `lock` can be a `sync.RWMutex`.

3. **Handler Functions**:
   - `GET /health`: Return `{"status": "ok"}`.
   - `GET /{key}`:
     - Acquire `RLock`.
     - Check if key exists.
     - Check if expired.
     - Return value and TTL.
     - Release `RUnlock`.
   - `PUT /{key}`:
     - Read body.
     - Parse JSON.
     - Acquire `Lock`.
     - Store key with value and expiration.
     - Release `Unlock`.
     - Return `{"key": key}`.
   - `DELETE /{key}`:
     - Acquire `Lock`.
     - Check if key exists.
     - Delete if exists.
     - Return appropriate response.
     - Release `Unlock`.

4. **JSON Handling**:
   - Use `encoding/json` for marshaling and unmarshaling.

5. **Error Handling**:
   - Handle invalid JSON, missing keys, etc.

Let's draft the Go code:

```go
package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

type KVStore

## Metadata
- **Top-k**: 16
- **Elapsed**: 6540ms
- **Prompt tokens**: 474
- **Completion tokens**: 512
- **Total tokens**: 986
