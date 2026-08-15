# Output: top-64 × 06-code-transform

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
The Python code describes a key-value store with expiration (ttl).
It uses `HTTPServer` and `BaseHTTPRequestHandler`.
It has `GET`, `PUT`, `DELETE` operations.
It uses a `threading.Lock` (which in Go corresponds to `sync.RWMutex`).
The requirements specify:
- `sync.RWMutex` (RLock for reads).
- proper error handling.
- `GET /{key}`, `PUT /{key}`, `DELETE /{key}`, `GET /health`.
- Produce just the Go code.

I need to write a complete `main.go` in Go.
The structure should be:
- A handler struct.
- A store map.
- A mutex.
- HTTP server setup.
- GET, PUT, DELETE methods.
- JSON response handling.

Let's draft the Go code.

```go
package main

import (
    "net/http"
    "sync"
    "time"
    "encoding/json"
    "fmt"
)

type Handler struct {
    store map[string]struct {
        value string
        exp time.Time
    }
    lock sync.RWMutex
}

func (h *Handler) do_GET(w http.ResponseWriter, r *http.Request) {
    key := r.URL.Path[1:] // strip '/'
    if key == "health" {
        json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
        w.WriteHeader(200)
        return
    }
    h.lock.RLock()
    if key not in store {
        json.NewEncoder(w).Encode(map[string]string{"error": "not found"})
        w.WriteHeader(404)
        h.lock.RUnlock()
        return
    }
    val, exp := store[key]
    if time.Now() > exp {
        h.lock.RUnlock()
        h.lock.Lock()
        delete store[key]
        json.NewEncoder(w).Encode(map[string]string{"error": "expired"})
        w.WriteHeader(404)
        h.lock.Unlock()
        return
    }
    json.NewEncoder(w).Encode(map[string]string{"key": key, "value": val, "ttl": int(exp - time.Now())})

## Metadata
- **Top-k**: 64
- **Elapsed**: 8403ms
- **Prompt tokens**: 474
- **Completion tokens**: 512
- **Total tokens**: 986
