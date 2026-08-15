# Output: davidau-qwen3.6-40b × 06-code-transform

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
The user wants me to convert a Python key-value store HTTP handler to Go. Let me analyze the Python code:

1. It's an HTTP server with a key-value store
2. Uses a threading lock for synchronization
3. Supports GET, PUT, DELETE operations
4. Keys have TTL (time-to-live) expiration
5. Has a health check endpoint

Requirements for Go version:
- Use `sync.RWMutex` with RLock for reads
- Proper error handling
- Routes: GET /{key}, PUT /{key}, DELETE /{key}, GET /health
- Complete main.go file

Let me write the Go code:

```go
package main

import (
	"encoding/json"
	"net/http"
	"strings"
	"sync"
	"time"
)

type Entry struct {
	Value interface{}
	Exp   time.Time
}

type Store struct {
	mu    sync.RWMutex
	data map[string]Entry
}

func NewStore() *Store {
	return &Store{
		data: make(map[string]Entry),
	}
}

func (s *Store) Get(key string) (interface{}, time.Duration, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	entry, ok := s.data[key]
	if !ok {
		return nil, 0, http.ErrNotFound
	}
	
	now := time.Now()
	if now.After(entry.Exp) {
		// Need to delete expired entry - this requires write lock
		// We'll handle this differently - either return expired or upgrade lock
		// For simplicity, let's return expired and let caller handle
		return nil, 0, http.StatusText(http.StatusNotFound) // "expired"
	}
	
	return entry.Value, entry.Exp.Sub(now).Seconds(), nil
}

// Actually, the Python code deletes expired entries on read. Let me reconsider.
// I'll use a write lock for deletion but read lock for reading.
// Or I can just use write lock throughout for Get if expiration check might lead to deletion.

// Let me redo this more carefully.

func (s *Store) Get(key string) (interface{}, time.Duration, error) {
	s.mu.RLock()
	entry, ok := s.data[key]
	if !ok {
		s.mu.RUnlock()
		return nil, 0,

## Metadata
- **Model**: davidau-qwen3.6-40b
- **Elapsed**: 24376ms
- **Prompt tokens**: 474
- **Completion tokens**: 512
- **Total tokens**: 986
- **Gen tok/s**: 21.0
