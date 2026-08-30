# Output: intel-qwen3.6-27b-int4-16k × 06-code-transform

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

## Reasoning
The user wants to convert a Python key-value store handler to Go.
The Python code uses:
- `http.server` for HTTP handling
- `threading.Lock` for synchronization
- A dictionary `store` mapping keys to `(value, expiry_time)` tuples
- JSON serialization/deserialization
- HTTP methods: GET, PUT, DELETE
- Special endpoint: `/health`

Requirements for Go:
- Use `net/http`
- Use `sync.RWMutex` (with RLock for reads)
- Proper error handling
- Endpoints: `GET /{key}`, `PUT /{key}`, `DELETE /{key}`, `GET /health`
- Produce just the Go code in a complete `main.go`

Let's design the Go code:
1. Define a struct for the store entry: `type Entry struct { Value interface{}; Expiry time.Time }` or similar. Actually, the Python code stores `(data['value'], time.time() + data.get('ttl', 3600))`. So value can be any JSON value. I'll use `interface{}` or `json.RawMessage`. Let's use `interface{}` for simplicity, or `json.RawMessage` to preserve exact JSON. The Python code stores whatever `data['value']` is. I'll stick to `interface{}`.
2. Use `sync.RWMutex` to protect the map.
3. Create an HTTP handler that routes based on method and path.
4. Implement `GET`, `PUT`, `DELETE`.
5. Handle `/health` separately.
6. Use `time.Now()` and `time.Now().Add()` for expiry.
7. JSON marshaling/unmarshaling with `encoding/json`.

Let's draft the code structure:

```go
package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"
)

type Entry struct {
	Value  interface{}
	Expiry time.Time
}

var (
	store = make(map[string]Entry)
	mu    sync.RWMutex
)

func handler(w http.ResponseWriter, r *http.Request) {
	key := strings.TrimPrefix(r.URL.Path, "/")
	
	if key == "health" {
		if r.Method != http.MethodGet {
			http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
			return
		}
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
		return
	}

	if key == "" {
		http.Error(w, "Bad Request", http.StatusBadRequest)
		return
	}

	switch r.Method {
	case http.MethodGet:
		handleGet(w, r, key)
	case http.MethodPut:
		handlePut(w, r, key)
	case http.MethodDelete:
		handleDelete(w, r, key)
	default:
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
	}
}

func handleGet(w http.ResponseWriter, r *http.Request, key string) {
	mu.RLock()
	entry, exists := store[key]
	mu.RUnlock()

	if !exists {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
		return
	}

	if time.Now().After(entry.Expiry) {
		mu.Lock()
		delete(store, key)
		mu.Unlock()
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "expired"})
		return
	}

	ttl := int(time.Until(entry.Expiry).Seconds())
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"key":   key,
		"value": entry.Value,
		"ttl":   ttl,
	})
}

func handlePut(w http.ResponseWriter, r *http.Request, key string) {
	var data map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&data); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON"})
		return
	}

	value, ok := data["value"]
	if !ok {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "missing value"})
		return
	}

	ttl := 3600.0
	if t, ok := data["ttl"]; ok {
		if f, ok := t.(float64); ok {
			ttl = f
		}
	}

	mu.Lock()
	store[key] = Entry{
		Value:  value,
		Expiry: time.Now().Add(time.Duration(ttl) * time.Second),
	}
	mu.Unlock()

	writeJSON(w, http.StatusCreated, map[string]string{"key": key})
}

func handleDelete(w http.ResponseWriter, r *http.Request, key string) {
	mu.Lock()
	if _, exists := store[key]; exists {
		delete(store, key)
		mu.Unlock()
		writeJSON(w, http.StatusOK, map[string]string{"deleted": key})
		return
	}
	mu.Unlock()

	writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
}

func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func main() {
	http.HandleFunc("/", handler)
	fmt.Println("Server starting on :8080")
	http.ListenAndServe(":8080", nil)
}
```

Wait, let's check the Python code carefully:
- `do_GET`: strips `/`, checks `health`, then locks, checks existence, checks expiry, deletes if expired, returns 404 expired, else returns 200 with key, value, ttl.
- `do_PUT`: strips `/`, reads body, parses JSON, locks, stores `(value, time.time() + ttl)`, returns 201.
- `do_DELETE`: strips `/`, locks, checks existence, deletes, returns 200, else returns 404.
- `_json`: sends response code, content-type, headers, writes JSON.

My Go code matches this logic.
One detail: `time.Until(entry.Expiry).Seconds()` might be slightly different from `exp - time.time()` due to floating point, but it's fine. I'll use `int(time.Until(entry.Expiry).Seconds())`.
Also, in Python, `data.get('ttl', 3600)` defaults to 3600. My Go code handles that.
The Python code uses `self.rfile.read(int(self.headers.get('Content-Length', 0)))`. In Go, `json.NewDecoder(r.Body).Decode(&data)` handles it automatically.
I should make sure the JSON output matches exactly. Python's `json.dumps` doesn't add a newline, but Go's `json.NewEncoder(w).Encode(data)` adds a newline. That's usually fine, but to be precise, I can use `json.Marshal` and `w.Write`. Let's stick to `json.NewEncoder` or `json.Marshal`. I'll use `json.Marshal` to avoid trailing newline if needed, but it's minor. I'll just use `json.NewEncoder` for simplicity, or `json.Marshal` to be safe. Actually, `json.NewEncoder` is standard. I'll keep it.

Wait, the Python code for GET returns `{"key": key, "value": val, "ttl": int(exp - time.time())}`.
My Go code returns the same.

One thing: `sync.RWMutex` requirement says "RLock for reads". I used `mu.RLock()` in `handleGet` and `handleDelete` (for the check). But in `handleGet`, if expired, I need to delete it, which requires a write lock. I handled that by releasing RLock, then acquiring Lock. That's correct.
In `handleDelete`, I acquire Lock, check, delete, unlock. That's correct.

Let's refine `handleGet` to exactly match Python's behavior:
Python:
```python
        with lock:
            if key not in store:
                return self._json(404, {"error": "not found"})
            val, exp = store[key]
            if time.time() > exp:
                del store[key]
                return self._json(404, {"error": "expired"})
            self._json(200, {"key": key, "value": val, "ttl": int(exp - time.time())})
```
So it holds the lock for the whole check and response. In Go, holding a lock while writing to network is bad practice, but to match exactly, I could just release it before writing. The Python code holds it during `self._json`, which is synchronous in the handler. In Go, it's better to release the lock before sending the response. I'll release it before `writeJSON`. That's standard and safe.

Let's adjust `handleGet`:
```go
func handleGet(w http.ResponseWriter, r *http.Request, key string) {
	mu.RLock()
	entry, exists := store[key]
	mu.RUnlock()

	if !exists {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
		return
	}

	if time.Now().After(entry.Expiry) {
		mu.Lock()
		delete(store, key)
		mu.Unlock()
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "expired"})
		return
	}

	ttl := int(time.Until(entry.Expiry).Seconds())
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"key":   key,
		"value": entry.Value,
		"ttl":   ttl,
	})
}
```
This is fine.

For `handleDelete`:
```go
func handleDelete(w http.ResponseWriter, r *http.Request, key string) {
	mu.Lock()
	if _, exists := store[key]; exists {
		delete(store, key)
		mu.Unlock()
		writeJSON(w, http.StatusOK, map[string]string{"deleted": key})
		return
	}
	mu.Unlock()
	writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
}
```
This matches.

For `handlePut`:
```go
func handlePut(w http.ResponseWriter, r *http.Request, key string) {
	var data map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&data); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON"})
		return
	}

	value, ok := data["value"]
	if !ok {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "missing value"})
		return
	}

	ttl := 3600.0
	if t, ok := data["ttl"]; ok {
		if f, ok := t.(float64); ok {
			ttl = f
		}
	}

	mu.Lock()
	store[key] = Entry{
		Value:  value,
		Expiry: time.Now().Add(time.Duration(ttl) * time.Second),
	}
	mu.Unlock()

	writeJSON(w, http.StatusCreated, map[string]string{"key": key})
}
```
This matches.

One minor point: `time.Until(entry.Expiry).Seconds()` might return a float, casting to int truncates. Python's `int(exp - time.time())` also truncates. So it's consistent.

I should also handle the case where `key` is empty in the main handler, but the Python code just strips `/` and if it's empty, it would look up `""` in store. I'll keep it simple: if `key == ""`, return 400 or just let it proceed. The Python code would just treat `""` as a key. I'll allow it or return 400. I'll just pass it through, or add a simple check. Actually, `strings.TrimPrefix(r.URL.Path, "/")` on `/` gives `""`. I'll just let it be, or return 400. It doesn't matter much. I'll keep it as is.

Wait, the Python code uses `self.path.strip('/')`. If path is `/`, it becomes `""`. If path is `/foo`, it becomes `foo`. I'll use `strings.TrimPrefix(r.URL.Path, "/")` which is standard.

Let's assemble the complete `main.go`.

I'll make sure to import only necessary packages.
I'll use `json.Marshal` in `writeJSON` to avoid trailing newline, matching Python's `json.dumps` exactly, though it's minor.

```go
package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"
)

type Entry struct {
	Value  interface{}
	Expiry time.Time
}

var (
	store = make(map[string]Entry)
	mu    sync.RWMutex
)

func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func handler(w http.ResponseWriter, r *http.Request) {
	key := strings.TrimPrefix(r.URL.Path, "/")

	if key == "health" {
		if r.Method != http.MethodGet {
			http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
			return
		}
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
		return
	}

	if key == "" {
		http.Error(w, "Bad Request", http.StatusBadRequest)
		return
	}

	switch r.Method {
	case http.MethodGet:
		handleGet(w, r, key)
	case http.MethodPut:
		handlePut(w, r, key)
	case http.MethodDelete:
		handleDelete(w, r, key)
	default:
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
	}
}

func handleGet(w http.ResponseWriter, r *http.Request, key string) {
	mu.RLock()
	entry, exists := store[key]
	mu.RUnlock()

	if !exists {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
		return
	}

	if time.Now().After(entry.Expiry) {
		mu.Lock()
		delete(store, key)
		mu.Unlock()
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "expired"})
		return
	}

	ttl := int(time.Until(entry.Expiry).Seconds())
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"key":   key,
		"value": entry.Value,
		"ttl":   ttl,
	})
}

func handlePut(w http.ResponseWriter, r *http.Request, key string) {
	var data map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&data); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON"})
		return
	}

	value, ok := data["value"]
	if !ok {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "missing value"})
		return
	}

	ttl := 3600.0
	if t, ok := data["ttl"]; ok {
		if f, ok := t.(float64); ok {
			ttl = f
		}
	}

	mu.Lock()
	store[key] = Entry{
		Value:  value,
		Expiry: time.Now().Add(time.Duration(ttl) * time.Second),
	}
	mu.Unlock()

	writeJSON(w, http.StatusCreated, map[string]string{"key": key})
}

func handleDelete(w http.ResponseWriter, r *http.Request, key string) {
	mu.Lock()
	if _, exists := store[key]; exists {
		delete(store, key)
		mu.Unlock()
		writeJSON(w, http.StatusOK, map[string]string{"deleted": key})
		return
	}
	mu.Unlock()
	writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
}

func main() {
	http.HandleFunc("/", handler)
	fmt.Println("Server listening on :8080")
	http.ListenAndServe(":8080", nil)
}
```

Wait, `json.NewEncoder(w).Encode(data)` adds a newline at the end. Python's `json.dumps` does not. It's usually fine, but to be strictly equivalent, I could use `json.Marshal` and `w.Write`. I'll change `writeJSON` to use `json.Marshal` to avoid the trailing newline, matching Python's behavior more closely.

```go
func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	b, err := json.Marshal(data)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Write(b)
}
```

This is safer.

Also, check `handleGet` TTL calculation: `int(time.Until(entry.Expiry).Seconds())`. If `entry.Expiry` is in the past, `time.Until` returns negative. But we already checked `time.Now().After(entry.Expiry)`, so it's positive. Good.

One detail: Python's `data.get('ttl', 3600)` might receive an integer or float. JSON numbers are decoded as float64 in Go. My code handles that.

What about `handlePut` error handling? The Python code doesn't explicitly handle missing `value` or invalid JSON, it would crash. But the prompt says "proper error handling", so my checks are good.

Let's verify the `sync.RWMutex` usage requirement: "sync.RWMutex (RLock for reads)". I used `mu.RLock()` in `handleGet` for the initial read. That satisfies it.

One minor thing: `strings.TrimPrefix(r.URL.Path, "/")` on `/health` gives `health`. On `/foo` gives `foo`. On `/` gives `""`. That's correct.

I'll put it all together. I'll make sure the code is complete and compiles.

One thing: `time.Until(entry.Expiry).Seconds()` might be slightly off due to floating point, but it's fine. Alternatively, `int(entry.Expiry.Sub(time.Now()).Seconds())` is the same.

I'll stick with `int(time.Until(entry.Expiry).Seconds())`.

Let's double check the Python code's `do_GET`:
```python
        with lock:
            if key not in store:
                return self._json(404, {"error": "not found"})
            val, exp = store[key]
            if time.time() > exp:
                del store[key]
                return self._json(404, {"error": "expired"})
            self._json(200, {"key": key, "value": val, "ttl": int(exp - time.time())})
```
My Go code releases the lock before writing JSON. This is better practice in Go. The prompt doesn't forbid it. It says "proper error handling", which implies Go idioms.

I'll adjust `handleGet` to exactly match the logic but release lock before response:
Actually, to be extremely faithful, I could keep the lock until response, but that's bad. I'll keep my version. It's standard.

One more check: `handlePut` in Python:
```python
        data = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
        with lock:
            store[key] = (data['value'], time.time() + data.get('ttl', 3600))
        self._json(201, {"key": key})
```
My Go code matches.

All good.

I will produce the final code.
I'll wrap it in a single `main.go` file as requested.
No extra text.
Ready. 
Let's generate. 
Self-Correction/Refinement during thought:
- `json.NewEncoder` vs `json.Marshal`: I'll use `json.Marshal` to avoid trailing newline, matching Python exactly.
- `time.Until` vs `exp - time.time()`: I'll use `int(entry.Expiry.Sub(time.Now()).Seconds())` for clarity.
- I'll make sure imports are correct.
- I'll ensure the code handles `GET /health` correctly.
- The prompt says "Produce just the Go code."

Final code structure:
```go
package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"
)

type Entry struct {
	Value  interface{}
	Expiry time.Time
}

var (
	store = make(map[string]Entry)
	mu    sync.RWMutex
)

func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	b, err := json.Marshal(data)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Write(b)
}

func handler(w http.ResponseWriter, r *http.Request) {
	key := strings.TrimPrefix(r.URL.Path, "/")

	if key == "health" {
		if r.Method != http.MethodGet {
			http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
			return
		}
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
		return
	}

	if key == "" {
		http.Error(w, "Bad Request", http.StatusBadRequest)
		return
	}

	switch r.Method {
	case http.MethodGet:
		handleGet(w, r, key)
	case http.MethodPut:
		handlePut(w, r, key)
	case http.MethodDelete:
		handleDelete(w, r, key)
	default:
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
	}
}

func handleGet(w http.ResponseWriter, r *http.Request, key string) {
	mu.RLock()
	entry, exists := store[key]
	mu.RUnlock()

	if !exists {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
		return
	}

	if time.Now().After(entry.Expiry) {
		mu.Lock()
		delete(store, key)
		mu.Unlock()
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "expired"})
		return
	}

	ttl := int(entry.Expiry.Sub(time.Now()).Seconds())
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"key":   key,
		"value": entry.Value,
		"ttl":   ttl,
	})
}

func handlePut(w http.ResponseWriter, r *http.Request, key string) {
	var data map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&data); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON"})
		return
	}

	value, ok := data["value"]
	if !ok {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "missing value"})
		return
	}

	ttl := 3600.0
	if t, ok := data["ttl"]; ok {
		if f, ok := t.(float64); ok {
			ttl = f
		}
	}

	mu.Lock()
	store[key] = Entry{
		Value:  value,
		Expiry: time.Now().Add(time.Duration(ttl) * time.Second),
	}
	mu.Unlock()

	writeJSON(w, http.StatusCreated, map[string]string{"key": key})
}

func handleDelete(w http.ResponseWriter, r *http.Request, key string) {
	mu.Lock()
	if _, exists := store[key]; exists {
		delete(store, key)
		mu.Unlock()
		writeJSON(w, http.StatusOK, map[string]string{"deleted": key})
		return
	}
	mu.Unlock()
	writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
}

func main() {
	http.HandleFunc("/", handler)
	fmt.Println("Server listening on :8080")
	http.ListenAndServe(":8080", nil)
}
```

This looks solid. It meets all requirements: `net/http`, `sync.RWMutex` with `RLock` for reads, proper error handling, correct endpoints, and produces just the Go code.
I will output this.
One minor point: `time.Now().After(entry.Expiry)` is correct. `entry.Expiry.Sub(time.Now()).Seconds()` is correct.
The code is ready.
I'll produce it now. 
All checks passed. No extra fluff.
Output matches request.
Proceeds. 
[Done]

## Response


```go
package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"
)

type Entry struct {
	Value  interface{}
	Expiry time.Time
}

var (
	store = make(map[string]Entry)
	mu    sync.RWMutex
)

func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	b, err := json.Marshal(data)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Write(b)
}

func handler(w http.ResponseWriter, r *http.Request) {
	key := strings.TrimPrefix(r.URL.Path, "/")

	if key == "health" {
		if r.Method != http.MethodGet {
			http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
			return
		}
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
		return
	}

	if key == "" {
		http.Error(w, "Bad Request", http.StatusBadRequest)
		return
	}

	switch r.Method {
	case http.MethodGet:
		handleGet(w, r, key)
	case http.MethodPut:
		handlePut(w, r, key)
	case http.MethodDelete:
		handleDelete(w, r, key)
	default:
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
	}
}

func handleGet(w http.ResponseWriter, r *http.Request, key string) {
	mu.RLock()
	entry, exists := store[key]
	mu.RUnlock()

	if !exists {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
		return
	}

	if time.Now().After(entry.Expiry) {
		mu.Lock()
		delete(store, key)
		mu.Unlock()
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "expired"})
		return
	}

	ttl := int(entry.Expiry.Sub(time.Now()).Seconds())
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"key":   key,
		"value": entry.Value,
		"ttl":   ttl,
	})
}

func handlePut(w http.ResponseWriter, r *http.Request, key string) {
	var data map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&data); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON"})
		return
	}

	value, ok := data["value"]
	if !ok {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "missing value"})
		return
	}

	ttl := 3600.0
	if t, ok := data["ttl"]; ok {
		if f, ok := t.(float64); ok {
			ttl = f
		}
	}

	mu.Lock()
	store[key] = Entry{
		Value:  value,
		Expiry: time.Now().Add(time.Duration(ttl) * time.Second),
	}
	mu.Unlock()

	writeJSON(w, http.StatusCreated, map[string]string{"key": key})
}

func handleDelete(w http.ResponseWriter, r *http.Request, key string) {
	mu.Lock()
	if _, exists := store[key]; exists {
		delete(store, key)
		mu.Unlock()
		writeJSON(w, http.StatusOK, map[string]string{"deleted": key})
		return
	}
	mu.Unlock()
	writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
}

func main() {
	http.HandleFunc("/", handler)
	fmt.Println("Server listening on :8080")
	http.ListenAndServe(":8080", nil)
}
```

## Metadata
- **Model**: intel-qwen3.6-27b-int4-16k
- **Elapsed**: 134433ms
- **Prompt tokens**: 474
- **Completion tokens**: 6486
- **Total tokens**: 6960
- **Gen tok/s**: 48.2
