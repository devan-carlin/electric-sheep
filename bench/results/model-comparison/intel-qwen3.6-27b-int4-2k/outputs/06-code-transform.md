# Output: intel-qwen3.6-27b-int4-2k × 06-code-transform

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
Here's a thinking process:

1.  **Understand the Goal**: The user wants to convert a Python HTTP key-value store handler to Go. The Go code must use `net/http` and `sync.RWMutex`. It needs to handle `GET /{key}`, `PUT /{key}`, `DELETE /{key}`, and `GET /health`. It must be a complete `main.go` file.

2.  **Analyze the Python Code**:
   - Uses a global dictionary `store` mapping keys to `(value, expiration_time)`.
   - Uses a `threading.Lock()` for thread safety.
   - `do_GET`:
     - Strips leading/trailing slashes from path to get key.
     - If key is "health", returns `{"status": "ok"}` with 200.
     - Otherwise, locks, checks if key exists. If not, returns 404 `{"error": "not found"}`.
     - Checks expiration: if `time.time() > exp`, deletes key, returns 404 `{"error": "expired"}`.
     - Returns 200 with `{"key": key, "value": val, "ttl": int(exp - time.time())}`.
   - `do_PUT`:
     - Strips slashes to get key.
     - Reads JSON body, parses it.
     - Locks, stores `(data['value'], time.time() + data.get('ttl', 3600))`.
     - Returns 201 with `{"key": key}`.
   - `do_DELETE`:
     - Strips slashes to get key.
     - Locks, checks if key exists. If yes, deletes and returns 200 `{"deleted": key}`.
     - If not, returns 404 `{"error": "not found"}`.
   - `_json`: Helper to send JSON response with status code and content-type.

3.  **Map to Go Requirements**:
   - Use `net/http` for HTTP server and handlers.
   - Use `sync.RWMutex` for thread safety. The prompt specifically says `sync.RWMutex` (RLock for reads).
   - Need a struct or map to store data: `map[string]Entry` where `Entry` contains `Value` and `ExpiresAt`.
   - Handle methods: `GET`, `PUT`, `DELETE`.
   - Handle paths: `/health`, `/{key}`.
   - Proper error handling.
   - Complete `main.go`.

4.  **Design Go Structures**:
   ```go
   type Entry struct {
       Value     interface{} // or string, but Python uses arbitrary JSON value. Let's use interface{} or json.RawMessage. Actually, Python's data['value'] can be anything. I'll use interface{} or string. Let's stick to interface{} for flexibility, or just string since it's a simple KV store. The Python code stores whatever is in data['value']. I'll use interface{}.
       ExpiresAt float64     // Unix timestamp
   }
   ```
   Actually, in Go, it's better to use `json.RawMessage` or `interface{}`. I'll use `interface{}` for simplicity, or just `string` if we assume simple values. The Python code doesn't restrict it. I'll use `interface{}`.

   Wait, the Python code stores `data['value']` directly. I'll use `interface{}`.

   Store: `var store = make(map[string]Entry)`
   Mutex: `var mu sync.RWMutex`

5.  **Implement Handlers**:
   I'll create a single handler function that routes based on method and path, or use `http.HandleFunc` with a router. Since it's a simple app, I'll write a `handler` function that checks `r.Method` and `r.URL.Path`.

   Path parsing: `strings.TrimPrefix(r.URL.Path, "/")` and `strings.TrimSuffix(..., "/")` or just `strings.Trim(r.URL.Path, "/")`.

   Let's handle routing:
   ```go
   func handler(w http.ResponseWriter, r *http.Request) {
       key := strings.Trim(r.URL.Path, "/")

       switch r.Method {
       case http.MethodGet:
           if key == "health" {
               writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
               return
           }
           // GET key
           mu.RLock()
           entry, exists := store[key]
           mu.RUnlock()

           if !exists {
               writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
               return
           }

           // Check expiration
           now := time.Now().Unix()
           if now > entry.ExpiresAt {
               mu.Lock()
               // Double check after acquiring write lock
               if entry2, ok := store[key]; ok && now > entry2.ExpiresAt {
                   delete(store, key)
               }
               mu.Unlock()
               writeJSON(w, http.StatusNotFound, map[string]string{"error": "expired"})
               return
           }

           ttl := int(entry.ExpiresAt - now)
           writeJSON(w, http.StatusOK, map[string]interface{}{
               "key":   key,
               "value": entry.Value,
               "ttl":   ttl,
           })

       case http.MethodPut:
           // PUT key
           var body map[string]interface{}
           if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
               writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid json"})
               return
           }

           val, _ := body["value"]
           ttl, _ := body["ttl"].(float64)
           if ttl == 0 {
               ttl = 3600
           }

           mu.Lock()
           store[key] = Entry{
               Value:     val,
               ExpiresAt: time.Now().Unix() + int64(ttl),
           }
           mu.Unlock()

           writeJSON(w, http.StatusCreated, map[string]string{"key": key})

       case http.MethodDelete:
           // DELETE key
           mu.Lock()
           if _, exists := store[key]; exists {
               delete(store, key)
               mu.Unlock()
               writeJSON(w, http.StatusOK, map[string]string{"deleted": key})
               return
           }
           mu.Unlock()
           writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})

       default:
           writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
       }
   }
   ```

   Wait, there's a race condition in the GET expiration check. The Python code does:
   ```python
   with lock:
       if key not in store: ...
       val, exp = store[key]
       if time.time() > exp:
           del store[key]
           return ...
   ```
   It holds the lock for the whole read+check+delete. In Go, using `RWMutex`, if we want to strictly follow the Python behavior (which uses a regular lock for everything), we could just use `mu.Lock()` for GET as well, but the prompt says `sync.RWMutex` (RLock for reads). To properly use `RWMutex`, we should read with `RLock`, check expiration, and if expired, upgrade to `Lock` to delete. However, Go's `sync.RWMutex` doesn't support lock upgrading. The standard pattern is:
   ```go
   mu.RLock()
   entry, exists := store[key]
   mu.RUnlock()
   if !exists { ... }
   if time.Now().Unix() > entry.ExpiresAt {
       mu.Lock()
       // re-check
       if e, ok := store[key]; ok && time.Now().Unix() > e.ExpiresAt {
           delete(store, key)
       }
       mu.Unlock()
       ...
   }
   ```
   This is safe and idiomatic. I'll implement it this way.

   Also, `time.Now().Unix()` returns `int64`. Python's `time.time()` returns float. I'll stick to `int64` for simplicity, or `float64` to match exactly. I'll use `int64` as it's standard in Go, but I'll make sure TTL calculation matches. The Python code does `int(exp - time.time())`. I'll do `int(entry.ExpiresAt - now)`.

   Let's refine the `Entry` struct:
   ```go
   type Entry struct {
       Value     interface{}
       ExpiresAt int64
   }
   ```

   For PUT, `ttl` from JSON might be a number. `json.Decoder` decodes numbers as `float64`. So `ttl, _ := body["ttl"].(float64)` works. If not present, default to 3600.

   Let's handle the JSON response helper:
   ```go
   func writeJSON(w http.ResponseWriter, status int, data interface{}) {
       w.Header().Set("Content-Type", "application/json")
       w.WriteHeader(status)
       json.NewEncoder(w).Encode(data)
   }
   ```
   Note: `json.NewEncoder

## Response


## Metadata
- **Model**: intel-qwen3.6-27b-int4-2k
- **Elapsed**: 42611ms
- **Prompt tokens**: 474
- **Completion tokens**: 2048
- **Total tokens**: 2522
- **Gen tok/s**: 48.0
