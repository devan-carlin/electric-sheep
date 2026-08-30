# Cross-File Reference

**Category:** Long context
**Target:** Tracking references across multiple files, understanding system architecture

---

## Prompt

You are given the complete source code of a small but non-trivial Go application — a rate limiter service with 8 files. Answer the following questions by tracing through the code:

1. When a client makes a request that exceeds the rate limit, what HTTP status code is returned and what does the JSON response body contain?
2. If `config.yaml` sets `strategy: "sliding_window"` but does not set `window_size`, what value is used and why?
3. Trace the path from `httpHandler.HandleRequest()` to the point where the decision is made to allow or deny. List every function called in order.
4. What happens if the Redis connection fails while processing a request? Is the request allowed, denied, or does it error?
5. There is a subtle bug in the sliding window implementation where the count can be off by one under high concurrency. Identify the bug.

**Files:**

### `main.go`
```go
package main

import (
    "fmt"
    "log"
    "net/http"
    "os"
    "os/signal"
    "syscall"

    "ratelimit/config"
    "ratelimit/handler"
    "ratelimit/store"
    "ratelimit/strategy"
)

func main() {
    cfg, err := config.Load("config.yaml")
    if err != nil {
        log.Fatalf("Failed to load config: %v", err)
    }

    redisStore, err := store.NewRedisStore(cfg.RedisURL, cfg.RedisDB)
    if err != nil {
        log.Fatalf("Failed to connect to Redis: %v", err)
    }
    defer redisStore.Close()

    var limiter strategy.Limiter
    switch cfg.Strategy {
    case "token_bucket":
        limiter = strategy.NewTokenBucket(redisStore, cfg.Rate, cfg.Burst)
    case "sliding_window":
        limiter = strategy.NewSlidingWindow(redisStore, cfg.Rate, cfg.WindowSize)
    default:
        log.Fatalf("Unknown strategy: %s", cfg.Strategy)
    }

    mux := http.NewServeMux()
    h := handler.NewHTTPHandler(limiter, cfg)
    mux.HandleFunc("/check", h.HandleRequest)
    mux.HandleFunc("/health", h.HealthCheck)
    mux.HandleFunc("/reset", h.HandleReset)

    addr := fmt.Sprintf(":%d", cfg.Port)
    log.Printf("Starting rate limiter on %s (strategy: %s, rate: %d/s)", addr, cfg.Strategy, cfg.Rate)

    server := &http.Server{Addr: addr, Handler: mux}

    go func() {
        if err := server.ListenAndServe(); err != http.ErrServerClosed {
            log.Fatalf("Server failed: %v", err)
        }
    }()

    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
    <-quit
    log.Println("Shutting down...")
    server.Close()
}
```

### `config/config.go`
```go
package config

import (
    "os"
    "gopkg.in/yaml.v3"
)

type Config struct {
    Port       int    `yaml:"port"`
    RedisURL   string `yaml:"redis_url"`
    RedisDB    int    `yaml:"redis_db"`
    Strategy   string `yaml:"strategy"`
    Rate       int    `yaml:"rate"`
    Burst      int    `yaml:"burst"`
    WindowSize int    `yaml:"window_size"`
    KeyPrefix  string `yaml:"key_prefix"`
}

func Load(path string) (*Config, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, err
    }

    cfg := &Config{
        Port:       8080,
        RedisURL:   "redis://localhost:6379",
        RedisDB:    0,
        Strategy:   "token_bucket",
        Rate:       100,
        Burst:      150,
        WindowSize: 60,
        KeyPrefix:  "rl:",
    }

    if err := yaml.Unmarshal(data, cfg); err != nil {
        return nil, err
    }

    // Validate
    if cfg.Rate <= 0 {
        cfg.Rate = 100
    }
    if cfg.Burst < cfg.Rate {
        cfg.Burst = cfg.Rate
    }
    // Note: WindowSize is NOT validated — if set to 0 in YAML, it stays 0

    return cfg, nil
}
```

### `store/redis.go`
```go
package store

import (
    "context"
    "fmt"
    "time"

    "github.com/redis/go-redis/v9"
)

type RedisStore struct {
    client *redis.Client
}

func NewRedisStore(url string, db int) (*RedisStore, error) {
    opt, err := redis.ParseURL(url)
    if err != nil {
        return nil, fmt.Errorf("parse redis URL: %w", err)
    }
    opt.DB = db

    client := redis.NewClient(opt)
    
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    
    if err := client.Ping(ctx).Err(); err != nil {
        return nil, fmt.Errorf("redis ping: %w", err)
    }

    return &RedisStore{client: client}, nil
}

func (s *RedisStore) Close() error {
    return s.client.Close()
}

func (s *RedisStore) Increment(ctx context.Context, key string, ttl time.Duration) (int64, error) {
    return s.client.Incr(ctx, key).Result()
}

func (s *RedisStore) SetWithExpiry(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
    return s.client.Set(ctx, key, value, ttl).Err()
}

func (s *RedisStore) Get(ctx context.Context, key string) (string, error) {
    return s.client.Get(ctx, key).Result()
}

func (s *RedisStore) Delete(ctx context.Context, key string) error {
    return s.client.Del(ctx, key).Err()
}

func (s *RedisStore) Pipeline(ctx context.Context, cmds func(*redis.Pipeline)) ([]redis.Cmder, error) {
    pipe := s.client.Pipeline()
    cmds(pipe)
    return pipe.Exec(ctx)
}

func (s *RedisStore) Eval(ctx context.Context, script string, keys []string, args ...interface{}) (*redis.Cmd, error) {
    return s.client.Eval(ctx, script, keys), nil
}
```

### `strategy/limiter.go`
```go
package strategy

import (
    "context"
    "time"
)

type Result struct {
    Allowed    bool  `json:"allowed"`
    Remaining int64 `json:"remaining"`
    Limit     int64 `json:"limit"`
    ResetAt   time.Time `json:"reset_at"`
}

type Limiter interface {
    Allow(ctx context.Context, key string) (*Result, error)
    Reset(ctx context.Context, key string) error
}
```

### `strategy/token_bucket.go`
```go
package strategy

import (
    "context"
    "fmt"
    "time"

    "ratelimit/store"
)

type TokenBucket struct {
    store  *store.RedisStore
    rate   int
    burst  int
}

func NewTokenBucket(store *store.RedisStore, rate, burst int) *TokenBucket {
    return &TokenBucket{store: store, rate: rate, burst: burst}
}

func (tb *TokenBucket) Allow(ctx context.Context, key string) (*Result, error) {
    now := time.Now().UnixNano()
    bucketKey := fmt.Sprintf("tb:%s", key)

    script := `
        local bucket = redis.call('hmget', KEYS[1], 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1])
        local last_refill = tonumber(bucket[2])
        
        if tokens == nil then
            tokens = tonumber(ARGV[1])  -- burst
            last_refill = tonumber(ARGV[2])  -- now
        end
        
        local elapsed = (tonumber(ARGV[2]) - last_refill) / 1e9
        local refill = elapsed * tonumber(ARGV[3])  -- rate
        tokens = math.min(tonumber(ARGV[1]), tokens + refill)
        
        local allowed = 0
        if tokens >= 1 then
            tokens = tokens - 1
            allowed = 1
        end
        
        redis.call('hmset', KEYS[1], 'tokens', tokens, 'last_refill', ARGV[2])
        redis.call('expire', KEYS[1], tonumber(ARGV[4]))
        
        return {allowed, math.floor(tokens)}
    `

    cmd, err := tb.store.Eval(ctx, script, []string{bucketKey},
        tb.burst, now, tb.rate, 3600)
    if err != nil {
        // On Redis failure, allow the request (fail-open)
        return &Result{Allowed: true, Remaining: 0, Limit: int64(tb.rate)}, nil
    }

    results, err := cmd.Results()
    if err != nil {
        return &Result{Allowed: true, Remaining: 0, Limit: int64(tb.rate)}, nil
    }

    allowed := results[0].(int64) == 1
    remaining := results[1].(int64)

    return &Result{
        Allowed:   allowed,
        Remaining: remaining,
        Limit:     int64(tb.rate),
        ResetAt:   time.Now().Add(time.Duration(tb.burst) * time.Second),
    }, nil
}

func (tb *TokenBucket) Reset(ctx context.Context, key string) error {
    return tb.store.Delete(ctx, fmt.Sprintf("tb:%s", key))
}
```

### `strategy/sliding_window.go`
```go
package strategy

import (
    "context"
    "fmt"
    "time"

    "ratelimit/store"
)

type SlidingWindow struct {
    store      *store.RedisStore
    rate       int
    windowSize int
}

func NewSlidingWindow(store *store.RedisStore, rate, windowSize int) *SlidingWindow {
    return &SlidingWindow{store: store, rate: rate, windowSize: windowSize}
}

func (sw *SlidingWindow) Allow(ctx context.Context, key string) (*Result, error) {
    now := time.Now()
    windowKey := fmt.Sprintf("sw:%s", key)
    windowDuration := time.Duration(sw.windowSize) * time.Second

    // Bug: if windowSize is 0 (not validated in config), this creates a 0-duration window
    if sw.windowSize <= 0 {
        windowDuration = 60 * time.Second
    }

    windowStart := now.Add(-windowDuration).UnixNano()

    script := `
        local key = KEYS[1]
        local window_start = tonumber(ARGV[1])
        local window_size = tonumber(ARGV[2])
        local rate_limit = tonumber(ARGV[3])
        
        -- Remove expired entries
        redis.call('zremrangebyscore', key, '-inf', window_start)
        
        -- Count current entries
        local count = redis.call('zcard', key)
        
        local allowed = 0
        if count < rate_limit then
            redis.call('zadd', key, ARGV[4], ARGV[4])
            allowed = 1
            count = count + 1
        end
        
        redis.call('expire', key, window_size)
        
        return {allowed, rate_limit - count}
    `

    cmd, err := sw.store.Eval(ctx, script, []string{windowKey},
        windowStart, int64(windowDuration.Seconds()), sw.rate, now.UnixNano())
    if err != nil {
        return &Result{Allowed: true, Remaining: 0, Limit: int64(sw.rate)}, nil
    }

    results, err := cmd.Results()
    if err != nil {
        return &Result{Allowed: true, Remaining: 0, Limit: int64(sw.rate)}, nil
    }

    allowed := results[0].(int64) == 1
    remaining := results[1].(int64)

    return &Result{
        Allowed:   allowed,
        Remaining: max(0, remaining),
        Limit:     int64(sw.rate),
        ResetAt:   now.Add(windowDuration),
    }, nil
}

func (sw *SlidingWindow) Reset(ctx context.Context, key string) error {
    return sw.store.Delete(ctx, fmt.Sprintf("sw:%s", key))
}
```

### `handler/http.go`
```go
package handler

import (
    "encoding/json"
    "net/http"
    "time"
    "ratelimit/config"
    "ratelimit/strategy"
)

type HTTPHandler struct {
    limiter Limiter
    config  *config.Config
}

func NewHTTPHandler(limiter strategy.Limiter, cfg *config.Config) *HTTPHandler {
    return &HTTPHandler{limiter: limiter, config: cfg}
}

func (h *HTTPHandler) HandleRequest(w http.ResponseWriter, r *http.Request) {
    ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
    defer cancel()

    // Extract client identifier
    clientKey := r.Header.Get("X-Client-ID")
    if clientKey == "" {
        clientKey = r.RemoteAddr
    }
    key := h.config.KeyPrefix + clientKey

    result, err := h.limiter.Allow(ctx, key)
    if err != nil {
        http.Error(w, "Internal error", http.StatusInternalServerError)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    w.Header().Set("X-RateLimit-Limit", fmt.Sprintf("%d", result.Limit))
    w.Header().Set("X-RateLimit-Remaining", fmt.Sprintf("%d", result.Remaining))
    w.Header().Set("X-RateLimit-Reset", fmt.Sprintf("%d", result.ResetAt.Unix()))

    if !result.Allowed {
        w.Header().Set("Retry-After", fmt.Sprintf("%d", time.Until(result.ResetAt).Seconds()))
        w.WriteHeader(http.StatusTooManyRequests)
        json.NewEncoder(w).Encode(map[string]interface{}{
            "error":     "rate_limit_exceeded",
            "message":   "You have exceeded the rate limit. Please retry later.",
            "retry_after": time.Until(result.ResetAt).Seconds(),
            "limit":     result.Limit,
        })
        return
    }

    w.WriteHeader(http.StatusOK)
    json.NewEncoder(w).Encode(map[string]interface{}{
        "allowed":   true,
        "remaining": result.Remaining,
        "limit":     result.Limit,
    })
}

func (h *HTTPHandler) HealthCheck(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (h *HTTPHandler) HandleReset(w http.ResponseWriter, r *http.Request) {
    clientKey := r.Header.Get("X-Client-ID")
    if clientKey == "" {
        http.Error(w, "X-Client-ID required", http.StatusBadRequest)
        return
    }
    key := h.config.KeyPrefix + clientKey

    if err := h.limiter.Reset(r.Context(), key); err != nil {
        http.Error(w, "Failed to reset", http.StatusInternalServerError)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]string{"status": "reset"})
}
```

### `config.yaml` (example)
```yaml
port: 8080
redis_url: "redis://localhost:6379"
redis_db: 0
strategy: "sliding_window"
rate: 100
burst: 150
key_prefix: "rl:"
```

### `Dockerfile`
```dockerfile
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /ratelimit ./main.go

FROM alpine:3.19
RUN apk add --no-cache ca-certificates
COPY --from=builder /ratelimit /usr/local/bin/ratelimit
EXPOSE 8080
CMD ["ratelimit"]
```

**Answer all 5 questions with specific references to the code.**
