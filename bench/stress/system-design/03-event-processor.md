# System Design: Event Processing Pipeline

Design and implement an event processing pipeline that consumes messages from a message queue, processes them, and handles failures gracefully.

## Requirements

- **Consumer service** — reads from a RabbitMQ queue, processes each message.
- **Idempotency** — processing the same message twice produces the same result (no duplicate side effects).
- **Dead letter queue** — messages that fail after 3 retries are moved to a DLQ for manual inspection.
- **Exponential backoff** — retry delays: 1s, 2s, 4s (then DLQ).
- **Ordering guarantee** — messages with the same `tenant_id` must be processed in order (no parallel processing per tenant).
- **Metrics** — expose `/metrics` endpoint with: total processed, total failed, current queue depth, average processing time.

## Implementation

Build a Python application with:

1. A `MessageProcessor` class that:
   - Connects to RabbitMQ (use `pika` library).
   - Prefetch count = 1 per tenant (to maintain ordering).
   - Tracks retry count in message headers.
   - NACK + requeue for transient failures, NACK + reject for permanent failures.

2. A `IdempotencyStore` class that:
   - Uses SQLite to track processed message IDs.
   - Checks `INSERT OR IGNORE` before processing.
   - Has a cleanup method for entries older than 7 days.

3. A `MetricsCollector` class that:
   - Uses thread-safe counters (threading.Lock).
   - Exposes Prometheus-style metrics via HTTP.

## Constraints

- Use `pika` for RabbitMQ (not `aio-pika` — keep it synchronous for simplicity).
- SQLite for idempotency tracking (single-node deployment).
- The processing function should simulate work with `time.sleep(random.uniform(0.1, 0.5))`.
- Include a `producer.py` script that publishes test messages with `tenant_id` and `message_id` headers.

## Deliverable

1. `consumer.py` — the main consumer with retry logic and DLQ routing.
2. `idempotency.py` — the idempotency store.
3. `metrics.py` — the metrics collector and HTTP endpoint.
4. `producer.py` — test message publisher.
5. `rabbitmq.conf` — RabbitMQ configuration for queues, DLQ, and bindings.

## Evaluation Criteria

| Criterion | Weight |
|-----------|--------|
| Correct retry logic (3 retries, exponential backoff) | 25% |
| Idempotency (no duplicate processing) | 20% |
| DLQ routing (failed messages after max retries) | 20% |
| Per-tenant ordering (no parallel processing per tenant) | 20% |
| Metrics accuracy and completeness | 15% |
