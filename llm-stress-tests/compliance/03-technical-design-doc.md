# Technical Design Document

**Target Capability:** Clear architecture description, trade-off analysis, structured technical writing.

Tests whether a model can produce a professional design document — not code, but the reasoning behind architectural decisions.

---

## Prompt

```
Write a technical design document for a distributed job scheduler that runs scheduled tasks across a fleet of worker nodes.

The document should include:

1. **Overview** — What the system does, who uses it, and the problem it solves.
2. **Architecture** — High-level component diagram (use Mermaid syntax), data flow description.
3. **Core Components** — Detailed description of each service (scheduler, worker, result store, API gateway).
4. **Data Model** — Tables/collections for jobs, schedules, results, and worker registrations.
5. **API Design** — REST endpoints for creating, listing, canceling, and querying jobs.
6. **Failure Handling** — What happens when a worker dies mid-task, when the scheduler loses its lock, or when a job times out.
7. **Scaling Strategy** — How to handle 100K scheduled jobs and 500 active workers.
8. **Trade-offs** — At least 3 explicit trade-offs (e.g., consistency vs availability, polling vs push, centralized vs distributed locking).
9. **Operational Concerns** — Monitoring, alerting, and debugging strategies.

Constraints:
- Use PostgreSQL for persistent storage, Redis for task queue and distributed locking.
- Workers are stateless containers that pull tasks from Redis.
- Jobs can be one-time or recurring (cron-style).
- Job results must be retrievable for 30 days.

Write in a professional, concise style suitable for a design review meeting. No code blocks except for the Mermaid diagram and API endpoint definitions.
```
