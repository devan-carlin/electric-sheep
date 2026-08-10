# Docker Compose + Health Checks

**Category:** Data & infrastructure
**Target:** Multi-service orchestration, networking, dependency management

---

## Prompt

Create a `docker-compose.yml` that deploys a complete development environment for a web application with the following services:

1. **API** — Python FastAPI application (build from local `./api/`)
2. **Worker** — Celery worker that processes background tasks (shares code with API)
3. **PostgreSQL** — Database with initialized schema
4. **Redis** — Cache + Celery broker
5. **Nginx** — Reverse proxy that routes `/api/` to the API and serves static files
6. **Mailhog** — Local email testing server (UI on port 8025)

**Requirements:**

- All services must have health checks with appropriate intervals/timeouts
- API and Worker must wait for PostgreSQL and Redis to be healthy before starting
- Nginx must wait for API to be healthy
- PostgreSQL must initialize with a `dev` database and `app` user on first run
- Redis must have a memory limit of 256MB with `allkeys-lru` eviction
- Shared volumes for:
  - PostgreSQL data persistence
  - Redis data persistence
  - API static files (shared between API and Nginx)
- Custom network `app-net` for inter-service communication
- Mailhog exposed only on localhost (not on 0.0.0.0)
- Environment variables managed via `.env` file (provide a `.env.example`)
- `docker-compose.yml` must include:
  - `depends_on` with `condition: service_healthy`
  - Proper restart policies (`unless-stopped` for infra, `on-failure` for app)
  - Resource limits (CPU and memory) for each service
- Provide the Nginx configuration file
- Provide a minimal FastAPI `main.py` that connects to PostgreSQL and Redis
- Provide a `Makefile` with common commands (`make up`, `make down`, `make test`, `make migrate`, `make logs`)

**Constraints:**

- Docker Compose v3.8+ syntax
- No external Docker images beyond: python:3.12-slim, postgres:16, redis:7, nginx:alpine, mailhog/mailhog
- All custom images built from Dockerfiles you provide
- Health checks must use appropriate tools (`pg_isready`, `redis-cli ping`, HTTP endpoint)

Produce all files with complete working code. No placeholders, no TODOs.
