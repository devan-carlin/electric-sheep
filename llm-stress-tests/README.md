# LLM Stress Tests

A battery of prompts designed to evaluate LLM code generation across thirteen categories. Each prompt forces the model to produce complete, working output — no external dependencies, no placeholders, no broken logic.

Use these to compare output quality between models, quantization levels, or platforms.

---

## Categories

### Game Prompts (`game-prompts/`) — 12 tests

Single-file HTML applications that test canvas physics, state machines, AI opponents, and UI logic.

| # | Test | What It Probes |
|---|------|----------------|
| 1 | [Falling Words Typing Test](./game-prompts/01-typing-speed-test.md) | Real-time input handling, collision detection, analytics |
| 2 | [Wordle Clone](./game-prompts/02-wordle-clone.md) | Strict duplicate-letter evaluation logic, keyboard state sync |
| 3 | [2048 Engine](./game-prompts/03-2048-engine.md) | Matrix shift/merge algorithm, game-over detection |
| 4 | [Brick Breaker](./game-prompts/04-brick-breaker.md) | Canvas physics, dynamic paddle-bounce angle |
| 5 | [Flappy Bird](./game-prompts/05-flappy-bird.md) | Gravity physics, procedural pipe generation |
| 6 | [Sudoku](./game-prompts/06-sudoku.md) | Backtracking generator/solver, uniqueness constraint |
| 7 | [Minesweeper](./game-prompts/07-minesweeper.md) | Recursive flood-fill, first-click safety, flag logic |
| 8 | [Snake](./game-prompts/08-snake.md) | Grid movement, self-collision, progressive speed increase |
| 9 | [Tetris](./game-prompts/09-tetris.md) | Rotation matrices, wall kicks, ghost piece, hold/next queue |
| 10 | [Connect Four](./game-prompts/10-connect-four.md) | Minimax AI with alpha-beta pruning, win detection (4 directions) |
| 11 | [Memory Card Match](./game-prompts/11-memory-match.md) | 3D CSS flip animation, Fisher-Yates shuffle, move/timer tracking |
| 12 | [Simple Platformer](./game-prompts/12-platformer.md) | Gravity, platform collision, scrolling camera, moving platforms |

### Compliance (`compliance/`) — 4 tests

Strict output formatting, agentic planning, and technical writing.

| # | Test | Target Capability |
|---|------|-------------------|
| 1 | [Strict Output Compliance](./compliance/01-strict-output-compliance.md) | Deterministic JSON formatting, zero conversational fluff |
| 2 | [Agentic Function Calling](./compliance/02-agentic-function-calling.md) | Multi-step orchestration, conditional tool execution |
| 3 | [Technical Design Document](./compliance/03-technical-design-doc.md) | Architecture description, trade-off analysis, Mermaid diagrams |
| 4 | [API Documentation from Code](./compliance/04-api-documentation.md) | Accurate OpenAPI spec generation from FastAPI code |

### Multi-File Projects (`multi-file/`) — 4 tests

Multi-module architecture, routing, data layer, and testing.

| # | Test | Target Capability |
|---|------|-------------------|
| 1 | [Full-Stack CRUD](./multi-file/01-fullstack-crud.md) | FastAPI, SQLAlchemy, JWT auth, pytest |
| 2 | [CLI Tool](./multi-file/02-cli-tool.md) | Typer, modular architecture, config management |
| 3 | [Library API](./multi-file/03-library-api.md) | Type safety, docstrings, mypy --strict, zero deps |
| 4 | [FastAPI Microservice](./multi-file/04-fastapi-microservice.md) | Single-file production code, SQLite, REST standards |

### Code Transformation (`code-transform/`) — 3 tests

Cross-language reasoning, control flow refactoring, type inference.

| # | Test | Target Capability |
|---|------|-------------------|
| 1 | [Python → Go](./code-transform/01-python-to-go.md) | Idiomatic Go, error handling, httptest |
| 2 | [Callbacks → Async/Await](./code-transform/02-refactor-callback-to-async.md) | Promise.allSettled, retry logic, timeout |
| 3 | [Add TypeScript Types](./code-transform/03-add-typescript-types.md) | Generics, union types, strict mode, no `any` |

### Debugging (`debugging/`) — 5 tests

Concurrency bugs, memory leaks, query optimization, and CI diagnosis.

| # | Test | Target Capability |
|---|------|-------------------|
| 1 | [Race Condition Hunt](./debugging/01-race-condition.md) | Thread interleaving, lock ordering, deadlock |
| 2 | [Memory Leak Diagnosis](./debugging/02-memory-leak.md) | Reference cycles, profile interpretation |
| 3 | [SQL Optimization](./debugging/03-sql-optimization.md) | N+1 detection, JOINs, index selection |
| 4 | [Code Audit — Bug Hunt](./debugging/04-code-audit-bug-hunt.md) | Subtle race condition in rate limiter, restraint |
| 5 | [Debug CI Pipeline](./debugging/05-debug-ci-pipeline.md) | Read GitHub Actions logs, find root cause, fix YAML |

### Long Context (`long-context/`) — 2 tests

Information retrieval in large contexts, cross-file tracking.

| # | Test | Target Capability |
|---|------|-------------------|
| 1 | [Needle in Haystack](./long-context/01-needle-in-haystack.md) | 50+ page spec, find specific parameter |
| 2 | [Cross-File Reference](./long-context/02-cross-file-reference.md) | 8-file Go app, trace execution paths |

### Data & Infrastructure (`data-infra/`) — 4 tests

ETL pipelines, container orchestration, CI/CD, and database migrations.

| # | Test | Target Capability |
|---|------|-------------------|
| 1 | [ETL Pipeline](./data-infra/01-etl-pipeline.md) | CSV parsing, data cleaning, SQLite upsert |
| 2 | [Docker Compose](./data-infra/02-docker-compose.md) | Health checks, networking, dependency ordering |
| 3 | [CI/CD Pipeline](./data-infra/03-ci-cd-pipeline.md) | GitHub Actions, matrix builds, reusable workflows |
| 4 | [SQL Migration Script](./data-infra/04-sql-migration.md) | Schema changes, data backfill, rollback, idempotent |

### Security (`security/`) — 7 tests

Vulnerability detection, authentication, cryptography, supply chain, and infrastructure security.

| # | Test | Target Capability |
|---|------|-------------------|
| 1 | [Vulnerability Audit](./security/01-vulnerability-audit.md) | SQLi, XSS, auth bypass, path traversal |
| 2 | [Auth Implementation](./security/02-auth-implementation.md) | JWT RS256, refresh rotation, RBAC, rate limiting |
| 3 | [Input Sanitization](./security/03-input-sanitization.md) | XSS, encoding attacks, SSRF, command injection |
| 4 | [Crypto Audit](./security/04-crypto-audit.md) | Find 7 crypto mistakes (ECB, static IV, weak RNG, timing attack) |
| 5 | [OAuth 2.0 Provider](./security/05-oauth2-provider.md) | Build OIDC from scratch with PKCE, JWT RS256, refresh rotation |
| 6 | [Supply Chain Audit](./security/06-supply-chain-audit.md) | Parse lockfiles, detect CVEs/typosquatting/deprecated packages |
| 7 | [Terraform Security Review](./security/07-terraform-security-review.md) | Find 8 IaC misconfigurations (public S3, wildcard IAM, hardcoded secrets) |

### Reasoning & Math (`reasoning/`) — 4 tests

Analytical thinking, constraint satisfaction, algorithm design, and scheduling.

| # | Test | Target Capability |
|---|------|-------------------|
| 1 | [Multi-Step Math](./reasoning/01-multi-step-math.md) | Compound interest, relative speed, probability, optimization, sequences |
| 2 | [Logic Puzzle](./reasoning/02-logic-puzzle.md) | Five houses puzzle, scheduling constraint satisfaction |
| 3 | [Algorithm Design](./reasoning/03-algorithm-design.md) | Merge intervals, sliding window median, articulation points |
| 4 | [Shift Scheduler](./reasoning/04-shift-scheduler.md) | Multi-variable CSP with negative constraints, self-audit |

### System Design (`system-design/`) — 3 tests

Architecture, distributed systems, and scalability.

| # | Test | Target Capability |
|---|------|-------------------|
| 1 | [URL Shortener](./system-design/01-url-shortener.md) | API design, DB schema, ID generation, caching, scaling |
| 2 | [Rate Limiter](./system-design/02-rate-limiter.md) | Token bucket, Redis Lua scripts, distributed state, fail-open |
| 3 | [Event Processor](./system-design/03-event-processor.md) | RabbitMQ, idempotency, DLQ, exponential backoff, per-tenant ordering |

---

## Helper Scripts (`scripts/`)

Lightweight automation for running and scoring tests.

| Script | Purpose |
|--------|---------|
| [`run-test.sh`](./scripts/run-test.sh) | Send a prompt to a local LLM endpoint (OpenAI-compatible API) and save output |
| [`score-test.sh`](./scripts/score-test.sh) | Interactive scoring — rate each criterion 1–5, appends to `results/SCORES.csv` |
| [`compare-models.sh`](./scripts/compare-models.sh) | Side-by-side comparison of two models from scored results |

---

## How to Use

### Manual (Quick)

1. Copy the full prompt from a `.md` file.
2. Paste it into the LLM you want to test.
3. **Games:** Save the returned HTML as a `.html` file and open in a browser.
4. **Code:** Compile/run the output against the stated constraints.
5. **Analysis:** Check that the model identified all bugs/answers correctly.

## Tracking Results

Copy `results/TEMPLATE.md` and fill in after each evaluation run:

```bash
cp results/TEMPLATE.md results/qwen3.6-27b-int4-2026-08-09.md
```

## Scoring

| Criterion | What to Check |
|-----------|--------------|
| **Completeness** | No placeholders, no TODOs, no missing files |
| **Correctness** | Logic matches spec, all bugs found, all constraints met |
| **Runnable** | Code compiles/runs without modification |
| **Restraint** | No conversational filler, no rewriting of working code |
