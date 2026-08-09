# LLM Stress Tests

A battery of prompts designed to stress-test LLM code generation across two categories: **game development** (single-file HTML apps) and **real-world benchmarks** (strict output compliance, logic constraints, backend architecture, code auditing, and agentic planning).

Each prompt forces the model to produce complete, working output — no external dependencies, no placeholders, no broken logic.

---

## Game Prompts (`game-prompts/`)

Single-file HTML applications that test canvas physics, state machines, and UI logic.

| # | Test | What It Probes |
|---|------|----------------|
| 1 | [Falling Words Typing Test](./game-prompts/01-typing-speed-test.md) | Real-time input handling, collision detection, analytics (WPM/accuracy), localStorage persistence, progressive difficulty |
| 2 | [Wordle Clone](./game-prompts/02-wordle-clone.md) | Strict duplicate-letter evaluation logic, virtual keyboard state sync, animation, win/loss detection |
| 3 | [2048 Engine](./game-prompts/03-2048-engine.md) | Matrix shift/merge algorithm, game-over detection (full board + no valid merges), tile spawning probability |
| 4 | [Brick Breaker](./game-prompts/04-brick-breaker.md) | Canvas physics, dynamic paddle-bounce angle, brick grid collision, lives/win/loss state |
| 5 | [Flappy Bird](./game-prompts/05-flappy-bird.md) | Gravity physics, procedural pipe generation, collision detection, bird rotation, localStorage high score |
| 6 | [Sudoku](./game-prompts/06-sudoku.md) | Backtracking generator/solver, uniqueness constraint, conflict highlighting, notes/pencil mode, undo stack, win detection |

---

## Real-World Tests (`real-world-tests/`)

Tests that isolate specific failure modes — context drift, strict output compliance, logical reasoning, and state tracking.

| # | Test | Target Capability |
|---|------|-------------------|
| 1 | [Strict JSON Parser](./real-world-tests/01-strict-json-parser.md) | Deterministic formatting, edge-case handling, zero conversational fluff |
| 2 | [Shift Scheduler](./real-world-tests/02-shift-scheduler.md) | Multi-variable constraint satisfaction, negative rules, self-auditing |
| 3 | [FastAPI Microservice](./real-world-tests/03-fastapi-microservice.md) | Single-file production code, DB schema, async handling, REST standards |
| 4 | [Code Audit — Bug Hunt](./real-world-tests/04-code-audit-bug-hunt.md) | Subtle bug detection, restraint (fix only the broken function) |
| 5 | [Agentic Function Calling](./real-world-tests/05-agentic-function-calling.md) | Multi-step orchestration, conditional tool execution, structured JSON planning |

---

## How to Use

1. Copy the full prompt from a `.md` file.
2. Paste it into the LLM you want to test.
3. **Games:** Save the returned HTML as a `.html` file and open in a browser.
4. **Benchmarks:** Validate the output against the stated criteria (valid JSON, constraint satisfaction, runnable code, etc.).

## Scoring

Rate each LLM output on a 1–5 scale per criterion:

| Criterion | Games | Benchmarks |
|-----------|-------|------------|
| **Completeness** | No placeholders, no TODOs, no missing code | All required fields/endpoints/rules present |
| **Correctness** | Game logic matches the spec | Output conforms to schema / constraints satisfied |
| **Playability** | Runs in browser without console errors | Code runs / JSON parses / plan is executable |
| **Restraint** | N/A | No conversational filler; no rewriting of working code |
