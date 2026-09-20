# Swarm Architecture Harness

Two-machine inference fleet: one control plane, five endpoints.
Design doc. Last updated: 2026-09-05. Status: design, not built.

## 1. Purpose

Today the fleet is managed by hand: separate launchers per node, separate
machines, no shared view. This harness adds:

- One inventory of every inference node (machine, GPU, port, alias, role).
- One set of commands to start/stop/status the whole fleet.
- One routing layer so clients see a pool, not five URLs.
- Workload runners that exploit the pool (fan-out, ensembles, benchmarks).

Nothing here changes the models or the engines. It is glue around the
existing launchers.

## 2. Fleet topology

| Node | Machine | GPU | Standing service | Endpoint | Alias |
|------|---------|-----|------------------|----------|-------|
| B70-A | ai-server (Ubuntu) | GPU0 | ComfyUI A (swarm target: 27B 256k) | :8188 / :8000 | - / qwen-256k |
| B70-B | ai-server | GPU1 | ComfyUI B (swarm target: 27B 128k) | :8189 | - |
| B70-C | ai-server | GPU2 | llama.cpp 27B (qwen) | :8088 | qwen |
| B70-D | ai-server | GPU3 | llama.cpp 27B (gemma) | :8089 | gemma |
| WS-0 | Omen 45L (Windows) | RTX 5090 | llama.cpp / Ollama 27B int4 | tailnet `<ws>.ts.net:<port>` | qwen-256k (peer) |

Three operating states (the B70 box switches between the first two; WS-0
stays up in all of them):

- **Current default:** ComfyUI x2 + qwen + gemma. Mixed image/prose fleet.
- **Swarm (max LLM throughput):** all four B70 cards run 27B int4
  (GPU0 at 256k, GPU1-3 at 128k) + the 5090 node. Five concurrent 27B
  streams.
- **Escalation:** Flash-Next TP4 locks all four B70 cards (:8000,
  `qwen-256k`, MTP2) + the 5090 27B stream. Two streams, one of them the
  strongest model in the fleet.

Switch cost between swarm and escalation: stop the current stack, start
the other, ~2-4 min plus model load. The 5090 node never drops out.

## 3. Harness components

1. **Node registry** (`serve/fleet.json`)
   - Per node: machine, transport (local | tailnet), base_url, alias,
     context tier, engine, expected throughput, role.
   - Single source of truth for every other component.
2. **Health probe** (`fleet_health.py`)
   - GET `/v1/models` + a 16-token completion per node.
   - Reports up / down / degraded (slow TTFT); writes a timestamped
     status file the gateway and runners read.
3. **Fleet orchestrator** (`fleet_ctl.sh`)
   - `start | stop | status [default | swarm | escalation]` across both
     machines.
   - B70 side calls the existing launchers unchanged. 5090 side via SSH
     or a printed checklist (the Windows box is not always reachable).
4. **Routing gateway** (small FastAPI proxy, optional phase)
   - Exposes one OpenAI-compatible URL that fans out to the pool.
   - Policies: round-robin, sticky-per-session, by alias, by context tier
     (long prompts to 256k nodes), failover on probe failure.
   - Register in Open WebUI as a single connection.
5. **Workload runners** (`swarm/` scripts)
   - `fanout.py`: split a job into N chunks, dispatch to N nodes, merge.
   - `ensemble.py`: K samples across nodes and seeds, vote or judge.
   - `bench_fleet.py`: matched-prompt suite against any node, normalized
     results (tok/s, coherence probe, acceptance where applicable).

## 4. What the swarm accomplishes

Concrete patterns, with the math:

- **Parallel agents.** Five independent coding / research / chat sessions
  at once, each pinned to its own endpoint. Today this means juggling
  five base URLs; with the gateway it is one URL.
- **Map-reduce over big corpora.** A 200k-token document split into five
  40k chunks: each node processes its chunk in ~40k/80 s, about 8 min
  wall, versus ~40 min serial on one node. Merge step is trivial.
- **Self-consistency / ensembles.** Sample the same reasoning prompt on
  five nodes (or five seeds on one node), majority-vote the answer.
  Buys a few accuracy points on factual and reasoning tasks at the cost
  of 5x tokens.
- **Best-of-N drafting.** Generate five candidate answers in parallel
  (wall time equals one draft), then rank them with the strongest
  available model (Flash-Next during escalation windows).
- **Two-stage pipelines.** Cheap stages (extract, tag, chunk) on the 27B
  nodes; expensive stage (synthesize, verify) on the quality node.
- **Bulk data generation.** Synthetic-data and calibration-set throughput
  scales roughly linearly with node count. The parked MTP re-fit datagen
  ran at ~7 tok/s on one eager engine; five engines cut that wall time
  ~5x for any hook-free generation task.
- **A/B and regression suites.** Identical prompt battery against every
  node and every model swap, scored blind. Catches regressions when
  weights or engines change behind a stable alias.
- **Overflow absorption.** Burst traffic queues on the gateway and drains
  across all healthy nodes instead of stalling one endpoint.

## 5. Configuration comparison

### 5x27b swarm vs 1x27b + 1xFlash-Next

Feasibility note: Flash-Next needs all four B70 cards (TP4 is the memory
floor). So "1x27b + 1xFlash-Next" necessarily means the 27B lives on the
5090. Both configs spend the full hardware budget; the difference is what
the B70 box does.

| Dimension | 5x27b swarm | 1x27b + 1xFlash-Next |
|-----------|-------------|----------------------|
| Concurrent streams | 5 | 2 |
| Aggregate decode (est) | 4 x ~80 + ~70 = ~390 tok/s | ~75 + ~70 = ~145 tok/s |
| Quality ceiling | 27B abliterated on every stream | 125B MoE + PLE on the main stream |
| Context | 256k x2 (B70-0, 5090) + 128k x3 | 256k native x2 |
| Multimodal | 27B GGUF mmproj (image in, text out) | Full ViT tower (true vision) |
| Failure mode | Lose 1 of 5 (20% capacity) | Lose the box (main capability gone) |
| Flexibility | Any card repurposable freely | Box locked until Flash-Next stops |
| Power | 4 x 180W + 5090 | 4 x 180W + 5090 (identical) |
| Best for | Volume: agents, bulk, fan-out, ensembles | Depth: one hard problem, vision, long-doc synthesis |

Throughput figures: B70 27B int4 measured ~80 tok/s (llama.cpp slot;
vLLM 83.04 p512/g128 at 180W). 5090 27B int4 estimated 60-90 tok/s
(bandwidth bound: ~1792 GB/s over ~15 GB of weights). Flash-Next MTP2
measured 75.0-76.4 tok/s at temp 0, 69.82 at temp 0.5.

Reading the table:

- The ~2.7x throughput ratio favors the swarm, but it compares unlike
  things: the swarm's extra streams are 27B-class, while Flash-Next's
  single stream is the best answer quality in the fleet.
- The real decision variable is the shape of the day. Many medium tasks
  in parallel: swarm. One hard task that must be answered well:
  escalation.
- The configs are complements, not rivals. The 5090 node carries a 27B
  stream in both, so even in escalation mode a fast fallback stream
  stays alive while Flash-Next works.
- Recommended standing state: swarm. Escalate on demand; the switch is
  minutes, not hours.

## 6. Honest limits

- **No cross-machine KV sharing.** Prefix caching is per node. Repeated
  system prompts pay prefill on every node.
- **Tailnet dependency.** The 5090 leg rides on Tailscale stability and
  latency. Prefill-heavy or interactive jobs prefer the local B70 nodes;
  the 5090 suits steady decode and background work.
- **Sampling parity is approximate.** The same seed does not reproduce
  the same token stream across engines (llama.cpp and vLLM kernels
  differ). Fine for ensembles; not for bit-exact replay.
- **5090 setup is a prerequisite.** Needs a CUDA int4 27B build (see
  `build/windows/04-llama-cpp-deployment-guide.md`), bound to 0.0.0.0,
  reachable on the tailnet. Until then the fleet is four nodes.
- **The gateway adds a hop.** Local clients can bypass it and hit nodes
  directly; the proxy exists for unified and remote access.

## 7. Build plan

Phased; each phase is independently useful.

1. **Phase 0 (~1 h):** `fleet.json` registry + `fleet_health.py` probe.
   Immediate value: one screen showing the whole fleet.
2. **Phase 1 (half day):** `fleet_ctl.sh` orchestrator wrapping the
   existing launchers on both machines.
3. **Phase 2 (a day):** routing gateway + Open WebUI single-connection
   registration.
4. **Phase 3 (as needed):** workload runners (fanout, ensemble, bench).
5. **Phase 4 (later):** policies: auto-failover, context-tier routing,
   power-aware scheduling (idle nodes drop to a low power state).

Stop criterion: if manual management stops hurting (fewer than ~3 active
nodes most days), phases 2-4 can stay unbuilt.

## 8. Related docs

- `docs/architecture.md` : fleet topology, hardware, runtime paths.
- `serve/SERVICES.md` : per-service detail, second-node section.
- `serve/HANDOFF.md` : how to call the B70 endpoints.
- `build/common/tailscale-setup.md` : mesh network between the machines.
- `build/windows/04-llama-cpp-deployment-guide.md` : 5090 node setup.
- `electric-sheep/BACKLOG.md` : MTP re-fit (parked), kernel levers.
