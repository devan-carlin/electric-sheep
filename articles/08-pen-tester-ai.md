# Designing a Pen-Tester AI: RAG + LoRA + a Kali Agent

*How to build an agent that plans like a senior red-teamer, grounds every
technique in a knowledge base, knows what the defender will see, and runs real
tools — without handing an  model the keys to the keyboard.*

> **Status: concept draft.** This is a design, not a build log. The full
> architecture lives in `~/neon-demon/PENTEST-AI-CONCEPT.md`.

---

## The hook

"Build me a pen-tester AI" sounds like a model problem. It isn't. It's three
different problems glued together, and each has a different failure mode:

| Problem | Naive failure | The fix |
|---------|---------------|---------|
| **Knowledge** | Hallucinated techniques, stale CVEs | RAG over a curated, versioned corpus |
| **Reasoning** | Generic chatbot, no methodology | LoRA fine-tune + a structured pentest loop |
| **Execution** | Model runs arbitrary shell → injection RCE | Text-only RAG + a separate, allow-listed tool layer |

The single most important design decision: **the RAG service has no tools.**
Retrieval can read the corpus but cannot execute anything. The tool layer is a
separate, tightly-scoped component. This is the load-bearing control, because
the base model is ** — more compliant with embedded instructions
than a vanilla model — and therefore a *better* prompt-injection target.

## The three layers

**1. Knowledge (RAG).** A PhD-level offensive-security corpus (techniques,
methodologies, frameworks, lab scenarios) is chunked, embedded with a
multilingual model, and stored in a vector DB. Retrieval is hybrid — dense +
BM25, fused, then cross-encoder reranked. A second, separate collection holds
real-world SOC attack reports, which answer a different question: *"if I run
this, what will the defender see?"*

**2. Reasoning (the brain).** A 27B model with a 256K context window, plus an
optional LoRA fine-tune to internalize the depth. The 256K window is the
killer feature: an entire engagement — recon dump, every tool's output, the
running plan — fits in one context, so the agent never has to
summarize-and-forget mid-engagement.

**3. Execution (the hands).** A Kali VM holds the real tools — `nmap`,
`metasploit`, `burp`, `hydra`, `sqlmap`. The brain *plans*; Kali *executes* and
returns output. This is the agent loop, and it's the security-critical part.

## The pentest loop

Each phase is **plan → retrieve → execute → interpret → (detection check) →
next**:

1. **Recon** — plan OSINT, run the tools, build the target model.
2. **Enumerate** — read the scans, pick high-value targets.
3. **Exploit** — select a RAG-grounded technique (with source + MITRE ID), run
   it, interpret the result.
4. **Detect-aware** — *before* each action, query the SOC corpus: what will the
   defender see? Adapt — stay under radar, or confirm detection on purpose.
5. **Report** — emit findings with MITRE mapping, CVSS, evidence, remediation.

The detection-awareness step is the differentiator. Most "AI pentester"
projects stop at "here's how to exploit it." Knowing *what the SOC will see*
is what separates a good pen tester from a reciter — and it's exactly the kind
of thing a RAG layer over real telemetry can provide.

## The security controls (non-negotiable)

1. **RAG is text-only.** No tools. Retrieval can't execute.
2. **The tool layer is separate and scoped.** Allow-list of commands, no
   arbitrary shell, sandboxed working directory, no access to the brain's
   filesystem or credentials.
3. **Retrieved content is framed as untrusted** in the prompt — data, not
   instructions.
4. **An injection-marker quarantine pass** at ingest flags
   "ignore previous instructions"-style patterns before they enter the corpus.
5. **Binaries are skipped at ingest** — never parsed, never a vector.
6. **API-key auth, LAN-only** on the RAG service.
7. **Tool output is capped and framed** before it reaches the model, so a
   malicious tool result can't flood or hijack the context.

## What this is *not*

Not an autonomous weapon. It's a **tool for a human operator** who reviews
findings and makes the call. Not a general chatbot — a structured agent with a
defined loop and scoped capabilities. Not a replacement for a human pen tester
— it accelerates the grind (recon, enumeration, report writing) and forces
detection-awareness, but the judgment stays human.

## Takeaway

- **"Build an AI agent" is three problems: knowledge, reasoning, execution.**
  Solve each with a different tool, and keep them separate.
- **The retrieval layer must not be the execution layer.** That separation is
  the difference between a useful agent and an injection RCE.
- **Detection-awareness is the moat.** Knowing what the defender sees is what
  makes it a *pen tester* and not just a vulnerability reciter.
