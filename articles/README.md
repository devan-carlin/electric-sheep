# Electric Sheep — Article Library

Draft articles for publishing (Medium, etc.) drawn from the issues, optimizations,
and experiments in this project. Each is a working draft: the technical core is
real and verified, but each needs a final pass (voice, screenshots, a stronger
hook) before posting.

## House style

- No emojis. Plain language. Short paragraphs.
- Lead with the surprising result or the bug, not the setup.
- Show the numbers. A claim without a number is a rumor.
- Redact internal IPs (`192.168.x.x`) and usernames. Hardware specs are fine.
- Every article ends with a "Takeaway" a reader can actually use.

## The hardware (shared by most articles)

- **AI server:** 4× Intel Arc Pro B70 (Battlemage, ~32 GB each, `xe` driver),
  AMD Threadripper PRO 3945WX, 247 GiB RAM, Ubuntu 26.04.
- **Workstation:** NVIDIA RTX 5090 (32 GB, Blackwell), Windows.
- **The model at the center of most of these:** Qwen 3.8 27B — a 3:1
  Gated-DeltaNet (linear attention) / full-attention hybrid, 256K native context,
  one MTP (multi-token prediction) layer.

## Drafted (ready for a final pass)

| # | File | The hook |
|---|------|----------|
| 1 | `01-xpu-mtp-int64-overflow.md` | A 27B model crashes on Intel Arc because a GPU pointer is too big for a 64-bit integer. |
| 2 | `02-mtp-slower-on-xpu.md` | We fixed the crash — and found the "speedup" feature made the model 33% slower. |
| 3 | `03---on-xpu.md` | Removing a model's refusals without fine-tuning, on consumer Intel GPUs. |
| 4 | `04-256k-context-on-two-gpus.md` | Fitting a 256K context window on two 32 GB GPUs, and the concurrency math that comes with it. |

## Planned (outline + key facts, needs expansion)

| # | File | The hook |
|---|------|----------|
| 5 | `05-autoround-int4-on-xpu.md` | Quantizing a 27B hybrid model to 4-bit on Intel Arc, and the layers you must not touch. |
| 6 | `06-moe-topk-kernel-patch.md` | Patching vLLM's MoE kernel so a 35B-A3B model actually runs on XPU. |
| 7 | `07-rtx-5090-power-tuning.md` | What a 70% power cap does to an RTX 5090's real throughput. |
| 8 | `08-pen-tester-ai.md` | Designing a RAG + LoRA + Kali agent that pentests like a senior red-teamer. |

## Cross-references

- The  process is documented in `~/neon-demon/-PROCESS.md`.
- The pen-tester concept is designed in `~/neon-demon/PENTEST-AI-CONCEPT.md`.
- Quantization patch details live in `../docs/vllm/INT4-QUANTIZATION-PATCHES.md`.
- MoE patch details live in `../docs/guides/moe-topk-kernel-patch.md`.
