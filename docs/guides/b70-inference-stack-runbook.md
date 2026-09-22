# B70 Inference Stack — Ops Runbook (Qwen3.8-27B, guide-spec vLLM)

Current best configuration for serving Qwen3.8-27B on the Intel Arc Pro B70
fleet, measured 2026-09-04. This is the setup behind the Open WebUI on :3000.

## What is running

| Container | Port | GPU | Served name | Draft path | Decode (p512/g128, 180 W) |
|---|---|---|---|---|---|
| `qw38speed-int4` | :8001 | GPU1 (renderD131) | `qwen-256k-int4` | INT4 (S+M1 overlay) | **81.46 tok/s** |
| `qw38speed` | :8000 | GPU0 (renderD130) | `qwen-256k` | BF16 | 54.15 tok/s |
| `open-webui` | :3000 | - | - | - | points at :8001 |

- WebUI talks to instance B (`qwen-256k-int4`) via `OPENAI_API_BASE_URL=http://host.docker.internal:8001/v1`.
- Instance A is kept as a fallback / A/B reference.
- All four GPUs capped at **180 W** (persistent across reboots).

## Key facts

- Image: `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
  (vLLM 0.27.2rc1.dev77+gac7509e2b.xpu, vllm-xpu-kernels 0.1.12.3). Pinned by digest, stored locally.
- Model: `/mnt/data/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` (GPTQ Int4 sym g128 target + BF16 MTP draft tensors).
- Context window: `--max-model-len 131072` (128K) on both instances. KV cache fp8.
  - A: 141,882 KV tokens (1.08x a full 128K request)
  - B: 154,043 KV tokens (1.18x) — INT4 draft frees ~2 GB, more goes to KV
- The alias `qwen-256k` overstates reality: single-card ceiling is ~128-150K. True 256K needs TP=2.

## Patches applied inside each container (in order)

1. `patch_mtp_nightly.py` — MTP draft/verify path optimization
2. `patch_mtp_boundary.py` — GDN attention boundary fix
3. `patch_gdn_mixed_split_v5.py` — **required**: lets spec-decode + non-spec (prefill) tokens share one batch. Without it, two concurrent requests crash EngineCore (`causal_conv1d does not support spec-decode and non-spec tokens in the same invocation`).
4. (instance B only) `patch_draft_lmhead_int4.py` + `patch_draft_mtp_int4.py` — S+M1 draft-INT4 overlay, gated by env `B70_DRAFT_LMHEAD_INT4=1` + `B70_DRAFT_MTP_INT4=1`.

Patches live in `~/intel-arc-pro-b70-inference-cookbook/patches/`. Note: the two draft-INT4 patch hashes in the cookbook docs are stale; the repo HEAD (2026-09-02, "fix: guard draft INT4 under tensor parallelism") has newer versions. Use the clean checkout as-is.

## Launching (manual, e.g. after a fresh clone or if a container is removed)

Launcher: `serve/run-guide-docker-gpu1.sh {start|logs|health|stop}`. Env overrides:
`B70_RENDER_NODE`, `B70_AFFINITY_MASK`, `GUIDE_HOST_PORT`, `GUIDE_SERVED_NAME`,
`GUIDE_CONTAINER_NAME`, `DRAFT_INT4` (0/1).

```bash
cd ~/electric-sheep/serve

# Instance A (baseline, GPU0)
B70_RENDER_NODE=/dev/dri/renderD130 B70_AFFINITY_MASK=0 \
GUIDE_HOST_PORT=8000 GUIDE_SERVED_NAME=qwen-256k \
GUIDE_CONTAINER_NAME=qw38speed DRAFT_INT4=0 \
bash run-guide-docker-gpu1.sh start

# Instance B (draft-INT4, GPU1)
B70_RENDER_NODE=/dev/dri/renderD131 B70_AFFINITY_MASK=1 \
GUIDE_HOST_PORT=8001 GUIDE_SERVED_NAME=qwen-256k-int4 \
GUIDE_CONTAINER_NAME=qw38speed-int4 DRAFT_INT4=1 \
bash run-guide-docker-gpu1.sh start

# WebUI (points at B)
docker rm -f open-webui 2>/dev/null
docker run -d --name open-webui -p 3000:8080 \
  --add-host host.docker.internal:host-gateway \
  -e OPENAI_API_BASE_URL="http://host.docker.internal:8001/v1" \
  -e OPENAI_API_KEY="EMPTY" \
  -v open-webui:/app/backend/data \
  ghcr.io/open-webui/open-webui:main
```

GPU render-node map (stable across reboots on this box):
GPU0=renderD130 (mask 0), GPU1=renderD131 (mask 1), GPU2=renderD128 (mask 2), GPU3=renderD129 (mask 3).

## Benchmark

```bash
python3 serve/bench_guide_p512g128.py http://127.0.0.1:8001 qwen-256k-int4   # instance B
python3 serve/bench_guide_p512g128.py http://127.0.0.1:8000 qwen-256k         # instance A
```
Guide-exact p512/g128, temp 0, concurrency 1, streaming. Reports median/mean/min/max decode + median TTFT.

## Power management

- Fleet-wide 180 W cap. Applied on demand by `serve/apply-gpu-power-caps.sh`
  (matches GPUs by hwmon `name == "xe"`, never index; writes `CAP_UW` to each `power1_cap`).
  The companion `arc-b70-power-caps.service` exists but is DISABLED per user preference
  (nothing should auto-start after reboot). Re-enable only if wanted:
  `sudo systemctl enable arc-b70-power-caps.service`.
- Change the cap: edit `CAP_UW` in `serve/apply-gpu-power-caps.sh`, then
  `sudo ~/electric-sheep/serve/apply-gpu-power-caps.sh` (passwordless).
- Measured curve (GPU0, p512/g128): 160 W = 53.1, 180 W = 54.2, 200 W = 54.6, 230 W = 55.1 tok/s.
  Gain is front-loaded 160->180 W; beyond that <1% per 30 W (bandwidth-bound). 180 W chosen as the sweet spot.
- One-time root bootstrap (already done): `sudo bash serve/install-power-caps-service.sh`.
  It installs the sudoers drop-in `/etc/sudoers.d/dc-gpu-persist` (lets dc run the apply-script + systemctl passwordless) and enables the service.

## Reboot behavior (manual bring-up, per user preference)

Nothing LLM/WebUI-related auto-starts after a reboot (all containers use
`--restart=no`; `arc-b70-power-caps.service` is disabled). After boot, to restore
the stack:

```bash
# 1. power caps (reset to 160 W default on reboot)
sudo ~/electric-sheep/serve/apply-gpu-power-caps.sh

# 2. engines (see Launching section above)
cd ~/electric-sheep/serve
B70_RENDER_NODE=/dev/dri/renderD130 B70_AFFINITY_MASK=0 GUIDE_HOST_PORT=8000 \
  GUIDE_SERVED_NAME=qwen-256k GUIDE_CONTAINER_NAME=qw38speed DRAFT_INT4=0 \
  bash run-guide-docker-gpu1.sh start
B70_RENDER_NODE=/dev/dri/renderD131 B70_AFFINITY_MASK=1 GUIDE_HOST_PORT=8001 \
  GUIDE_SERVED_NAME=qwen-256k-int4 GUIDE_CONTAINER_NAME=qw38speed-int4 DRAFT_INT4=1 \
  bash run-guide-docker-gpu1.sh start

# 3. webui (same docker run as in Launching section)
```

Model dir, cookbook, and both images persist on disk. Allow ~5 min per engine
(weight load + torch.compile + XPU graph capture) before `/health` returns OK.

## Troubleshooting

- Engine slow to come up: normal, ~5 min. Watch `docker logs -f <container>`. Ready when you see `Application startup complete` and `/health` returns 200.
- Garbled output: check the model loaded cleanly (`Using XPUwNa16LinearKernel for AutoGPTQLinearMethod` in logs). If factual completions are corrupted, that's the parked garbage-text thread (MTP-on-this-host vs PCIe fabric), separate from throughput.
- Concurrent-request crash (`causal_conv1d ... mutually exclusive`): the v5 patch isn't applied. Confirm `patch_gdn_mixed_split_v5.py` ran (log line `[gdn-split-v5] patched ...`).
- oneCCL/XCCL `ze_fd_manager` error on launch: add `--cap-add SYS_PTRACE --security-opt seccomp=unconfined --ipc=host` to the `docker run`.
- Flip WebUI back to instance A: recreate the container with `OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1`.
