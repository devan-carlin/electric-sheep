# vLLM Supports Intel Arc. Getting a 125B MoE Onto Four of Them Still Took Two Patches

*A 125B MoE model, 256K context, 53 tokens a second, no NVIDIA anywhere in the stack. The XPU backend is official. What isn't official is this model, or four discrete Arc cards sharing one PCIe root.*

---

The result first, because it sounds wrong:

| Metric | Value |
|--------|-------|
| Model | Qwen3.8-Flash-Next, 125B params (6B active, MoE) |
| Decode throughput | **53.4 tok/s** (single stream, 512 out, greedy) |
| Time to first token | ~0.11s |
| Context | 256K, fp8 KV cache |
| GPUs | 4x Intel Arc Pro B70, 32 GB each |
| Baseline (llama.cpp, same box) | ~27 tok/s |

Let me be precise about what is and isn't supported, because it matters. vLLM's XPU backend is real and official: Intel GPUs are a listed install target, there are official Docker images since v0.26.0, and there's a whole set of XPU attention backends. Nobody is hacking a fork into existence here.

What is *not* supported is the last mile. The specific model architecture has no upstream implementation, and the multi-GPU path assumes hardware properties that discrete Arc cards on desktop PCIe do not have. On this box the result beats the llama.cpp SYCL build by roughly 2x (53.4 vs ~27 tok/s, single stream, 512 tokens out, greedy, median of 3), holds 256K context, and has run for days as a daily driver.

Closing that last mile took exactly two modifications to vLLM. One is large and model-specific. The other is thirty lines and applies to anyone running multiple discrete Arc cards. This article is about both, and about why they exist.

## The setup

- 4x Intel Arc Pro B70 (Xe3), 32 GB each, 128 GB total VRAM
- AMD Threadripper PRO 3945WX, 256 GB RAM
- Ubuntu 26.04, oneAPI Base Toolkit, Level-Zero driver `20.2.0`
- vLLM built from upstream `main` at commit `c39076fef`, not the pip release
- torch `2.13.0+xpu`, `vllm-xpu-kernels 0.1.12` (pinned; `0.1.13.2` is broken)

The model is W4A16 quantized, so 128 GB of VRAM holds the weights with room left for a very large fp8 KV cache. That is what makes 256K context possible on consumer-class cards.

## Wall one: vLLM does not know this model exists

Stock vLLM answers a new architecture with one line:

```
ValueError: Model architecture ... is not supported
```

Qwen3.8-Flash-Next uses a hybrid layout vLLM has never seen: 36 layers of Gated DeltaNet linear attention interleaved with 12 full-attention layers, a 512-expert MoE block (top-10 plus one shared expert), and a residual mechanism with four parallel streams. None of it maps onto an existing model class.

The fix is a port: register the architecture in vLLM's model registry, add the config class, wire the W4A16 quantized linear kernels to the XPU backend, and implement the linear-attention forward path. Ours is roughly 2,800 lines across a dozen files. If you are serving a vanilla model (Llama, dense Qwen3), you skip this entire wall. It only exists because the model is new.

## Wall two: every worker sees every GPU

This one bites everyone who scales past two cards, and the error looks like a driver bug:

```
RuntimeError: oneCCL: ze_call.cpp:28 do_call: EXCEPTION: ze error at
zeMemOpenIpcHandle, code: ZE_RESULT_ERROR_INVALID_ARGUMENT
```

It fires at the first `all_reduce`, during memory profiling, before the server is ever healthy.

The cause is mundane. vLLM spawns one worker process per GPU. Each worker inherits the parent's `ZE_AFFINITY_MASK`, which on a 4-card box lists all four devices. So rank 0 initializes all four GPUs, and rank 1 initializes all four GPUs, and so on. Every rank opens IPC handles to devices it will never use, and Arc desktop PCIe cannot open those peer-memory handles. Crash.

The fix is thirty lines in the executor's spawn path: split the comma-separated mask, and give rank N exactly card N before the worker starts. With it, each rank initializes one device, host-RAM duplication disappears, and the IPC handles that used to be created simply are not.

That patch also explains a quieter symptom people hit before the hard crash: enormous CPU RAM usage and slow startup on multi-card boxes. Fewer devices per rank means far less duplicated state.

## The environment contract

The patches handle correctness. A handful of environment variables handle speed. Each one exists because something specific broke without it:

```bash
export VLLM_TARGET_DEVICE=xpu
export ZE_AFFINITY_MASK=0,1,2,3
export B70_WORKER_AFFINITY=1        # activates the spawn patch
export UR_L0_SYNC_MODE=BLOCKING
export VLLM_XPU_ENABLE_XPU_GRAPH=1  # the big one
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29530
export CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296
export CCL_SYCL_ALLTOALL_TMP_BUF=1
```

Three deserve commentary.

**Graph capture is the whole decode win.** A short A/B probe on this rig measured 7.0 tok/s with `--enforce-eager` against 43.0 tok/s with `VLLM_XPU_ENABLE_XPU_GRAPH=1`. Same model, same weights, same everything else. (These are short-probe numbers, not the 512-token benchmark in the table above, which lands higher at 53.4.) If your XPU vLLM feels slow, check this first.

**The oneCCL thresholds force the host-staged path for large collectives.** Discrete Arc cards on desktop PCIe cannot exchange peer-memory IPC handles for big messages. Telling oneCCL to stage through host memory above 4 GiB avoids hangs and crashes in allreduce. This is the number-one thing people miss when going from two cards to four.

**Do not set `ONEAPI_DEVICE_SELECTOR` alongside the per-worker masks.** In this torch build, any selector makes device initialization report zero devices once workers run with per-worker masks. The mask alone works. We hit this; it cost an evening.

## The launch

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <model-dir> \
  --served-model-name qwen-256k \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --dtype bfloat16 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --generation-config vllm \
  --override-generation-config '{"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"repetition_penalty":1.0}'
```

`--enable-expert-parallel` is MoE-only. Drop it for dense models. Until your fourth card arrives, run `--tensor-parallel-size 2` with `ZE_AFFINITY_MASK=0,1`; nothing else changes.

## Half of "bad quant" is bad sampling

That `--override-generation-config` flag above is doing more work than it looks like.

We chased looping and repetition collapse across several quants before realizing a large share of it was the sampler, not the weights. This model ships `temperature 1.0`, which degrades fast at long context. Pinning `temperature 0.7 / top_p 0.8 / top_k 20` plus `presence_penalty 1.5` made a real difference on an otherwise good quant.

Check your sampler before you blame the weights.

## Expect the RAM bill

The server sits at roughly 140 GB of the 256 GB. Most of it is one object: this model carries a lookup table for its n-gram layer, about 102 GB, memory-mapped from host RAM and shared across all four ranks through the page cache. It counts once, not four times. Add oneCCL staging buffers and per-rank pinned memory and you land where we do.

Practical consequences: do not run anything else heavy on that machine while it serves, and if you serve a model without such a table, the footprint drops substantially.

## Takeaway

The XPU backend is supported. The last mile is not. Stock vLLM will not run a new hybrid MoE architecture on four discrete Arc cards, and neither will stock vLLM plus only one of the two fixes. You need both: a model port so the engine knows the architecture, and a thirty-line spawn patch so each rank sees exactly one GPU.

Add the environment contract above, keep graph capture on, and the rest is a normal OpenAI-compatible server. Nothing about it is exotic once it is running, which is the whole point: the work is front-loaded into two patches and a handful of environment variables.

## Resources

- `qwen4exp-xpu-port.patch` and `b70-worker-affinity.patch` (the two patches, with apply instructions)
- The full launcher script, including the env contract and health checks
- `VLLM-XPU-SOURCE-BUILD.md` in the B70 inference cookbook for the full build, patch stack, and failure-mode table

<!-- TODO: replace the three bullets above with real repo links before publishing. -->
