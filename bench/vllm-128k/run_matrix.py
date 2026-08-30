#!/usr/bin/env python3
"""Benchmark int4 -ara at 128k context window across TP=1/2/4.

For each TP: launch vLLM (max-model-len=131072, fp8 KV, prefix caching,
util 0.85 — same flags as the baseline tp2 script), wait for /health, then
send 3 prompts of 4096 tokens (seed-rotated so each is a fresh prefill)
and measure TTFT (prefill throughput) + decode tok/s. Averages runs[1:]
(drops warmup).
"""
import json
import os
import signal
import subprocess
import sys
import time

VENV = "/home/dc/vllm-fresh-venv"
PY = f"{VENV}/bin/python"
MODEL = ("/home/dc/electric-sheep/models/"
         "Qwen3.8-27B--ara-int4-AutoRound/"
         "Qwen3.8-27B--ara-w4g128")
CLIENT = "/home/dc/electric-sheep/bench/vllm-128k/client.py"
PORT = 8000
MAX_LEN = 131072   # 128k context window (as requested)
PROMPT = 4096      # long enough for a stable prefill-throughput number, not a full 128k
GEN = 128
RUNS = 3

TPS = [1, 2, 4]

# KV-cache need scales with TP (KV heads sharded). TP=1 holds all 16.6 GiB of
# weights + the full 4.3 GiB of 128k KV on one card -> needs 0.96. TP=2/4 shard
# both, so 0.90 leaves ample KV headroom AND avoids the razor-thin 0.96 margin
# that fails WorkerProc init when the worker's own XPU context eats ~1.3 GiB.
UTIL = {1: 0.96, 2: 0.90, 4: 0.90}


def xpu_env(tp):
    """Same device-selection semantics as start-qwen.sh: affinity mask picks
    physical GPUs 0..tp-1 and renumbers them; selector uses 0..tp-1."""
    gpus = ",".join(str(i) for i in range(tp))
    return {
        **os.environ,
        "ZE_AFFINITY_MASK": gpus,
        "ONEAPI_DEVICE_SELECTOR": f"level_zero:{gpus}",
        "UR_L0_SYNC_MODE": "BLOCKING",
        "TORCH_LLM_ALLREDUCE": "1",
        "CCL_ZE_IPC_EXCHANGE": "pidfd",
        "CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK": "0",
        "VLLM_XPU_FORCE_GRAPH_WITH_COMM": "1",
        "VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ENGINE_ITERATION_TIMEOUT_S": "300",
        "TRITON_CACHE_DIR": os.path.expanduser("~/.cache/triton"),
        "UVICORN_KEEP_ALIVE_TIMEOUT": "300",
        "VLLM_XPU_ENABLE_XPU_GRAPH": "1",
        "VLLM_TARGET_DEVICE": "xpu",
        "VLLM_CACHE_ROOT": os.path.expanduser("~/.cache/vllm"),
    }


def wait_gpus_free(tp, min_free_gib=29.5, timeout_s=300):
    """Wait until GPUs 0..tp-1 each report >= min_free_gib free. The previous
    server's XPU teardown can lag 10-30s after SIGTERM, and util 0.96 needs
    29.08 GiB free at startup — launching too early fails WorkerProc init."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        r = subprocess.run(
            [PY, "-c",
             "import torch;print(' '.join(str(torch.xpu.mem_get_info(i)[0]//1024**3)"
             " for i in range(torch.xpu.device_count())))"],
            capture_output=True, text=True, timeout=120)
        try:
            free = [int(x) for x in r.stdout.split()]
            if all(free[i] >= min_free_gib for i in range(tp)):
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def launch_server(tp):
    cmd = [
        PY, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--served-model-name", "qwen-128k",
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--tensor-parallel-size", str(tp),
        "--max-model-len", str(MAX_LEN),
        "--kv-cache-dtype", "fp8",
        "--enable-prefix-caching",
        "--trust-remote-code",
        # Per-TP util (see UTIL): TP=1 needs 0.96 for the full 128k KV on one
        # card; TP=2/4 shard KV so 0.90 is plenty and avoids the 0.96 margin
        # that trips WorkerProc init (worker XPU context eats ~1.3 GiB).
        "--gpu-memory-utilization", str(UTIL[tp]),
        "--block-size", "32",
        "--max-num-batched-tokens", "16384",
        "--max-num-seqs", "8",
        "--language-model-only",
        "--quantization", "auto-round",
        "--generation-config", "vllm",
        "--enable-auto-tool-choice",
        "--reasoning-parser", "qwen3",
        "--tool-call-parser", "qwen3_coder",
    ]
    log = open(f"/tmp/vllm-128k-tp{tp}.log", "w")
    proc = subprocess.Popen(cmd, env=xpu_env(tp), stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True)
    return proc, log


def wait_health(timeout_s=900):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        try:
            r = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                 f"http://127.0.0.1:{PORT}/health"],
                capture_output=True, text=True, timeout=10)
            if r.stdout.strip() == "200":
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def teardown(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=60)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    time.sleep(20)  # XPU teardown lags ~10s after SIGTERM; give it margin


def main():
    tps = [int(x) for x in sys.argv[1:] if x.isdigit()] or TPS
    results = {}
    try:  # merge with an existing matrix (partial re-runs)
        with open("/tmp/vllm-128k-matrix.json") as f:
            results = json.load(f)
    except Exception:
        pass
    for tp in tps:
        key = f"tp{tp}"
        print(f"\n########## {key}  (GPUs 0..{tp-1}, max-model-len={MAX_LEN}) ##########",
              flush=True)
        if not wait_gpus_free(tp):
            print("   !! GPUs never reached the free-memory threshold", flush=True)
            results[key] = {"error": True, "reason": "gpus not free"}
            continue
        proc, log = launch_server(tp)
        t0 = time.perf_counter()
        ok = wait_health()
        load_s = round(time.perf_counter() - t0, 1)
        if not ok:
            print(f"   !! server did not become healthy in {load_s}s", flush=True)
            print("   log tail:", flush=True)
            for line in open(f"/tmp/vllm-128k-tp{tp}.log").readlines()[-15:]:
                print("   ", line.rstrip(), flush=True)
            teardown(proc)
            log.close()
            results[key] = {"error": True, "load_s": load_s}
            continue
        print(f"   healthy in {load_s}s", flush=True)

        runs = []
        for i in range(RUNS):
            out = f"/tmp/vllm-128k-{key}-run{i}.json"
            r = subprocess.run(
                [PY, CLIENT, "--model", MODEL, "--served", "qwen-128k",
                 "--prompt-tokens", str(PROMPT), "--gen-tokens", str(GEN),
                 "--seed", str(i), "--out", out],
                capture_output=True, text=True, timeout=1800)
            if r.returncode != 0:
                print(f"   run{i} FAILED rc={r.returncode}", flush=True)
                print("   ", (r.stderr or r.stdout).strip().splitlines()[-3:], flush=True)
                runs.append({"run": i, "error": True})
                continue
            d = json.loads(r.stdout.strip().splitlines()[-1])
            d["run"] = i
            runs.append(d)
            print(f"   run{i}: ttft={d['ttft_s']}s prefill={d['prefill_tok_s']} "
                  f"decode={d['decode_tok_s']} tok/s wall={d['wall_s']}s", flush=True)

        teardown(proc)
        log.close()

        good = [r for r in runs[1:] if not r.get("error")]
        if good:
            avg = lambda k: round(sum(r[k] for r in good) / len(good), 2)
            results[key] = {
                "load_s": load_s,
                "prompt_tokens": good[0]["prompt_tokens"],
                "gen_tokens": good[0]["gen_tokens"],
                "avg_ttft_s": avg("ttft_s"),
                "avg_prefill_tok_s": avg("prefill_tok_s"),
                "avg_decode_tok_s": avg("decode_tok_s"),
                "runs": runs,
            }
        else:
            results[key] = {"load_s": load_s, "error": True, "runs": runs}

    with open("/tmp/vllm-128k-matrix.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n\n=================== 128k CONTEXT MATRIX ===================")
    print(f"{'config':8s} {'ttft s':>8s} {'prefill tok/s':>14s} {'decode tok/s':>13s} {'load s':>8s}")
    for key in [f"tp{tp}" for tp in tps]:
        d = results.get(key, {})
        if d.get("error"):
            print(f"{key:8s} {'ERROR':>8s}")
        else:
            print(f"{key:8s} {d['avg_ttft_s']:>8} {d['avg_prefill_tok_s']:>14} "
                  f"{d['avg_decode_tok_s']:>13} {d['load_s']:>8}")
    print("\nwrote /tmp/vllm-128k-matrix.json")


if __name__ == "__main__":
    main()
