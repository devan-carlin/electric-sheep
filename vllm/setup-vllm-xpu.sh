#!/usr/bin/env bash
# =============================================================================
# setup-vllm-xpu.sh — one-shot installer for the qwen4_exp (Qwen3.8-Flash-Next)
# vLLM XPU build on Intel Arc Pro B70.
#
# What it does:
#   1. Installs the Rust toolchain (vLLM's frontend is Rust).
#   2. Creates a fresh Python 3.12 venv.
#   3. Installs torch 2.13.0+xpu + vllm-xpu-kernels 0.1.12 + common deps
#      (from the fork's requirements/xpu.txt).
#   4. Builds + installs the patched vLLM from the fork branch.
#   5. Smoke-tests that the qwen4_exp arch is registered.
#
# Prereqs on the host:
#   - Intel oneAPI Base + Runtime (setvars.sh at /opt/intel/oneapi/setvars.sh)
#   - python3.12, cmake, ninja, a C++ compiler (icx from oneAPI)
#   - 4x Arc Pro B70 (or other XPU) + >=128GB host RAM (for the PLE table)
#
# Usage:
#   bash setup-vllm-xpu.sh                 # clones the fork branch, builds
#   VLLM_SRC=/path/to/vllm bash setup-vllm-xpu.sh   # build from a local tree
#
# After it succeeds, serve with:
#   source <venv>/bin/activate
#   python -m vllm.entrypoints.openai.api_server --model <W4A16-ckpt> ...
#   (see the operations doc for the full flag set)
# =============================================================================
set -uo pipefail

FORK_BRANCH="${VLLM_FORK_BRANCH:-xpu-qwen4exp}"
FORK_URL="${VLLM_FORK_URL:-https://github.com/devan-carlin/vllm.git}"
VENV="${VENV:-$HOME/vllm-xpu-venv}"
SRC="${VLLM_SRC:-}"

echo "=== setup-vllm-xpu start $(date) ==="

# 0. oneAPI (compiler + runtime). Adjust path if installed elsewhere.
ONEAPI="${ONEAPI:-/opt/intel/oneapi/setvars.sh}"
if [ -f "$ONEAPI" ]; then
  source "$ONEAPI" >/dev/null 2>&1
  echo "--- oneAPI loaded ---"
else
  echo "WARN: oneAPI setvars not found at $ONEAPI. Set ONEAPI=/path/to/setvars.sh"
fi

# 1. Rust toolchain (skip if present)
if ! command -v cargo >/dev/null 2>&1; then
  echo "--- installing rustup ---"
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
  export PATH="$HOME/.cargo/bin:$PATH"
fi
command -v cargo >/dev/null 2>&1 || { echo "ERROR: cargo not available after install"; exit 1; }
cargo --version

# 2. Source tree: use VLLM_SRC if given, else clone the fork branch
if [ -z "$SRC" ]; then
  SRC="$HOME/vllm-xpu-src"
  if [ -d "$SRC/.git" ]; then
    echo "--- updating existing clone at $SRC ---"
    git -C "$SRC" fetch -q origin "$FORK_BRANCH" && git -C "$SRC" checkout -q "$FORK_BRANCH" && git -C "$SRC" reset -q --hard "origin/$FORK_BRANCH"
  else
    echo "--- cloning $FORK_URL @ $FORK_BRANCH ---"
    rm -rf "$SRC"
    git clone -q --branch "$FORK_BRANCH" "$FORK_URL" "$SRC"
  fi
fi
echo "--- source: $SRC ---"

# 3. Fresh venv
rm -rf "$VENV"
python3.12 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
echo "--- venv: $("$VENV/bin/python" --version) ---"

# 4. Core deps (torch 2.13.0+xpu, vllm-xpu-kernels 0.1.12, common)
"$VENV/bin/pip" install -q -r "$SRC/requirements/xpu.txt"
echo "--- core deps installed ---"

# 5. Build + install patched vLLM (the long step)
cd "$SRC"
export VLLM_TARGET_DEVICE=xpu
echo "--- building vLLM (long) ---"
"$VENV/bin/pip" install --no-build-isolation --no-deps .
echo "--- vLLM installed ---"

# 6. Smoke test
cd /tmp
"$VENV/bin/python" - <<'EOF'
import vllm
print("vllm version:", vllm.__version__)
from vllm.model_executor.models.registry import ModelRegistry
archs = set(ModelRegistry().get_supported_archs())
need = {"Qwen4ExpForCausalLM", "Qwen4ExpMTP"}
assert need <= archs, f"MISSING qwen4_exp archs: {need - archs}"
print("qwen4_exp archs registered: OK")
print("SMOKE PASS — venv ready at", __import__("sys").argv[0] if False else "")
EOF

echo "=== setup-vllm-xpu done $(date) ==="
echo "Activate with:  source $VENV/bin/activate"