#!/usr/bin/env bash
# =============================================================================
# Phase 0 — Isolated XPU venv for Heretic abliteration
#
# Creates a throwaway venv with the Intel XPU PyTorch build + heretic-llm,
# WITHOUT touching the working vLLM venv (/home/dc/electric-sheep/vllm/.venv).
#
# Why isolated: heretic-llm pulls transformers/accelerate/peft/bitsandbytes/optuna
# and a plain `pip install` would grab CUDA torch. We install the XPU torch first
# from Intel's index so the venv is XPU-native.
#
# No GPU needed (pip + network only) — safe to run while the INT4 quant finishes.
# =============================================================================
set -euo pipefail

VENV=/home/dc/electric-sheep/vllm/heretic-venv
XPU_INDEX=https://download.pytorch.org/whl/xpu

echo "==> Creating venv at $VENV"
if [[ ! -d "$VENV" ]]; then
  python3.12 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing XPU PyTorch (Intel build) from $XPU_INDEX"
# Pin to the same versions as the working vLLM venv (torch 2.13.0+xpu).
pip install torch==2.13.0 torchvision==0.28.0 --index-url "$XPU_INDEX"

echo "==> Installing heretic-llm (deps resolve against the XPU torch above)"
pip install -U heretic-llm

echo "==> Verifying XPU torch survived the heretic-llm install"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.xpu.is_available():", torch.xpu.is_available())
assert "xpu" in torch.__version__, "ERROR: torch is not the XPU build!"
assert torch.xpu.is_available(), "ERROR: XPU not available (source set-env / check ONEAPI_DEVICE_SELECTOR)"
print("XPU devices:", torch.xpu.device_count())
PY

echo "==> Verifying heretic CLI"
# The console script is `heretic`; fall back to `python -m heretic` if absent.
if command -v heretic >/dev/null 2>&1; then
  heretic --help >/dev/null 2>&1 && echo "heretic CLI OK"
else
  echo "NOTE: 'heretic' not on PATH; will use 'python -m heretic' in run scripts."
fi

echo
echo "==> Done. Activate with:  source $VENV/bin/activate"
echo "    Then run:  bash heretic-probe-qwen3.5-4b.sh"
