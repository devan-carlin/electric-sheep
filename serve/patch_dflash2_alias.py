#!/usr/bin/env python3
"""Alias DFlash2DraftModel -> the pinned image's generic qwen3_dflash drafter.

This only gets the arch past the registry gate so we can OBSERVE how far the
DFlash2 weights get. The pinned DFlashQwen3Model is a plain Qwen3-shaped
drafter and lacks DFlash2's attention_conv / mlp_conv / candidate_selector,
so weight loading is expected to fail or drop those tensors. Diagnostic only.
"""
import re, sys

REG = "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/registry.py"
ANCHOR = '    "DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),'
ADD = '    "DFlash2DraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM"),'

with open(REG) as f:
    src = f.read()

if "DFlash2DraftModel" in src:
    print("[dflash2-alias] already present, no-op")
    sys.exit(0)

if ANCHOR not in src:
    print("[dflash2-alias] ERROR: anchor line not found", file=sys.stderr)
    sys.exit(1)

src = src.replace(ANCHOR, ANCHOR + "\n" + ADD, 1)
with open(REG, "w") as f:
    f.write(src)
print("[dflash2-alias] added DFlash2DraftModel alias to registry")
