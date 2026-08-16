#!/usr/bin/env python3
"""Test auto_round INT4 quantization on XPU with the K mismatch patch."""

import os
import time
from auto_round import AutoRound

MODEL_PATH = "/home/dc/electric-sheep/models/DeepSeek-V4-Flash-0731--FP8"
OUTPUT_DIR = "/mnt/data/models/DeepSeek-V4-Flash-0731--INT4-xpu"

print(f"Model: {MODEL_PATH}")
print(f"Output: {OUTPUT_DIR}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"XPU available: {torch.xpu.is_available()}")
print()

# Load and quantize on XPU
print("[1/3] Loading model...")
t0 = time.time()
model = AutoRound(
    MODEL_PATH,
    scheme="INT4",
    group_size=128,
    sym=False,
    iters=10,
    enable_quanted_input=True,
    batch_size=4,
    amp=True,
    device_map="auto",  # Will use XPU + CPU for 156GB model
    low_cpu_mem_usage=True,
    seed=42,
)
print(f"  Model loaded in {time.time() - t0:.0f}s")

print()
print("[2/3] Running calibration and quantization...")
t1 = time.time()
model.quantize_and_save(OUTPUT_DIR)
print(f"  Quantization complete in {time.time() - t1:.0f}s")

print()
print("[3/3] Done! Output saved to:")
print(f"  {OUTPUT_DIR}")
