#!/usr/bin/env python3
"""Test FP8 K mismatch fix on XPU — load model and run a forward pass."""

import torch
import time

MODEL_PATH = "/home/dc/electric-sheep/models/DeepSeek-V4-Flash-0731-Abliterated-FP8"

print(f"PyTorch: {torch.__version__}")
print(f"XPU available: {torch.xpu.is_available()}")
print(f"Model: {MODEL_PATH}")
print()

# Load model on XPU
print("[1/2] Loading model on XPU...")
t0 = time.time()
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # Split across XPU + CPU (model is 156GB, VRAM is 128GB)
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
print(f"  Loaded in {time.time() - t0:.0f}s")

# Run a forward pass
print()
print("[2/2] Running forward pass (this will hit the FP8 matmul path)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
inputs = tokenizer("Hello, world!", return_tensors="pt").to("xpu")

t1 = time.time()
with torch.no_grad():
    outputs = model(**inputs)
print(f"  Forward pass in {time.time() - t1:.2f}s")
print(f"  Output shape: {outputs.logits.shape}")
print()
print("SUCCESS — no K mismatch error!")
