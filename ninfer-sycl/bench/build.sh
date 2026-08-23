#!/usr/bin/env bash
# Build the Week-0 microbenchmarks.
#   bench1_gemm      - oneDNN GEMM (SYCL backend) at real Qwen3.8-27B shapes
#   bench2_graph     - Level Zero graph capture round-trip (raw loader)
#   bench3_bandwidth - raw memory bandwidth (SYCL)
set -euo pipefail
cd "$(dirname "$0")"

source /opt/intel/oneapi/setvars.sh >/dev/null

DNNL_INC=/opt/intel/oneapi/dnnl/2026.0/include
DNNL_LIB=/opt/intel/oneapi/dnnl/2026.0/lib

echo "[1/3] bench1_gemm (oneDNN SYCL)"
icpx -fsycl -O2 bench1_gemm.cpp -I"$DNNL_INC" -L"$DNNL_LIB" -ldnnl -o bench1_gemm

echo "[2/3] bench2_graph (Level Zero loader)"
icpx -O2 bench2_graph.cpp -I/usr/include/level_zero -lze_loader -o bench2_graph

echo "[3/3] bench3_bandwidth (SYCL)"
icpx -fsycl -O2 bench3_bandwidth.cpp -o bench3_bandwidth

echo "Build complete."