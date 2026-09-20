#!/usr/bin/env python3
"""Host-staged TP allreduce for vLLM-XPU (load-time monkey patch).

WHY:
  Stock XpuCommunicator.all_reduce routes through oneCCL (xccl) on-device.
  On discrete Arc GPUs without P2P, oneCCL's SYCL runtime adds per-collective
  dispatch overhead on top of the actual reduction. For decode-sized messages
  (a few hundred KB to a few MB), staging partial results through pinned host
  memory and reducing on the CPU (gloo) is cheaper end-to-end. This rebuilds
  the "host-staged TP2 collectives" production delta from the LocalMaxxing
  141.3 tok/s Qwen3.8-27B run (private vLLM 0.28.1.dev1.xpu fork).

WHAT THIS PATCH DOES:
  Wraps XpuCommunicator.all_reduce. When B70_HOST_STAGED_AR=1 and the message
  is <= B70_HS_MAX_ELEMS elements (default 4M ~= 8 MiB bf16), it:
    1) D2H copies the tensor into a reusable pinned buffer,
    2) converts to fp32 on host (gloo has no bf16 allreduce),
    3) gloo all-reduce on the CPU group,
    4) converts back and H2D copies into a fresh device tensor.
  Larger messages (prefill chunks) and streams that are currently capturing a
  CUDA/XPU graph fall back to the stock oneCCL path (graph capture cannot
  contain CPU-side collectives).
  Default (env unset/0) = wrapper calls the original implementation, i.e. inert.

REQUIREMENTS:
  - A vLLM build with vllm/distributed/device_communicators/xpu_communicator.py
    (present in vllm-openai-xpu v0.28.0; older images skip gracefully).
  - A gloo cpu_group (always created by vLLM's parallel_state).

KILL SWITCHES:
  B70_HOST_STAGED_AR=0   -> stock behavior (wrapper passes through)
  B70_HS_MAX_ELEMS=N     -> max element count routed through host staging
"""
import os
import sys

SP = "/opt/venv/lib/python3.12/site-packages"
sys.path.insert(0, SP)

try:
    import torch
    import torch.distributed as dist
    from vllm.distributed.device_communicators.xpu_communicator import (
        XpuCommunicator,
    )
except Exception as e:
    print(f"[hstar] skipped ({type(e).__name__}: {e}); no XpuCommunicator here")
    sys.exit(0)

if getattr(XpuCommunicator, "_b70_hs_patched", False):
    print("[hstar] already patched, no-op")
    sys.exit(0)

_orig_all_reduce = XpuCommunicator.all_reduce


def _stream_sync():
    try:
        torch.xpu.current_stream().synchronize()
    except Exception:
        torch.xpu.synchronize()


def _capturing():
    try:
        return bool(torch.xpu.is_current_stream_capturing())
    except Exception:
        return False


class _HostStagedState:
    """Per-instance reusable pinned buffers (lazy, grow-on-demand)."""

    __slots__ = ("bf16", "f32")

    def __init__(self):
        self.bf16 = None
        self.f32 = None

    def ensure(self, n):
        cap = max(n, 8 << 20)  # start at 8M elems (16 MiB bf16 / 32 MiB f32)
        if self.bf16 is None or self.bf16.numel() < n:
            self.bf16 = torch.empty(cap, dtype=torch.bfloat16, pin_memory=True)
            self.f32 = torch.empty(cap, dtype=torch.float32, pin_memory=True)


def _host_staged_all_reduce(self, input_):
    if os.environ.get("B70_HOST_STAGED_AR", "0") != "1":
        return _orig_all_reduce(self, input_)
    if _capturing():
        # CPU-side collectives cannot run inside graph capture.
        return _orig_all_reduce(self, input_)
    cpu_group = getattr(self, "cpu_group", None)
    if cpu_group is None:
        return _orig_all_reduce(self, input_)
    n = input_.numel()
    if n == 0:
        return input_.clone()
    try:
        max_elems = int(os.environ.get("B70_HS_MAX_ELEMS", "4000000"))
    except ValueError:
        max_elems = 4_000_000
    if n > max_elems:
        return _orig_all_reduce(self, input_)
    st = getattr(self, "_b70_hs_state", None)
    if st is None:
        st = _HostStagedState()
        self._b70_hs_state = st
    st.ensure(n)
    flat = input_.reshape(-1)
    if input_.dtype == torch.bfloat16:
        pb = st.bf16[:n]
        pf = st.f32[:n]
        pb.copy_(flat, non_blocking=True)
        _stream_sync()
        pf.copy_(pb.to(torch.float32))
        dist.all_reduce(pf, group=cpu_group)
        out = input_.new_empty(input_.shape)
        out.reshape(-1).copy_(pf.to(torch.bfloat16), non_blocking=True)
        return out
    if input_.dtype == torch.float32:
        pf = st.f32[:n]
        pf.copy_(flat, non_blocking=True)
        _stream_sync()
        dist.all_reduce(pf, group=cpu_group)
        out = input_.new_empty(input_.shape)
        out.reshape(-1).copy_(pf, non_blocking=True)
        return out
    # Other dtypes: stay on the stock path.
    return _orig_all_reduce(self, input_)


XpuCommunicator.all_reduce = _host_staged_all_reduce
XpuCommunicator._b70_hs_patched = True
print(
    "[hstar] host-staged allreduce wrapper installed on XpuCommunicator "
    "(active when B70_HOST_STAGED_AR=1)"
)
