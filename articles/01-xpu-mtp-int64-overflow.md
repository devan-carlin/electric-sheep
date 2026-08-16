# Your GPU Pointer Is Too Big for a 64-Bit Integer

*How a 27B model crashed on Intel Arc — and the 15-line fix that was hiding in plain sight.*

---

The error was absurd:

```
ValueError: Overflow when unpacking long long
```

It fired deep inside vLLM's speculative-decoding path, on a model that had been
running fine for weeks. No OOM. No NaN. No kernel fault. Just a Python integer
overflow, in a place where no one expected to do arithmetic on a number that
large.

The number in question was a **GPU memory pointer**.

## The setup

The model is Qwen 3.8 27B, a hybrid architecture: three layers of linear
attention (Gated DeltaNet) for every one layer of full attention, plus a single
multi-token-prediction (MTP) layer for speculative decoding. It runs on four
Intel Arc Pro B70 GPUs through the vLLM XPU fork, tensor-parallel across all
four.

We wanted to turn on MTP — the feature that's supposed to make generation
faster by drafting multiple tokens per step. The moment we did, the engine
crashed at startup with the overflow above.

## The root cause

The crash was in `mamba_utils.py`, in the code that prepares buffers for the
hybrid model's state. vLLM stores the base address of each state block in an
**int64 tensor**:

```python
self.state_base_addrs[idx] = state.data_ptr()
```

`data_ptr()` returns the device address of a tensor as a Python `int`. On CUDA
those addresses are small — a few gigabytes in, well under 2^63. On the Intel
XPU backend, they are not.

The XPU allocator hands out memory from a **high virtual-address region whose
top bit is set**. A real pointer we captured looked like this:

```
0xffffe000ff800000   =   18,446,708,893,624,041,472
```

The maximum value a signed 64-bit integer can hold is:

```
9,223,372,036,854,775,807   (2^63 - 1)
```

The pointer is **twice** that. Assigning it to an int64 tensor element makes
PyTorch's `unpack_longlong` raise `OverflowError`. The engine dies before it
serves a single token.

The subtle part: the value is *fine* as a 64-bit **bit pattern**. It only
breaks because the code chose to store it in a *signed* int64. The Triton
kernels that later consume these buffers never interpret the number as a
magnitude — they bitcast it straight back to a pointer:

```python
ptr = addr.to(tl.pointer_type(tl.float32))
```

So the information is not lost. It's just that the *storage* type was the wrong
signedness for this platform.

## The fix

Reinterpret the unsigned 64-bit pointer as a signed int64 before storing it.
In two's complement, any 64-bit bit pattern maps to exactly one signed int64,
and the mapping is reversible:

```python
def _ptr_to_i64(ptr: int) -> int:
    """Reinterpret an unsigned 64-bit device pointer as signed int64.

    On XPU, tensor.data_ptr() returns addresses in a high virtual-address
    region whose top bit is set (e.g. 0xffffe000ff800000), i.e. a value larger
    than int64 max. Assigning such a value to an int64 tensor element raises
    ValueError: Overflow when unpacking long long. The fused mamba kernels only
    ever bitcast these values back to pointers, so storing the two's-complement
    signed form is lossless and round-trips to the exact original address.
    """
    if ptr >= (1 << 63):
        ptr -= 1 << 64
    return ptr
```

Two call sites, one line each:

```python
self.state_base_addrs[idx] = _ptr_to_i64(state.data_ptr())
# ...
self.block_table_ptrs[i]   = _ptr_to_i64(bt.data_ptr())
```

That's the whole fix. Fifteen lines including the docstring.

## Why it round-trips exactly

The kernels never do `int64 → magnitude`. They do `int64 → pointer` via a
bitcast. A bitcast preserves the 64 bits regardless of how you *name* the
signedness. So:

```
0xffffe000ff800000  --(store as signed)-->  -274,877,906,944
        --(bitcast back to pointer)-->      0xffffe000ff800000
```

We verified this on the actual XPU: a synthetic in-range pointer passes
through unchanged, a real high pointer fails the naive assignment but
round-trips bit-for-bit through `_ptr_to_i64`. The address the kernel
ultimately dereferences is the exact one the allocator gave us.

## The catch: it's a venv-local patch

The fix lives in the installed `site-packages/vllm/v1/worker/mamba_utils.py`.
It is **not** in the vLLM source tree we build from, and it is **not** upstream.
Rebuild the venv, or `pip install` vLLM again, and it's gone.

That's the real lesson here. The bug is in upstream vLLM — the same two lines
exist in `vllm-project/vllm`'s `mamba_utils.py`. It simply has never been
triggered, because no one runs the fused mamba path on a backend that hands
out high virtual addresses. CUDA's addresses are small. The XPU's are not.

A proper fix upstream would either (a) store these as uint64, matching the
non-fused path that already does so, or (b) apply the same two's-complement
reinterpretation. The uint64 route is cleaner; the reinterpretation is smaller.

## Takeaway

- **A "too big for int64" error on a GPU pointer is a platform quirk, not a
  bug in your model.** If your backend allocates from a high virtual-address
  region, any code that stores `data_ptr()` in a signed int64 will break.
- **Check the signedness of your pointer storage.** If a value is only ever
  bitcast back to a pointer, store it as uint64 — or reinterpret it as signed
  int64. The bit pattern is what matters, not the magnitude.
- **Patches in `site-packages` are ephemeral.** If you fix something in an
  installed package, write down where, and consider sending it upstream. This
  one is a genuine upstream bug waiting for a reporter.
