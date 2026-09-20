#!/usr/bin/env python3
"""Install the MTP re-fit wide-state tracer (v5: _preprocess id capture).

Why v5: v4 stashed input_ids in gpu_model_runner._model_forward, but the
runner sets input_ids=None in _preprocess (VL/multimodal path passes
inputs_embeds) BEFORE _model_forward runs, so the stash held None. v5
captures the real ids from self.input_ids.gpu inside _preprocess, where
they are still valid, and stashes positions in _model_forward.

Two coordinated edits:

1. vllm/v1/worker/gpu_model_runner.py
   - _preprocess: after num_scheduled_tokens is known, stash
     self.input_ids.gpu[:num_scheduled_tokens] into qwen4_exp.datagen_stash.
   - _model_forward: stash positions into qwen4_exp.datagen_stash.

2. vllm/model_executor/models/qwen4_exp.py -- in Qwen4ExpModel.forward, after
   the existing wide-buffer fill, append (token_ids, positions, wide_state)
   for this step to B70_MTP_DATAGEN_TRACE (rank 0 only). Token ids come from
   the runner stash; sliced to residual.shape[0] to drop any SP padding.

Driven entirely by env vars (inherited by spawned TP workers):
  B70_MTP_DATAGEN_TRACE=/path/to/trace.bin   enable tracing, write here
  B70_MTP_DATAGEN=0                          runtime kill switch (default 1)

Record format (little-endian, one per forward step, rank 0 only):
  u32 n_tokens | u32 wide_dim | i32 token_ids[n] | i64 positions[n]
  | u16 wide[n*wide_dim]  (raw bfloat16 bits)

Idempotent: guarded by marker B70_MTP_DATAGEN_HOOK.

Usage:
  B70_VLLM_ROOTS=/home/dc/vllm-fresh-venv/lib/python3.12/site-packages/vllm \
    python3 patch_mtp_datagen_hook.py
"""
import os
import py_compile
import sys

ROOTS = [r.strip() for r in os.environ.get(
    "B70_VLLM_ROOTS",
    "/home/dc/vllm-fresh-venv/lib/python3.12/site-packages/vllm").split(",") if r.strip()]

MARKER = "B70_MTP_DATAGEN_HOOK"

# ---- edit 1a: gpu_model_runner._preprocess id capture ----------------------
PRE_ANCHOR = (
    "        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens\n"
    "        is_first_rank = get_pp_group().is_first_rank\n"
)
PRE_INSERT = (
    "        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens\n"
    "        # [B70_MTP_DATAGEN_HOOK] Capture this step's real token ids now,\n"
    "        # before the VL/multimodal path below drops them (input_ids=None).\n"
    "        try:\n"
    "            import vllm.model_executor.models.qwen4_exp as _q4x\n"
    "            _q4x.datagen_stash[\"input_ids\"] = (\n"
    "                self.input_ids.gpu[:num_scheduled_tokens]\n"
    "                if num_scheduled_tokens > 0 else None)\n"
    "        except Exception:\n"
    "            pass\n"
    "        is_first_rank = get_pp_group().is_first_rank\n"
)

# ---- edit 1b: gpu_model_runner._model_forward position stash ---------------
FWD_ANCHOR = (
    "        return self.model(\n"
    "            input_ids=input_ids,\n"
    "            positions=positions,\n"
    "            intermediate_tensors=intermediate_tensors,\n"
    "            inputs_embeds=inputs_embeds,\n"
    "            **model_kwargs,\n"
    "        )\n"
)
FWD_INSERT = (
    "        # [B70_MTP_DATAGEN_HOOK] Expose this step's positions to the\n"
    "        # qwen4_exp wide-state tracer (ids are stashed in _preprocess).\n"
    "        try:\n"
    "            import vllm.model_executor.models.qwen4_exp as _q4x\n"
    "            _q4x.datagen_stash[\"positions\"] = positions\n"
    "        except Exception:\n"
    "            pass\n"
    "        return self.model(\n"
    "            input_ids=input_ids,\n"
    "            positions=positions,\n"
    "            intermediate_tensors=intermediate_tensors,\n"
    "            inputs_embeds=inputs_embeds,\n"
    "            **model_kwargs,\n"
    "        )\n"
)

# ---- edit 2a: qwen4_exp module-level stash ---------------------------------
QWEN_DOC_ANCHOR = '"""Inference-only'
QWEN_DOC_INSERT = f'"""[{MARKER}] Inference-only'
STASH_DEF = (
    "\n"
    "# [B70_MTP_DATAGEN_HOOK] Per-step token ids / positions, populated by\n"
    "# gpu_model_runner (_preprocess stashes ids, _model_forward stashes\n"
    "# positions) so the wide-state tracer in Qwen4ExpModel.forward can pair\n"
    "# them with the wide residual even when the model receives\n"
    "# input_ids=None (VL/multimodal inputs_embeds path).\n"
    "datagen_stash: dict = {}\n"
)

# ---- edit 2b: qwen4_exp forward tracer -------------------------------------
FILL_ANCHOR = (
    "        if getattr(self, \"_mtp_hidden_buffer\", None) is not None:\n"
    "            _nt = residual.shape[0]\n"
    "            self._mtp_hidden_buffer[:_nt].copy_(residual.flatten(1))\n"
)
TRACER = (
    "        # [B70_MTP_DATAGEN_HOOK] Offline MTP re-fit tracer. Appends\n"
    "        # (token_ids, positions, wide_state) for this step to\n"
    "        # B70_MTP_DATAGEN_TRACE (rank 0 only). Token ids come from the\n"
    "        # runner stash (the model may receive input_ids=None on the\n"
    "        # VL/multimodal path); sliced to _nt to drop any SP padding.\n"
    "        # Uses a private copy of the wide residual so it never touches\n"
    "        # the MTP _mtp_hidden_buffer lifecycle (avoids the undersized-\n"
    "        # buffer crash when a later prefill is longer than an earlier one).\n"
    "        _tr = os.environ.get(\"B70_MTP_DATAGEN_TRACE\", \"\")\n"
    "        if _tr and os.environ.get(\"B70_MTP_DATAGEN\", \"1\") == \"1\":\n"
    "            try:\n"
    "                from vllm.distributed import get_tensor_model_parallel_rank as _tpk\n"
    "                _is_r0 = _tpk() == 0\n"
    "            except Exception:\n"
    "                _is_r0 = True\n"
    "            if _is_r0:\n"
    "                _ids = datagen_stash.get(\"input_ids\")\n"
    "                _pos = datagen_stash.get(\"positions\")\n"
    "                if _ids is not None:\n"
    "                    _nt = residual.shape[0]\n"
    "                    _wide = residual.flatten(1).to(torch.bfloat16).detach().contiguous()\n"
    "                    _wdim = _wide.shape[1]\n"
    "                    _tok = _ids.detach().cpu().view(-1)[:_nt].to(torch.int32)\n"
    "                    _posv = _pos.detach().cpu().view(-1)[:_nt].to(torch.int64) if _pos is not None else torch.arange(_nt, dtype=torch.int64)\n"
    "                    _wf = _wide.view(torch.uint16).cpu().numpy().tobytes()\n"
    "                    import struct as _st\n"
    "                    _rec = _st.pack(\"<II\", _nt, _wdim) + _tok.numpy().tobytes() + _posv.numpy().tobytes() + _wf\n"
    "                    _fh = getattr(self, \"_datagen_fh\", None)\n"
    "                    if _fh is None:\n"
    "                        _fh = open(_tr, \"ab\")\n"
    "                        self._datagen_fh = _fh\n"
    "                    _fh.write(_rec)\n"
    "                    _fh.flush()\n"
)


def patch_file(path, edits):
    """edits: list of (anchor, insert, mode). mode: 'replace' or 'append_after'."""
    src = open(path).read()
    already = MARKER in src
    changed = False
    for anchor, insert, mode in edits:
        if already:
            continue  # idempotent: file already carries the hook
        if anchor not in src:
            print(f"ERROR: anchor not found in {path}:\n{anchor[:120]}...", file=sys.stderr)
            return False
        if mode == "replace":
            src = src.replace(anchor, insert, 1)
        else:  # append_after
            src = src.replace(anchor, anchor + insert, 1)
        changed = True
    if MARKER not in src:
        print(f"ERROR: marker missing after patching {path}", file=sys.stderr)
        return False
    if changed:
        open(path, "w").write(src)
    py_compile.compile(path, doraise=True)
    print(f"patched: {path} (compile OK)")
    return True


def main():
    ok = True
    for root in ROOTS:
        runner = os.path.join(root, "v1", "worker", "gpu_model_runner.py")
        qwen = os.path.join(root, "model_executor", "models", "qwen4_exp.py")
        if not os.path.exists(runner) or not os.path.exists(qwen):
            print(f"skip: {root} (missing files)", file=sys.stderr)
            continue
        ok &= patch_file(runner, [
            (PRE_ANCHOR, PRE_INSERT, "replace"),
            (FWD_ANCHOR, FWD_INSERT, "replace"),
        ])
        ok &= patch_file(qwen, [
            (QWEN_DOC_ANCHOR, QWEN_DOC_INSERT, "replace"),
            ("logger = init_logger(__name__)",
             "logger = init_logger(__name__)" + STASH_DEF, "replace"),
            (FILL_ANCHOR, TRACER, "append_after"),
        ])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
