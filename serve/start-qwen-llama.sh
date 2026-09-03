#!/usr/bin/env bash
# =============================================================================
# start-qwen-llama.sh — Interactive llama.cpp launcher for ANY Qwen GGUF model
#
# The llama.cpp counterpart of serve/fallback/start-qwen.sh (which serves
# safetensors via vLLM). Same interactive flow, but it discovers GGUF models
# and launches the SYCL llama-server build instead of vLLM.
#
# Interactively asks for:
#   1. Which Qwen GGUF model to serve (from the model library)
#   2. How many GPUs to use, and specifically which ones
#   3. Optimization options (context length, KV-cache dtype, reasoning,
#      MTP speculative decoding, port, served name)
#
# Model discovery:
#   A "model" is a top-level Qwen dir holding at least one MAIN GGUF
#   (projectors *mmproj* and MTP/draft tables *mtp*/*fastmtp* are excluded).
#   Multi-shard sets are represented by their first shard; llama-server loads
#   the remaining shards automatically. An optional vision projector
#   (*mmproj*.gguf) and MTP draft (*mtp*.gguf) are picked up when present.
#
# Served name defaults to  qwen-<context window size in k>  (e.g. 262144 ->
# "qwen-256k"), matching the vLLM launcher. Type "qwen" for the version-
# agnostic WebUI slot.
#
# XPU environment comes from electric-sheep/llama/set-env.sh (oneAPI + SYCL
# tuning + ONEAPI_DEVICE_SELECTOR for the chosen physical GPUs).
#
# Tuned defaults (benchmarked on this box, 2026-09-02):
#   - GPUs:  ONE card whenever the weights fit (~<=30 GiB). Discrete Arc B70s
#     have no P2P/USM, so cross-device sync stages through host RAM: 35B-A3B
#     Q4_K_P decodes ~94 t/s on 1 GPU vs ~32 t/s on a 4-GPU layer split, and
#     --split-mode row crashes (no USM). Split only past one 32 GiB card.
#   - KV:    q4_0 (q8_0/iq4_nl can segfault in sustained decode here).
#   - Fixed: -ngl 99 -fa on -np 1 -b 2048 -ub 512 (env-overridable).
#   - MTP:   only offered when the model dir ships a *mtp*.gguf draft.
#
# Usage:
#   bash start-qwen-llama.sh            # fully interactive (foreground)
#   bash start-qwen-llama.sh --dry-run  # print the command, don't launch
#
# Env overrides (skip the matching prompt by pre-setting):
#   QWEN_LLAMA_BATCH (2048)  QWEN_LLAMA_UBATCH (512)  QWEN_LLAMA_NP (1)
#   QWEN_LLAMA_TENSOR_SPLIT  QWEN_LLAMA_REASONING_FORMAT
# =============================================================================
set -uo pipefail

B="$HOME/electric-sheep/llama/llama.cpp/build/bin"
SET_ENV="$HOME/electric-sheep/llama/set-env.sh"
MODELS_DIR="$HOME/electric-sheep/models"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# Perf defaults (env-overridable; not prompted to keep the flow short).
BATCH="${QWEN_LLAMA_BATCH:-2048}"
UBATCH="${QWEN_LLAMA_UBATCH:-512}"
NP="${QWEN_LLAMA_NP:-1}"          # single-user server: always 1 slot

ask() { local prompt="$1" def="${2:-}" ans; if [[ -n "$def" ]]; then read -rp "$prompt [$def]: " ans; else read -rp "$prompt: " ans; fi; ans="${ans:-$def}"; printf '%s' "$ans"; }

# --- GPU detection -----------------------------------------------------------
# Detect Level-Zero GPUs once (cached) — sycl-ls can misbehave under command
# substitution with piped stdin, so capture it a single time up front.
SYCL_OUT="$(sycl-ls 2>/dev/null)"
GPU_IDS=""
for i in 0 1 2 3 4 5 6 7; do
  if grep -q "level_zero:$i" <<<"$SYCL_OUT"; then GPU_IDS+="$i "; fi
done
GPU_IDS="${GPU_IDS:-0 1 2 3}"   # fallback if detection failed

# --- helpers -----------------------------------------------------------------
# First main GGUF under a model dir (excludes projectors + MTP tables).
# Sorting puts a sharded set's 00001-of-NNNNN shard first.
pick_main_gguf() {
  find "$1" -maxdepth 2 -name '*.gguf' \
    ! -iname '*mmproj*' ! -iname '*mtp*' ! -iname '*fastmtp*' 2>/dev/null | sort | head -1
}

# --- 1. Qwen GGUF model selection --------------------------------------------
echo "=== Qwen GGUF model selection ==="
# Trailing slash: $MODELS_DIR is a symlink to /mnt/data/models; globbing a bare
# symlink won't expand, so append /.
mapfile -t QWEN < <(
  for d in "$MODELS_DIR/"*/; do
    [[ "$(basename "$d")" =~ [Qq]wen ]] || continue
    [[ -n "$(pick_main_gguf "$d")" ]] || continue
    printf '%s\n' "${d%/}"
  done | sort
)
if [[ ${#QWEN[@]} -eq 0 ]]; then echo "ERROR: no Qwen GGUF models found in $MODELS_DIR"; exit 1; fi
for i in "${!QWEN[@]}"; do
  # Weight-only size (excludes projectors/MTP tables) + how many cards it needs.
  wbytes=0
  while read -r f; do wbytes=$((wbytes + $(stat -c %s "$f" 2>/dev/null || echo 0))); done \
    < <(find "${QWEN[$i]}" -maxdepth 2 -name '*.gguf' ! -iname '*mmproj*' ! -iname '*mtp*' ! -iname '*fastmtp*' 2>/dev/null)
  wgib=$(( (wbytes / 1073741824) + 1 ))
  tag="fits 1 GPU"
  (( wgib > 32 )) && tag="needs $(( (wgib + 31) / 32 ))+ GPUs"
  printf "  %2d) %-55s ~%2dG weights  [%s]\n" "$((i+1))" "${QWEN[$i]#$MODELS_DIR/}" "$wgib" "$tag"
done
read -rp "Qwen model number: " qn
MODEL_PATH="${QWEN[$((qn-1))]:-}"
[[ -z "$MODEL_PATH" ]] && { echo "ERROR: bad selection."; exit 1; }
MODEL_FILE="$(pick_main_gguf "$MODEL_PATH")"
[[ -z "$MODEL_FILE" ]] && { echo "ERROR: no GGUF found under $MODEL_PATH"; exit 1; }
# Optional extras, picked up when present in the same dir.
MMPROJ="$(find "$MODEL_PATH" -iname '*mmproj*.gguf' 2>/dev/null | head -1)"
DRAFT="$(find "$MODEL_PATH" -iname '*mtp*.gguf' ! -iname '*fastmtp*' 2>/dev/null | head -1)"
echo "Model: $MODEL_FILE"
[[ -n "$MMPROJ" ]] && echo "Vision projector: $MMPROJ"
[[ -n "$DRAFT"  ]] && echo "MTP draft:        $DRAFT"

# --- 2. GPU selection ---------------------------------------------------------
# Total main-weight bytes (all shards, excluding projectors + MTP tables).
WBYTES=0
while read -r f; do WBYTES=$((WBYTES + $(stat -c %s "$f" 2>/dev/null || echo 0))); done \
  < <(find "$MODEL_PATH" -maxdepth 2 -name '*.gguf' ! -iname '*mmproj*' ! -iname '*mtp*' ! -iname '*fastmtp*' 2>/dev/null)
WGIB=$(( (WBYTES / 1073741824) + 1 ))   # round up to whole GiB
# Optimal GPU count: one card whenever the weights fit. Discrete Arc B70s have
# no P2P/USM shared memory, so a multi-GPU split stages every cross-device sync
# through host RAM over PCIe. Benchmarked (35B-A3B Q4_K_P): 1 GPU ~94 t/s vs
# 4-GPU layer split ~32 t/s; --split-mode row crashes (no USM). Only split when
# the weights exceed one 32 GiB card.
REC_GPU=1
(( WGIB > 32 )) && REC_GPU=$(( (WGIB + 31) / 32 ))
echo
echo "=== GPU selection ==="
echo "Detected GPUs: $GPU_IDS"
echo "Weights: ~${WGIB} GiB -> optimal: $REC_GPU GPU(s)"
read -rp "How many GPUs to use (1-4) [recommended: $REC_GPU]: " n
n="${n:-$REC_GPU}"
read -rp "Which GPUs (comma-separated, e.g. 0,1,2,3): " gsel
GPU_SET="${gsel:-$(seq -s, 0 $((n-1)))}"
N_GPU="$(echo "$GPU_SET" | awk -F, '{print NF}')"
echo "Using GPUs: $GPU_SET  ($N_GPU GPU(s))"
# Safety net if the user overrides the recommendation.
if [[ "$N_GPU" -gt 1 && "$WGIB" -le 32 ]]; then
  echo "NOTE: weights fit on a single 32 GiB card; on discrete Arc B70s a"
  echo "      multi-GPU split is SLOWER than one GPU (~3x for MoE decode)."
fi

# --- 3. Optimization options --------------------------------------------------
echo
echo "=== Optimization options ==="
cl="$(ask "Context length (1=32768, 2=65536, 3=131072, 4=196608, 5=262144, 0=custom)" "5")"
case "$cl" in
  1) CTX=32768 ;; 2) CTX=65536 ;; 3) CTX=131072 ;;
  4) CTX=196608 ;; 5) CTX=262144 ;;
  0) CTX="$(ask "Custom ctx-size" "262144")" ;;
  *) CTX=262144 ;;
esac
# KV cache dtype. OPTIMAL: q4_0. On this SYCL build q4_0/q5_0 are the safe
# choices; q8_0 and iq4_nl can segfault in sustained decode (and enable a
# Hadamard K-rotation that some attention paths reject, hence
# LLAMA_ATTN_ROT_DISABLE below).
kv="$(ask "KV cache dtype (1=q4_0 [recommended], 2=q5_0, 3=q8_0, 4=f16, 0=custom)" "1")"
case "$kv" in
  1) KV=q4_0 ;; 2) KV=q5_0 ;; 3) KV=q8_0 ;; 4) KV=f16 ;;
  0) KV="$(ask "Custom KV cache type (f32,f16,bf16,q8_0,q4_0,q4_1,iq4_nl,q5_0,q5_1)" "q4_0")" ;;
  *) KV=q4_0 ;;
esac
rz="$(ask "Reasoning/thinking (1=on, 2=off, 3=auto)" "1")"
case "$rz" in
  1) REASONING=on ;; 2) REASONING=off ;; 3) REASONING=auto ;;
  *) REASONING=on ;;
esac
# MTP only helps when the model dir ships a draft table.
if [[ -n "$DRAFT" ]]; then
  mtp="$(ask "Enable MTP speculative decoding? (y/n)" "n")"
else
  mtp="n"
fi
PORT="$(ask "Port" "8088")"
# Served name = qwen-<context window size in k> (type "qwen" for the
# version-agnostic WebUI slot).
ALIAS="$(ask "Served model name" "qwen-$((CTX/1024))k")"
# Optional explicit tensor split (fractions summing to ~1). Only meaningful
# with multiple GPUs; blank = let llama.cpp auto-split by free VRAM.
TSPLIT=""
if [[ "$N_GPU" -gt 1 ]]; then
  TSPLIT="$(ask "Tensor split (comma fractions, e.g. 1,1,1,1; blank=auto)" "")"
fi

# --- 4. Build command ----------------------------------------------------------
CMD=("$B/llama-server"
  -m "$MODEL_FILE"
  -ngl 99 -fa on
  -c "$CTX" -np "$NP" -b "$BATCH" -ub "$UBATCH"
  -ctk "$KV" -ctv "$KV"
  --jinja
  --reasoning "$REASONING"
  --alias "$ALIAS"
  --host 0.0.0.0 --port "$PORT"
)
# Vision tower (only when a projector ships with the model).
[[ -n "$MMPROJ" ]] && CMD+=(--mmproj "$MMPROJ")
# Optional reasoning format (e.g. deepseek) — only passed when set.
[[ -n "${QWEN_LLAMA_REASONING_FORMAT:-}" ]] && CMD+=(--reasoning-format "$QWEN_LLAMA_REASONING_FORMAT")
# MTP speculative decoding via a draft model file.
if [[ "${mtp,,}" == "y" ]]; then
  if [[ -n "$DRAFT" ]]; then
    CMD+=(--spec-draft-model "$DRAFT" --spec-draft-n-max 1)
  else
    echo "NOTE: MTP requested but no *mtp*.gguf draft found under $MODEL_PATH; skipping."
  fi
fi
# Explicit tensor split: also pin the (renumbered) device list so the split
# maps onto the filtered GPU space.
if [[ -n "$TSPLIT" ]]; then
  DEV_NAMES="$(seq -s, 0 $((N_GPU-1)) | sed 's/^/SYCL/')"
  CMD+=(--device "$DEV_NAMES" --tensor-split "$TSPLIT")
fi

echo
echo "=== Launch command ==="
printf '%q ' "${CMD[@]}"; echo
echo "GPUs=$GPU_SET  n=$N_GPU  ctx=$CTX  served=$ALIAS  kv=$KV  mtp=${mtp:-n}  np=$NP  batch=$BATCH  ubatch=$UBATCH  mmproj=${MMPROJ:+yes}  tsplit=${TSPLIT:-auto}"
echo

if [[ "$DRY_RUN" == "1" ]]; then
  echo "(dry-run: not launching)"
  exit 0
fi

# Load oneAPI + SYCL tuning + ONEAPI_DEVICE_SELECTOR for the chosen physical GPUs.
# shellcheck disable=SC1091
source "$SET_ENV" "$GPU_SET"
# Quantized KV (q8_0/iq4_nl) enables a Hadamard K-rotation input that some
# attention paths assert away; disabling the rotation satisfies it at negligible
# cost for these precisions.
case "$KV" in q8_0|iq4_nl) export LLAMA_ATTN_ROT_DISABLE=1 ;; esac

cd "$HOME/electric-sheep/serve"
exec "${CMD[@]}"
