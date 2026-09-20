#!/usr/bin/env bash
# Dual-B70 TP2 launcher for the JonathanColetti Qwen3.8-27B-Uncensored + DFlash2
# reproduction study (LocalMaxxing run cmtorjvuo00zbn701i7wo61iv, 141.3 tok/s).
#
# Runs the pinned vLLM-XPU image on GPU0+GPU1 (renderD130+renderD131, mask 0,1)
# with tensor-parallel-size 2, host-staged oneCCL collectives (no P2P on discrete
# Arcs), and per-worker ZE_AFFINITY_MASK via the cookbook worker-affinity patch.
#
# Usage:
#   bash run-dual-b70-tp2.sh start    # launch container
#   bash run-dual-b70-tp2.sh logs     # tail container logs
#   bash run-dual-b70-tp2.sh health   # curl health + models
#   bash run-dual-b70-tp2.sh stop     # stop + remove container
#
# Env overrides (defaults in parens):
#   DB_MODEL_DIR      target model dir (/mnt/data/models/JonathanColetti-Qwen3.8-27B-Uncensored-BF16)
#   DB_DRAFT_DIR      DFlash2 draft dir (/mnt/data/models/incoai-Qwen3.8-27B-DFlash2); empty = no draft
#   DB_QUANT          fp8 | bf16 | gptq  (fp8)
#   DB_SPEC_METHOD    mtp | dflash | none (mtp)
#   DB_SPEC_TOKENS    num_speculative_tokens (8 for mtp, 7 for dflash)
#   DB_TP_SIZE        tensor parallel size (2)
#   DB_HOST_PORT      host port (8002)
#   DB_SERVED_NAME    served model name (qwen38-unc)
#   DB_CONTAINER      container name (qw38unc_tp2)
#   DB_MAX_MODEL_LEN  (131072)
#   DB_MAX_NUM_SEQS   (8)
#   DB_GPU_MEM_UTIL   (0.95)
#   DB_GEN_CONFIG     optional JSON -> --override-generation-config
#   DB_API_KEY        optional -> --api-key
#   DB_RENDER_NODES   space-separated render nodes ("renderD130 renderD131")
#   DB_AFFINITY_MASK  ("0,1")
#   DB_AFFINITY_BASE  physical GPU index of the first render node (0 for renderD130,
#                     2 for renderD128); worker rank r binds to GPU base+r
#   DB_HOST_STAGED_AR 1 = route small TP allreduces through pinned-host gloo
#                     (rebuilds the LocalMaxxing host-staged collectives delta)
#   DB_HS_MAX_ELEMS   max tensor elements routed through host staging (4000000)
#   DB_VISION         1 = enable the VL tower (--limit-mm-per-prompt {"image":4}); 0 = text-only
#   DB_FIX_ACCEPT_SYNC / DB_FIX_BACKWARD_COPY / DB_FIX_EAGLE_DROP
#                     1 = apply the GDN/mamba spec-decode corruption fixes (default);
#                     0 = leave vLLM stock (A/B only). Fixes guard the target model's
#                     GDN recurrent-state rollback under spec + prefix caching.
#   DB_IMAGE          docker image (default: pinned vllm-openai-xpu 0.27.2rc1 sha256:f01e24f6...;
#                     use vllm/vllm-openai-xpu:v0.28.0 for native DFlash2 support)
set -euo pipefail

IMAGE="${DB_IMAGE:-vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f}"
COOKBOOK="$HOME/intel-arc-pro-b70-inference-cookbook"

MODEL_DIR="${DB_MODEL_DIR:-/mnt/data/models/JonathanColetti-Qwen3.8-27B-Uncensored-BF16}"
DRAFT_DIR="${DB_DRAFT_DIR:-/mnt/data/models/incoai-Qwen3.8-27B-DFlash2}"
QUANT="${DB_QUANT:-fp8}"
SPEC_METHOD="${DB_SPEC_METHOD:-mtp}"
SPEC_TOKENS="${DB_SPEC_TOKENS:-8}"
TP_SIZE="${DB_TP_SIZE:-2}"
HOST_PORT="${DB_HOST_PORT:-8002}"
SERVED_NAME="${DB_SERVED_NAME:-qwen38-unc}"
CONTAINER="${DB_CONTAINER:-qw38unc_tp2}"
MAX_MODEL_LEN="${DB_MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${DB_MAX_NUM_SEQS:-8}"
GPU_MEM_UTIL="${DB_GPU_MEM_UTIL:-0.95}"
GEN_CONFIG="${DB_GEN_CONFIG:-}"
API_KEY="${DB_API_KEY:-}"
RENDER_NODES="${DB_RENDER_NODES:-renderD130 renderD131}"
AFFINITY_MASK="${DB_AFFINITY_MASK:-0,1}"
AFFINITY_BASE="${DB_AFFINITY_BASE:-0}"
HOST_STAGED_AR="${DB_HOST_STAGED_AR:-0}"
HS_MAX_ELEMS="${DB_HS_MAX_ELEMS:-4000000}"
VISION="${DB_VISION:-0}"
FIX_ACCEPT_SYNC="${DB_FIX_ACCEPT_SYNC:-1}"
FIX_BACKWARD_COPY="${DB_FIX_BACKWARD_COPY:-1}"
FIX_EAGLE_DROP="${DB_FIX_EAGLE_DROP:-1}"
# All Arc render nodes share one gid on this host; a single --group-add covers them.
RENDER_GID="$(stat -c '%g' /dev/dri/renderD130)"

cmd="${1:-start}"

case "$cmd" in
start)
  echo "model dir:   $MODEL_DIR"
  echo "draft dir:   ${DRAFT_DIR:-<none>}"
  echo "quant:       $QUANT   spec: $SPEC_METHOD x$SPEC_TOKENS   tp: $TP_SIZE"
  echo "image:       $IMAGE"
  echo "render:      $RENDER_NODES (mask $AFFINITY_MASK)"
  echo "host port:   $HOST_PORT  served: $SERVED_NAME  container: $CONTAINER"
  [ -d "$MODEL_DIR" ] || { echo "MISSING model dir: $MODEL_DIR"; exit 1; }
  if [ "$SPEC_METHOD" = "dflash" ]; then
    [ -d "$DRAFT_DIR" ] || { echo "MISSING draft dir: $DRAFT_DIR"; exit 1; }
  fi

  # Verify requested render nodes exist (whole /dev/dri is mounted; ZE_AFFINITY_MASK
  # + the per-worker patch select which GPUs each TP rank actually uses).
  for rn in $RENDER_NODES; do
    [ -e "/dev/dri/$rn" ] || { echo "MISSING render node: /dev/dri/$rn"; exit 1; }
  done

  docker rm -f "$CONTAINER" 2>/dev/null || true

  # Draft mount (only when a draft is used).
  DRAFT_MOUNT=""
  if [ "$SPEC_METHOD" = "dflash" ]; then
    DRAFT_MOUNT="-v $DRAFT_DIR:/draft:ro"
  fi

  docker run -d \
    --name "$CONTAINER" \
    -p ${HOST_PORT}:8000 \
    --device /dev/dri:/dev/dri \
    -v /dev/dri:/dev/dri:ro \
    --group-add "$RENDER_GID" \
    --cap-add SYS_PTRACE \
    --ipc=host \
    -v "$MODEL_DIR:/model:ro" \
    $DRAFT_MOUNT \
    -v "$COOKBOOK/patches/patch_vllm_worker_affinity.py:/patch_affinity.py:ro" \
    -v "$COOKBOOK/patches/patch_fp8_w8a16.py:/patch_fp8w8a16.py:ro" \
    -v "$COOKBOOK/patches/patch_mtp_nightly.py:/patch_mtp.py:ro" \
    -v "$COOKBOOK/patches/patch_mtp_boundary.py:/patch_boundary.py:ro" \
    -v "$COOKBOOK/patches/patch_gdn_mixed_split_v5.py:/patch_v5.py:ro" \
    -v "$HOME/electric-sheep/serve/patch_dflash2_alias.py:/patch_dflash2.py:ro" \
    -v "$HOME/electric-sheep/serve/patch_host_staged_ar.py:/patch_hstar.py:ro" \
    -v "$COOKBOOK/patches/patch_fix_accepted_sync.py:/patch_acc.py:ro" \
    -v "$COOKBOOK/patches/patch_fix_backward_copy.py:/patch_bwd.py:ro" \
    -v "$COOKBOOK/patches/patch_fix_eagle_drop.py:/patch_eag.py:ro" \
    -e VLLM_TARGET_DEVICE=xpu \
    -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
    -e ZE_AFFINITY_MASK="$AFFINITY_MASK" \
    -e B70_WORKER_AFFINITY=1 \
    -e B70_AFFINITY_BASE="$AFFINITY_BASE" \
    -e B70_HOST_STAGED_AR="$HOST_STAGED_AR" \
    -e B70_HS_MAX_ELEMS="$HS_MAX_ELEMS" \
    -e VISION="$VISION" \
    -e B70_FIX_ACCEPT_SYNC="$FIX_ACCEPT_SYNC" \
    -e B70_FIX_BACKWARD_COPY="$FIX_BACKWARD_COPY" \
    -e B70_FIX_EAGLE_DROP="$FIX_EAGLE_DROP" \
    -e CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296 \
    -e CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296 \
    -e CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296 \
    -e CCL_SYCL_ALLTOALL_TMP_BUF=1 \
    -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    -e SERVED_NAME="$SERVED_NAME" \
    -e QUANT="$QUANT" \
    -e SPEC_METHOD="$SPEC_METHOD" \
    -e SPEC_TOKENS="$SPEC_TOKENS" \
    -e TP_SIZE="$TP_SIZE" \
    -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    -e MAX_NUM_SEQS="$MAX_NUM_SEQS" \
    -e GPU_MEM_UTIL="$GPU_MEM_UTIL" \
    -e GEN_CONFIG="$GEN_CONFIG" \
    -e API_KEY="$API_KEY" \
    --entrypoint bash \
    "$IMAGE" \
    -lc 'set -e
# --- patch stage (runs on ephemeral container FS every start) ---
python /patch_affinity.py
if [ "$QUANT" = "fp8" ]; then python /patch_fp8w8a16.py; fi
# MTP drafter patches only needed for the mtp method.
if [ "$SPEC_METHOD" = "mtp" ]; then python /patch_mtp.py; python /patch_boundary.py; fi
# DFlash2 draft arch is not in the pinned image registry; alias it so the
# drafter at least loads far enough to surface any missing-component errors.
if [ "$SPEC_METHOD" = "dflash" ]; then python /patch_dflash2.py; fi
python /patch_hstar.py
python /patch_v5.py
# GDN/mamba spec-decode corruption fixes (silent-corruption class, vllm#53919/
# #53505/#48375): guard the target model recurrent-state rollback under spec +
# prefix caching. Required for coherent multi-turn chats on hybrid GDN models.
python /patch_acc.py
python /patch_bwd.py
python /patch_eag.py
# --- build serve args ---
QUANT_ARG=""
case "$QUANT" in
  fp8)   QUANT_ARG="--quantization fp8 --dtype bfloat16";;
  bf16)  QUANT_ARG="--dtype bfloat16";;
  gptq)  QUANT_ARG="--quantization gptq --dtype float16";;
  *)     QUANT_ARG="--dtype bfloat16";;
esac
SPEC_ARG=""
if [ "$SPEC_METHOD" = "mtp" ]; then
  SPEC_ARG="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_TOKENS}"
elif [ "$SPEC_METHOD" = "dflash" ]; then
  SPEC_ARG="--speculative-config {\"method\":\"dflash\",\"model\":\"/draft\",\"num_speculative_tokens\":$SPEC_TOKENS}"
fi
if [ -n "$GEN_CONFIG" ]; then GENCFG_ARG="--override-generation-config $GEN_CONFIG"; else GENCFG_ARG=""; fi
if [ -n "$API_KEY" ]; then APIKEY_ARG="--api-key $API_KEY"; else APIKEY_ARG=""; fi
# v0.28.0 wants JSON here (the pinned image accepted image=4; do not mix forms).
if [ "$VISION" = "1" ]; then MM_ARG="--limit-mm-per-prompt {\"image\":4}"; else MM_ARG="--language-model-only"; fi
exec vllm serve /model \
  --tensor-parallel-size $TP_SIZE \
  $QUANT_ARG \
  --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization $GPU_MEM_UTIL \
  --kv-cache-dtype fp8 \
  --port 8000 \
  --max-num-seqs $MAX_NUM_SEQS \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --served-model-name $SERVED_NAME \
  $MM_ARG \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  $GENCFG_ARG \
  $APIKEY_ARG \
  $SPEC_ARG'
  echo "launched $CONTAINER on :$HOST_PORT (served as $SERVED_NAME, tp=$TP_SIZE, quant=$QUANT, spec=$SPEC_METHOD)"
  ;;
logs)
  docker logs --tail "${2:-40}" "$CONTAINER" 2>&1
  ;;
health)
  curl -fsS "http://127.0.0.1:$HOST_PORT/health" && echo " HEALTH OK"
  curl -fsS "http://127.0.0.1:$HOST_PORT/v1/models" | python3 -c "import json,sys;d=json.load(sys.stdin);print('models:',[m['id'] for m in d.get('data',[])])"
  ;;
stop)
  docker rm -f "$CONTAINER" 2>/dev/null || true
  echo "stopped"
  ;;
*)
  echo "usage: $0 {start|logs|health|stop}"
  ;;
esac
