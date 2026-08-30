#!/usr/bin/env bash
# =============================================================================
# start-qwen-256k.sh - one front door for Qwen3.8-Flash-Next
#
# Flash-Next can be served two ways, each with its own tuned launcher:
#   vllm   start-qwen-256k-vllm.sh   :8000  alias qwen-256k  (W4A16, 4x GPU)
#   llama  start-flashnext-llama.sh :8090  alias flash-next (Q4_K_XL, 4x GPU)
#
# This script does NOT duplicate that launch logic - it delegates to the two
# launchers and adds a unified status / smoke / logs / restart surface.
#
# Usage:
#   bash start-qwen-256k.sh start [vllm|llama]     # default: vllm
#   bash start-qwen-256k.sh stop  [vllm|llama|all] # default: all
#   bash start-qwen-256k.sh restart [vllm|llama]   # default: vllm
#   bash start-qwen-256k.sh status                 # both engines
#   bash start-qwen-256k.sh smoke [vllm|llama]     # default: vllm (completion test)
#   bash start-qwen-256k.sh logs  [vllm|llama] [N] # tail N lines (default 30)
#   bash start-qwen-256k.sh help
#
# Env overrides pass straight through to the underlying launcher
# (QWEN256K_* for vllm, FLASHNEXT_* for llama).
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SH="$HERE/start-qwen-256k-vllm.sh"
LLAMA_SH="$HERE/start-flashnext-llama.sh"

# engine -> port / alias / log
port_of()  { case "$1" in vllm) echo "${QWEN256K_PORT:-8000}";; llama) echo "${FLASHNEXT_PORT:-8090}";; esac; }
alias_of() { case "$1" in vllm) echo "${QWEN256K_ALIAS:-qwen-256k}";; llama) echo "${FLASHNEXT_ALIAS:-flash-next}";; esac; }
log_of()   { case "$1" in vllm) echo "$HERE/logs/vllm_$(port_of vllm).log";; llama) echo "$HERE/logs/llama_$(port_of llama).log";; esac; }
sh_of()    { case "$1" in vllm) echo "$VLLM_SH";; llama) echo "$LLAMA_SH";; esac; }

is_up() { # engine -> 0 if its port answers /v1/models
  curl -s --max-time 2 "http://127.0.0.1:$(port_of "$1")/v1/models" >/dev/null 2>&1
}

status_one() {
  local e="$1"
  if is_up "$e"; then
    echo "  [UP]   $e  :$(port_of "$e")  alias $(alias_of "$e")"
  else
    echo "  [DOWN] $e  :$(port_of "$e")  alias $(alias_of "$e")"
  fi
}

smoke() { # engine
  local e="$1" port; port="$(port_of "$e")"
  if ! is_up "$e"; then echo "  $e not up on :$port - start it first"; return 1; fi
  echo "  $e :$port  prompt='The capital of France is'"
  curl -s --max-time 120 "http://127.0.0.1:$port/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$(alias_of "$e")\",\"prompt\":\"The capital of France is\",\"max_tokens\":5,\"temperature\":0,\"logprobs\":1}" \
  | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
    c = d["choices"][0]
    print("    text:", repr(c["text"]))
    lp = c.get("logprobs") or {}
    for t, l in zip(lp.get("tokens", []), lp.get("token_logprobs", [])):
        print(f"    {t!r:12} {l:.3f}")
except Exception as ex:
    print("    (could not parse response:", ex, ")")
'
}

usage() { sed -n "2,20p" "${BASH_SOURCE[0]}"; }

cmd="${1:-help}"
arg="${2:-}"
case "$cmd" in
  start)
    e="${arg:-vllm}"
    [[ -f "$(sh_of "$e")" ]] || { echo "ERROR: launcher missing: $(sh_of "$e")"; exit 1; }
    bash "$(sh_of "$e")" start
    ;;
  stop)
    case "$arg" in
      all) bash "$VLLM_SH" stop; bash "$LLAMA_SH" stop ;;
      vllm|llama) bash "$(sh_of "$arg")" stop ;;
      *) echo "usage: $0 stop [vllm|llama|all]"; exit 1 ;;
    esac
    ;;
  restart)
    e="${arg:-vllm}"
    bash "$(sh_of "$e")" stop
    bash "$(sh_of "$e")" start
    ;;
  status)
    echo "Qwen3.8-Flash-Next engines:"
    status_one vllm
    status_one llama
    ;;
  smoke)
    e="${arg:-vllm}"
    smoke "$e"
    ;;
  logs)
    e="${arg:-vllm}"; n="${3:-30}"
    l="$(log_of "$e")"
    [[ -f "$l" ]] || { echo "  no log at $l"; exit 1; }
    echo "  --- $l (last $n) ---"
    tail -n "$n" "$l"
    ;;
  help|-h|--help) usage ;;
  *) echo "unknown command: $cmd"; usage; exit 1 ;;
esac