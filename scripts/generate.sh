#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT=""
PROMPT=""
MAX_NEW_TOKENS="128"
BUDGET="none"
CONFIG_PATH="sekv/configs/default.yaml"
MODEL_NAME=""

if [[ $# -ge 2 && "${1:-}" != --* ]]; then
  # Backward-compatible positional mode.
  CHECKPOINT="$1"
  PROMPT="$2"
  MAX_NEW_TOKENS="${3:-128}"
  BUDGET="${4:-none}"
  CONFIG_PATH="${5:-sekv/configs/default.yaml}"
else
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --checkpoint)
        CHECKPOINT="$2"
        shift 2
        ;;
      --prompt)
        PROMPT="$2"
        shift 2
        ;;
      --max-new-tokens)
        MAX_NEW_TOKENS="$2"
        shift 2
        ;;
      --budget)
        BUDGET="$2"
        shift 2
        ;;
      --config)
        CONFIG_PATH="$2"
        shift 2
        ;;
      --model)
        MODEL_NAME="$2"
        shift 2
        ;;
      *)
        echo "Unknown argument: $1"
        echo "Usage: $0 --prompt <text> --checkpoint <path> [--budget <int|none>] [--max-new-tokens <int>] [--config <path>] [--model <model_key>]"
        echo "   or: $0 <checkpoint_path> <prompt> [max_new_tokens] [budget] [config_path]"
        exit 1
        ;;
    esac
  done
fi

if [[ -z "$CHECKPOINT" || -z "$PROMPT" ]]; then
  echo "Missing required args: --checkpoint and --prompt"
  echo "Usage: $0 --prompt <text> --checkpoint <path> [--budget <int|none>] [--max-new-tokens <int>] [--config <path>] [--model <model_key>]"
  exit 1
fi

CMD=(sekv-generate \
  --config "$CONFIG_PATH" \
  --checkpoint "$CHECKPOINT" \
  --prompt "$PROMPT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --budget "$BUDGET")

if [[ -n "$MODEL_NAME" ]]; then
  CMD+=(--model "$MODEL_NAME")
fi

"${CMD[@]}"
