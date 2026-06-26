#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${CONFIG:-sekv/configs/default.yaml}"
MODEL_NAME="${MODEL:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
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
      echo "Usage: $0 [--config <path>] [--model <model_key>]"
      echo "Env vars: CONFIG=<path> MODEL=<model_key>"
      exit 1
      ;;
  esac
done

if [[ -n "$MODEL_NAME" ]]; then
  sekv-train --config "$CONFIG_PATH" --model "$MODEL_NAME"
else
  sekv-train --config "$CONFIG_PATH"
fi
