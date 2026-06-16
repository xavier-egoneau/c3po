#!/usr/bin/env bash
set -euo pipefail

# Bench local reproductible (hors CI) pour distinguer régression réelle,
# changement de modèle et changement de prompt.
#
# Variables optionnelles :
#   C3PO_BIN      binaire c3po à utiliser (défaut: c3po)
#   C3PO_MODEL_7B requête/chemin du modèle 7B
#   C3PO_MODEL_14B requête/chemin du modèle 14B
#   C3PO_CTX      contexte de bench (défaut: 4096)

C3PO_BIN="${C3PO_BIN:-c3po}"
C3PO_CTX="${C3PO_CTX:-4096}"
C3PO_MODEL_7B="${C3PO_MODEL_7B:-Qwen2.5-7B-Instruct-Q4_K_M}"
C3PO_MODEL_14B="${C3PO_MODEL_14B:-qwen2.5-coder-14b-instruct-q4_k_m-00001}"

run_case() {
  local label="$1"
  local model="$2"
  shift 2

  echo
  echo "================================================================"
  echo "$label"
  echo "model: $model"
  echo "args : $*"
  echo "================================================================"
  "$C3PO_BIN" stats "$model" --ctx "$C3PO_CTX" "$@"
}

echo "c3po perf baseline"
echo "c3po     : $C3PO_BIN"
echo "context  : $C3PO_CTX"
echo "model 7B : $C3PO_MODEL_7B"
echo "model 14B: $C3PO_MODEL_14B"

run_case "7B / general+code / baseline" "$C3PO_MODEL_7B"
run_case "7B / general+code / speculative" "$C3PO_MODEL_7B" --speculative
run_case "14B / general+code / baseline" "$C3PO_MODEL_14B"
run_case "14B / general+code / speculative" "$C3PO_MODEL_14B" --speculative
