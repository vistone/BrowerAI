#!/usr/bin/env bash
set -euo pipefail

# Qwen2.5-Coder-7B-Instruct GGUF downloader
# Optional: huggingface-cli (pip install -U huggingface_hub)
# Optional: llama-gguf-split (from llama.cpp build) if files are split

MODEL_REPO="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
# Choose one: q5_k_m (quality/size balance) or q4_k_m (lighter)
QUANT="q5_k_m"
TARGET_DIR="models/local/qwen2_5_coder_7b_gguf"
BASE_URL="https://huggingface.co/${MODEL_REPO}/resolve/main"
GGUF_FILE="qwen2.5-coder-7b-instruct-${QUANT}.gguf"
TOKENIZER_FILE="tokenizer.json"
BASE_MODEL_REPO="Qwen/Qwen2.5-Coder-7B-Instruct"
BASE_MODEL_URL="https://huggingface.co/${BASE_MODEL_REPO}/resolve/main"

mkdir -p "${TARGET_DIR}"

download_with_curl() {
  local url="$1"
  local out_file="$2"
  echo "Downloading ${out_file} via curl ..."
  curl -L --fail --progress-bar -C - "$url" -o "$out_file"
}

if command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli detected; using it for download."
  huggingface-cli download "${MODEL_REPO}" \
    --include "qwen2.5-coder-7b-instruct-${QUANT}-*.gguf" "${TOKENIZER_FILE}" \
    --local-dir "${TARGET_DIR}" \
    --local-dir-use-symlinks False
else
  echo "huggingface-cli not found; falling back to curl."
  download_with_curl "${BASE_URL}/${GGUF_FILE}?download=1" "${TARGET_DIR}/${GGUF_FILE}"
  # Pull tokenizer directly from the base model repo to avoid 404s in GGUF repo
  download_with_curl "${BASE_MODEL_URL}/${TOKENIZER_FILE}?download=1" "${TARGET_DIR}/${TOKENIZER_FILE}"
fi

# Some GGUF repos omit tokenizer.json; pull from the base HF repo if needed.
if [[ ! -f "${TARGET_DIR}/${TOKENIZER_FILE}" ]]; then
  echo "tokenizer.json not found; retrying download from base model repo."
  download_with_curl "${BASE_MODEL_URL}/${TOKENIZER_FILE}?download=1" "${TARGET_DIR}/${TOKENIZER_FILE}"
fi

# Merge shards if present (requires llama-gguf-split in PATH)
FIRST_SHARD="${TARGET_DIR}/qwen2.5-coder-7b-instruct-${QUANT}-00001-of-00002.gguf"
MERGED_OUT="${TARGET_DIR}/qwen2.5-coder-7b-instruct-${QUANT}.gguf"
if [[ -f "$FIRST_SHARD" && ! -f "$MERGED_OUT" ]]; then
  if command -v llama-gguf-split >/dev/null 2>&1; then
    echo "Merging shards into ${MERGED_OUT} ..."
    llama-gguf-split --merge "$FIRST_SHARD" "$MERGED_OUT"
  else
    echo "Shards detected but llama-gguf-split not found. Install llama.cpp tools to merge, or use a single-file quant." >&2
  fi
fi

echo "Done. Files in ${TARGET_DIR}"