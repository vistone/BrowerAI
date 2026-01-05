#!/usr/bin/env bash
# Batch collect site data with BrowerAI --learn
# Usage:
#   BATCH_SIZE=10 SLEEP_BETWEEN=2 START=1 STOP=100 ./collect_sites.sh
# Env vars:
#   BATCH_SIZE      Run this many sites then rest (default 10)
#   SLEEP_BETWEEN   Seconds to sleep between sites (default 2)
#   REST_SECONDS    Seconds to rest after each batch (default 5)
#   START           1-based start line from website_list.txt (default 1)
#   STOP            1-based stop line inclusive (default 0 = no stop)
#   LIST            Override website list path
#   BROWSER_BIN     Override browerai binary path
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LIST_PATH=${LIST:-"${ROOT_DIR}/training/data/website_list.txt"}
BROWSER_BIN=${BROWSER_BIN:-"${ROOT_DIR}/target/release/browerai"}
OUTPUT_DIR="${ROOT_DIR}/training/data"

BATCH_SIZE=${BATCH_SIZE:-10}
SLEEP_BETWEEN=${SLEEP_BETWEEN:-2}
REST_SECONDS=${REST_SECONDS:-5}
START=${START:-1}
STOP=${STOP:-0}

if [[ ! -x "${BROWSER_BIN}" ]]; then
  echo "[ERR] browerai binary not found or not executable at: ${BROWSER_BIN}" >&2
  echo "      Build first: cargo build --release --features ai" >&2
  exit 1
fi

if [[ ! -f "${LIST_PATH}" ]]; then
  echo "[ERR] website list not found: ${LIST_PATH}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

line_no=0
processed=0
batch_count=0

while IFS= read -r url || [[ -n "$url" ]]; do
  # skip blank or commented lines
  if [[ -z "${url}" || "${url}" =~ ^# ]]; then
    continue
  fi

  line_no=$((line_no + 1))
  if (( line_no < START )); then
    continue
  fi
  if (( STOP > 0 && line_no > STOP )); then
    break
  fi

  echo "[INFO] (#${line_no}) visiting ${url}"
  set +e
  "${BROWSER_BIN}" --learn "${url}"
  status=$?
  set -e
  if (( status != 0 )); then
    echo "[WARN] visit failed (code ${status}) for ${url}" >&2
  fi

  processed=$((processed + 1))
  batch_count=$((batch_count + 1))

  sleep "${SLEEP_BETWEEN}"

  if (( batch_count >= BATCH_SIZE )); then
    echo "[INFO] batch of ${batch_count} done, resting ${REST_SECONDS}s"
    sleep "${REST_SECONDS}"
    batch_count=0
  fi

done < "${LIST_PATH}"

echo "[DONE] processed ${processed} site(s) (lines ${START}-${line_no})"
