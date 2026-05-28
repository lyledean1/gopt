#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")" && pwd)}"
HF_DATA_URI="${HF_DATA_URI:-hf://buckets/lyledean/gopt/data/go_thin}"
HF_CORPUS_URI="${HF_CORPUS_URI:-hf://buckets/lyledean/gopt/corpora/go_thin}"
UPLOAD_CORPUS="${UPLOAD_CORPUS:-0}"

cd "$REPO_DIR"

if ! command -v hf >/dev/null 2>&1; then
  echo "missing 'hf' CLI; install it first"
  exit 1
fi

if [ ! -d data/go_thin ]; then
  echo "missing data/go_thin"
  exit 1
fi

echo "Uploading data/go_thin to $HF_DATA_URI"
hf sync ./data/go_thin "$HF_DATA_URI"

if [ "$UPLOAD_CORPUS" = "1" ]; then
  if [ ! -d corpora/go_thin ]; then
    echo "missing corpora/go_thin"
    exit 1
  fi
  echo "Uploading corpora/go_thin to $HF_CORPUS_URI"
  hf sync ./corpora/go_thin "$HF_CORPUS_URI"
fi

echo
echo "Upload complete."
