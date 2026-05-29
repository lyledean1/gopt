#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/gopt}"
REPO_URL="${REPO_URL:-https://github.com/lyledean1/gopt}"
GO_VERSION="${GO_VERSION:-1.25.0}"
HF_DATA_URI="${HF_DATA_URI:-hf://buckets/lyledean/gopt/data/go_thin}"
HF_CHECKPOINT_URI="${HF_CHECKPOINT_URI:-hf://buckets/lyledean/gopt/checkpoints/go_thin}"
DOWNLOAD_GO_THIN="${DOWNLOAD_GO_THIN:-1}"
DOWNLOAD_CHECKPOINTS="${DOWNLOAD_CHECKPOINTS:-1}"

if [ ! -d "$REPO_DIR" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if ! command -v go >/dev/null 2>&1; then
  curl -LO "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz"
  rm -rf /usr/local/go
  tar -C /usr/local -xzf "go${GO_VERSION}.linux-amd64.tar.gz"
fi

export PATH="$HOME/.local/bin:/usr/local/go/bin:$PATH"

uv sync --dev

if ! command -v hf >/dev/null 2>&1; then
  uv tool install "huggingface_hub[cli]"
fi

mkdir -p corpora/go corpora/go_thin data/go data/go_thin checkpoints samples

if [ "$DOWNLOAD_GO_THIN" = "1" ] && [ -n "$HF_DATA_URI" ]; then
  echo "Syncing go_thin artifacts from $HF_DATA_URI"
  hf sync "$HF_DATA_URI" ./data/go_thin
fi

if [ "$DOWNLOAD_CHECKPOINTS" = "1" ] && [ -n "$HF_CHECKPOINT_URI" ]; then
  echo "Syncing checkpoints from $HF_CHECKPOINT_URI"
  hf sync "$HF_CHECKPOINT_URI" ./checkpoints
fi

echo
echo "Environment ready in $REPO_DIR"
echo "PATH=$PATH"
echo "uv: $(uv --version)"
echo "go: $(go version)"
