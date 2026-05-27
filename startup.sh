#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/gopt}"
REPO_URL="${REPO_URL:-https://github.com/lyledean1/gopt}"
GO_VERSION="${GO_VERSION:-1.25.0}"

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

mkdir -p corpora/go data/go checkpoints samples

echo
echo "Environment ready in $REPO_DIR"
echo "PATH=$PATH"
echo "uv: $(uv --version)"
echo "go: $(go version)"
