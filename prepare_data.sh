#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/gopt}"
VOCAB_SIZE="${VOCAB_SIZE:-4096}"

cd "$REPO_DIR"
export PATH="$HOME/.local/bin:/usr/local/go/bin:$PATH"

uv run gopt fetch-go-repos \
  --manifest manifests/go_repos.txt \
  --out-dir corpora/go

uv run gopt train-go-bpe \
  --root corpora/go \
  --out data/go/bpe_model.json \
  --vocab-size "$VOCAB_SIZE"

uv run gopt split-go-bpe \
  --root corpora/go \
  --bpe-model data/go/bpe_model.json \
  --train-out data/go/train_bpe_tokens.txt \
  --val-out data/go/val_bpe_tokens.txt

uv run gopt compile-dataset \
  --dataset data/go/train_bpe_tokens.txt \
  --val-dataset data/go/val_bpe_tokens.txt \
  --out data/go/compiled_dataset.pt

echo
echo "Prepared corpus, tokenizer, splits, and compiled dataset."
