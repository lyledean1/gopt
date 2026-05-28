#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/gopt}"
VOCAB_SIZE="${VOCAB_SIZE:-4096}"
CORPUS_ROOT="${CORPUS_ROOT:-corpora/go}"
DATA_DIR="${DATA_DIR:-data/go}"

cd "$REPO_DIR"
export PATH="$HOME/.local/bin:/usr/local/go/bin:$PATH"

uv run gopt fetch-go-repos \
  --manifest manifests/go_repos.txt \
  --out-dir "$CORPUS_ROOT"

uv run gopt train-go-bpe \
  --root "$CORPUS_ROOT" \
  --out "$DATA_DIR/bpe_model.json" \
  --vocab-size "$VOCAB_SIZE"

uv run gopt split-go-bpe \
  --root "$CORPUS_ROOT" \
  --bpe-model "$DATA_DIR/bpe_model.json" \
  --train-out "$DATA_DIR/train_bpe_tokens.txt" \
  --val-out "$DATA_DIR/val_bpe_tokens.txt"

uv run gopt compile-dataset \
  --dataset "$DATA_DIR/train_bpe_tokens.txt" \
  --val-dataset "$DATA_DIR/val_bpe_tokens.txt" \
  --out "$DATA_DIR/compiled_dataset.pt"

echo
echo "Prepared corpus, tokenizer, splits, and compiled dataset."
