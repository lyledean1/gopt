# Setup

This file contains the practical workflow for running `gopt`.

## Setup

Install dependencies:

```bash
uv sync --dev
uv run gopt --help
```

## End-to-End Workflow

### 1. Fetch the repos

```bash
uv run gopt fetch-go-repos \
  --manifest manifests/go_repos.txt \
  --out-dir corpora/go
```

### 2. Train the tokenizer

```bash
uv run gopt train-go-bpe \
  --root corpora/go \
  --out data/go/bpe_model.json \
  --vocab-size 4096
```

### 3. Build train/val splits

```bash
uv run gopt split-go-bpe \
  --root corpora/go \
  --bpe-model data/go/bpe_model.json \
  --train-out data/go/train_bpe_tokens.txt \
  --val-out data/go/val_bpe_tokens.txt
```

### 4. Compile the dataset

```bash
uv run gopt compile-dataset \
  --dataset data/go/train_bpe_tokens.txt \
  --val-dataset data/go/val_bpe_tokens.txt \
  --out data/go/compiled_dataset.pt
```

### 5. Train a local baseline

```bash
uv run gopt train \
  --dataset data/go/train_bpe_tokens.txt \
  --val-dataset data/go/val_bpe_tokens.txt \
  --compiled-dataset data/go/compiled_dataset.pt \
  --checkpoint checkpoints/go-bpe-384x8.pt \
  --device mps \
  --steps 10000 \
  --eval-interval 200 \
  --eval-batches 10 \
  --batch-size 8 \
  --block-size 128 \
  --d-model 384 \
  --n-heads 8 \
  --n-layers 8
```

Training writes:

- latest checkpoint: `checkpoints/go-bpe-384x8.pt`
- best validation checkpoint: `checkpoints/go-bpe-384x8.best.pt`

Use the `*.best.pt` checkpoint when testing.

### 6. Sample code

```bash
uv run gopt sample \
  --checkpoint checkpoints/go-bpe-384x8.best.pt \
  --bpe-model data/go/bpe_model.json \
  --prompt $'package main\n\nfunc main() {\n' \
  --temperature 0.7 \
  --top-k 20
```

Other useful prompts:

- `if err != nil {\n`
- `type Node struct {\n`
- `func NewClient(addr string) (*Client, error) {\n`

### 7. Evaluate outputs

```bash
uv run gopt eval-go \
  --checkpoint checkpoints/go-bpe-384x8.best.pt \
  --bpe-model data/go/bpe_model.json \
  --prompt-file prompts/go.txt \
  --samples 10 \
  --max-new-tokens 200 \
  --temperature 0.7 \
  --top-k 20
```

## Corpus Rules

The corpus is built from public GitHub repos listed in `manifests/go_repos.txt`.

The repo skips low-signal files by default, including:

- `vendor/`
- generated files like `*.pb.go`
- common mock/generated suffixes
- files with `Code generated ... DO NOT EDIT.`

The point is to bias training toward ordinary Go, not compiler/runtime internals or generated blobs.

## Current Local Sweet Spot

On Apple Silicon, the best local baseline so far is roughly:

- `d_model=384`
- `n_layers=8`
- `n_heads=8`
- `block_size=128`

Bigger models can run locally, but they become harder to train well with small batch sizes.
