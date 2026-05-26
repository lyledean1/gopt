# gopt

`gopt` is an attempt at training a small Go code model.

It does one thing end to end:

1. fetch a curated set of public Go repos
2. build a Go-aware tokenized dataset
3. train a small transformer
4. sample Go code
5. evaluate outputs with `gofmt` and `go build`

This is not a general LLM framework. It is a focused repo for experimenting with Go code modeling.

## What This Repo Contains

- `src/gopt/`: Python training, sampling, dataset, and evaluation code
- `tools/go_tokenize/main.go`: Go lexer/token stream builder
- `manifests/go_repos.txt`: curated public Go repos used to build the corpus
- `prompts/go.txt`: prompts used by `eval-go`

Large artifacts like cloned repos, compiled datasets, and checkpoints are intentionally not part of source control.

## Setup

Install dependencies:

```bash
uv sync --dev
```

Show the CLI:

```bash
uv run gopt --help
```

## Workflow

### 1. Fetch the Go repos

This clones or updates the repos listed in `manifests/go_repos.txt`.

```bash
uv run gopt fetch-go-repos \
  --manifest manifests/go_repos.txt \
  --out-dir corpora/go
```

### 2. Train the BPE tokenizer

This learns subword pieces for identifiers and literals from the Go corpus.

```bash
uv run gopt train-go-bpe \
  --root corpora/go \
  --out data/go/bpe_model.json \
  --vocab-size 4096
```

### 3. Build randomized train/val splits

This tokenizes the corpus, applies BPE, and splits files into train and validation sets.

```bash
uv run gopt split-go-bpe \
  --root corpora/go \
  --bpe-model data/go/bpe_model.json \
  --train-out data/go/train_bpe_tokens.txt \
  --val-out data/go/val_bpe_tokens.txt
```

### 4. Compile the dataset

This converts the text token stream into integer ids for faster training.

```bash
uv run gopt compile-dataset \
  --dataset data/go/train_bpe_tokens.txt \
  --val-dataset data/go/val_bpe_tokens.txt \
  --out data/go/compiled_dataset.pt
```

### 5. Train the model

Example local run:

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

Notes:

- training writes the latest checkpoint to the path you pass with `--checkpoint`
- it also writes the best validation checkpoint next to it as `*.best.pt`
- if you rebuild the corpus or tokenizer, start a fresh checkpoint instead of resuming

### 6. Sample from the best checkpoint

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

This generates samples, runs `gofmt`, then tries `go build`.

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

## Corpus Notes

The corpus is built from public GitHub repositories listed in `manifests/go_repos.txt`.

The tokenizer and corpus builder skip low-signal files by default, including:

- vendored directories
- generated files such as `*.pb.go`
- common mock/generated suffixes
- files with `Code generated ... DO NOT EDIT.`

The goal is to bias the model toward ordinary Go code rather than compiler/runtime internals or generated blobs.

## Important Files

- `data/go/bpe_model.json`
  The trained tokenizer model.

- `data/go/train_bpe_tokens.txt`
- `data/go/val_bpe_tokens.txt`
  The text token corpora.

- `data/go/compiled_dataset.pt`
  The compiled integer-id dataset used for training.

- `checkpoints/*.pt`
  Latest checkpoints.

- `checkpoints/*.best.pt`
  Best validation checkpoints.

## Current Practical Limits

On an Apple Silicon laptop, the local sweet spot is roughly:

- `d_model=384`
- `n_layers=8`
- `n_heads=8`
- `block_size=128`

Larger models can train locally, but they become harder to optimize well with small batch sizes.

## Why This Exists

The point of this repo is to make the whole pipeline legible:

- corpus design matters
- tokenization matters
- train/val split quality matters
- decoding matters
- model size matters

It is meant to be small enough to understand, but real enough to learn from.
