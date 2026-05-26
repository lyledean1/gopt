# gpt

Tiny Go code model project.

This repo now does one thing:

- fetch a representative set of Go repositories
- build a corpus from `.go` files
- train a small Go code transformer on that corpus
- sample Go completions
- evaluate generated files with `gofmt` and `go build`

## Commands

Install dependencies:

```bash
uv sync --dev
```

Show the CLI:

```bash
uv run gpt --help
```

Build a Go corpus from a source tree:

```bash
uv run gpt build-go-corpus --root /path/to/go/repos --out data/go/input.txt
```

Fetch a curated set of Go repositories into one local corpus root:

```bash
uv run gpt fetch-go-repos --manifest manifests/go_repos.txt --out-dir corpora/go
```

Then point the tokenizer/split commands at that checkout root:

```bash
uv run gpt split-go-bpe \
  --root corpora/go \
  --bpe-model data/go/bpe_model.json \
  --train-out data/go/train_bpe_tokens.txt \
  --val-out data/go/val_bpe_tokens.txt
```

Tokenize a Go source tree into a text token stream:

```bash
uv run gpt tokenize-go --root /path/to/go/repos --out data/go/tokens.txt
```

Train a BPE model over identifier and literal payloads:

```bash
uv run gpt train-go-bpe --root /path/to/go/repos --out data/go/bpe_model.json
```

Emit a hybrid Go+BPE token stream:

```bash
uv run gpt tokenize-go-bpe \
  --root /path/to/go/repos \
  --bpe-model data/go/bpe_model.json \
  --out data/go/bpe_tokens.txt
```

Build randomized file-level train/val Go+BPE splits:

```bash
uv run gpt split-go-bpe \
  --root /path/to/go/repos \
  --bpe-model data/go/bpe_model.json \
  --train-out data/go/train_bpe_tokens.txt \
  --val-out data/go/val_bpe_tokens.txt
```

Compile the train/val text corpora into integer ids:

```bash
uv run gpt compile-dataset \
  --dataset data/go/train_bpe_tokens.txt \
  --val-dataset data/go/val_bpe_tokens.txt \
  --out data/go/compiled_dataset.pt
```

Train on the corpus:

```bash
uv run gpt train \
  --dataset data/go/train_bpe_tokens.txt \
  --val-dataset data/go/val_bpe_tokens.txt \
  --compiled-dataset data/go/compiled_dataset.pt \
  --checkpoint checkpoints/go.pt
```

Sample from the trained model:

```bash
uv run gpt sample --checkpoint checkpoints/go.pt --prompt "package main\n\nfunc main() {\n"
```

Evaluate generated samples:

```bash
uv run gpt eval-go --checkpoint checkpoints/go.pt --prompt-file prompts/go.txt --samples 5
```

## Prompt File Format

`eval-go` reads prompt blocks separated by a line containing only `---`.

See [prompts/go.txt](/Users/lyledean/personal/gpt/prompts/go.txt).

## Notes

- The starter manifest in [manifests/go_repos.txt](/Users/lyledean/personal/gpt/manifests/go_repos.txt) is intended to bias training toward ordinary Go, not just the Go compiler/runtime codebase.
- `gofmt` is the first syntax gate.
- `go build` is the stronger gate for complete generated files.
