# gopt

`gopt` is an attempt at building a small Go code model for learning.

- builds a Go-focused corpus from public repos
- trains a BPE tokenizer for Go identifiers and literals
- trains a small decoder-only transformer
- samples code from prompts
- evaluates outputs with `gofmt` and `go build`

See [SETUP.md](/Users/lyledean/personal/gpt/SETUP.md) for the actual workflow and commands.

## What It Is Not

- not a general LLM framework
- not a polished product
- not a frontier coding model

## Why This Repo Exists

Because the interesting part is seeing the whole system work:

- corpus design
- tokenizer design
- dataset compilation
- model training
- sampling
- evaluation

The repo is intentionally small enough to understand and modify.
