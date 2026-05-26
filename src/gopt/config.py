from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrainingConfig:
    dataset_path: str = "data/go/train_bpe_tokens.txt"
    val_dataset_path: str = "data/go/val_bpe_tokens.txt"
    compiled_dataset_path: str | None = "data/go/compiled_dataset.pt"
    checkpoint_path: str = "checkpoints/go.pt"
    resume: bool = False
    device: str = "auto"
    seed: int = 1337
    batch_size: int = 32
    block_size: int = 128
    max_steps: int = 2000
    eval_interval: int = 100
    eval_batches: int = 20
    learning_rate: float = 1e-3
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.0


@dataclass(slots=True)
class SamplingConfig:
    checkpoint_path: str = "checkpoints/go.pt"
    bpe_model_path: str = "data/go/bpe_model.json"
    prompt: str = "package main\n\nfunc main() {\n"
    max_new_tokens: int = 200
    temperature: float = 1.0
    top_k: int | None = None


@dataclass(slots=True)
class GoCorpusConfig:
    root: str
    output_path: str = "data/go/input.txt"
    include_file_headers: bool = True
    extensions: tuple[str, ...] = (".go",)
    skip_dirs: tuple[str, ...] = (
        ".git",
        "vendor",
        "node_modules",
        "third_party",
        "dist",
        "build",
    )


@dataclass(slots=True)
class GoRepoFetchConfig:
    manifest_path: str = "manifests/go_repos.txt"
    output_dir: str = "corpora/go"
    shallow: bool = True
    update_existing: bool = True


@dataclass(slots=True)
class GoTokenizeConfig:
    root: str
    output_path: str = "data/go/tokens.txt"
    include_file_headers: bool = True


@dataclass(slots=True)
class GoBPEConfig:
    root: str
    model_path: str = "data/go/bpe_model.json"
    vocab_size: int = 512
    min_frequency: int = 2


@dataclass(slots=True)
class GoBPETokenizeConfig:
    root: str
    model_path: str = "data/go/bpe_model.json"
    output_path: str = "data/go/bpe_tokens.txt"
    include_file_headers: bool = True


@dataclass(slots=True)
class GoBPESplitConfig:
    root: str
    model_path: str = "data/go/bpe_model.json"
    train_output_path: str = "data/go/train_bpe_tokens.txt"
    val_output_path: str = "data/go/val_bpe_tokens.txt"
    val_fraction: float = 0.1
    seed: int = 1337
    include_file_headers: bool = True


@dataclass(slots=True)
class CompileDatasetConfig:
    train_dataset_path: str = "data/go/train_bpe_tokens.txt"
    val_dataset_path: str = "data/go/val_bpe_tokens.txt"
    output_path: str = "data/go/compiled_dataset.pt"


@dataclass(slots=True)
class GoEvalConfig:
    checkpoint_path: str = "checkpoints/go.pt"
    bpe_model_path: str = "data/go/bpe_model.json"
    prompt_file: str = "prompts/go.txt"
    sample_count: int = 5
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int | None = None
    keep_samples: bool = False
