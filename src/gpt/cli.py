from __future__ import annotations

import argparse

from gpt.config import (
    CompileDatasetConfig,
    GoBPEConfig,
    GoBPESplitConfig,
    GoBPETokenizeConfig,
    GoCorpusConfig,
    GoEvalConfig,
    GoRepoFetchConfig,
    GoTokenizeConfig,
    SamplingConfig,
    TrainingConfig,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpt",
        description="Train and evaluate a tiny character-level transformer on Go code.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build-go-corpus",
        help="Build one Go training corpus from a directory tree of .go files.",
    )
    build_parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing Go source trees.",
    )
    build_parser.add_argument(
        "--out",
        default="data/go/input.txt",
        help="Path to write the concatenated corpus.",
    )
    build_parser.add_argument(
        "--no-file-headers",
        action="store_true",
        help="Do not insert // FILE: path.go separators in the corpus.",
    )

    fetch_parser = subparsers.add_parser(
        "fetch-go-repos",
        help="Clone or update a manifest of Go repositories into one local corpus root.",
    )
    fetch_parser.add_argument(
        "--manifest",
        default="manifests/go_repos.txt",
        help="Path to a manifest containing one repo per line.",
    )
    fetch_parser.add_argument(
        "--out-dir",
        default="corpora/go",
        help="Directory to clone repositories into.",
    )
    fetch_parser.add_argument(
        "--full-history",
        action="store_true",
        help="Clone full git history instead of a shallow checkout.",
    )
    fetch_parser.add_argument(
        "--no-update",
        action="store_true",
        help="Do not pull repositories that are already present.",
    )

    tokenize_parser = subparsers.add_parser(
        "tokenize-go",
        help="Tokenize Go source trees into a plain-text token stream.",
    )
    tokenize_parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing Go source trees.",
    )
    tokenize_parser.add_argument(
        "--out",
        default="data/go/tokens.txt",
        help="Path to write the serialized token stream.",
    )
    tokenize_parser.add_argument(
        "--no-file-headers",
        action="store_true",
        help="Do not insert FILE records in the token stream.",
    )

    train_bpe_parser = subparsers.add_parser(
        "train-go-bpe",
        help="Train a BPE model over Go identifier and literal payloads.",
    )
    train_bpe_parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing Go source trees.",
    )
    train_bpe_parser.add_argument(
        "--out",
        default="data/go/bpe_model.json",
        help="Path to write the BPE model JSON.",
    )
    train_bpe_parser.add_argument(
        "--vocab-size",
        type=int,
        default=512,
        help="Approximate vocabulary size including learned merges.",
    )
    train_bpe_parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum pair frequency required to add a merge.",
    )

    tokenize_bpe_parser = subparsers.add_parser(
        "tokenize-go-bpe",
        help="Tokenize Go into structural tokens plus BPE payload pieces.",
    )
    tokenize_bpe_parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing Go source trees.",
    )
    tokenize_bpe_parser.add_argument(
        "--bpe-model",
        default="data/go/bpe_model.json",
        help="Path to the trained BPE model JSON.",
    )
    tokenize_bpe_parser.add_argument(
        "--out",
        default="data/go/bpe_tokens.txt",
        help="Path to write the hybrid BPE token stream.",
    )
    tokenize_bpe_parser.add_argument(
        "--no-file-headers",
        action="store_true",
        help="Do not insert FILE records in the BPE token stream.",
    )

    split_bpe_parser = subparsers.add_parser(
        "split-go-bpe",
        help="Build randomized train/val Go+BPE corpora by file.",
    )
    split_bpe_parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing Go source trees.",
    )
    split_bpe_parser.add_argument(
        "--bpe-model",
        default="data/go/bpe_model.json",
        help="Path to the trained BPE model JSON.",
    )
    split_bpe_parser.add_argument(
        "--train-out",
        default="data/go/train_bpe_tokens.txt",
        help="Path to write the training split.",
    )
    split_bpe_parser.add_argument(
        "--val-out",
        default="data/go/val_bpe_tokens.txt",
        help="Path to write the validation split.",
    )
    split_bpe_parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of files to assign to validation.",
    )
    split_bpe_parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for file-level shuffling.",
    )
    split_bpe_parser.add_argument(
        "--no-file-headers",
        action="store_true",
        help="Do not insert FILE records in the split corpora.",
    )

    compile_parser = subparsers.add_parser(
        "compile-dataset",
        help="Compile Go+BPE train/val corpora into one integer-id dataset file.",
    )
    compile_parser.add_argument(
        "--dataset",
        default="data/go/train_bpe_tokens.txt",
        help="Path to the Go+BPE training corpus.",
    )
    compile_parser.add_argument(
        "--val-dataset",
        default="data/go/val_bpe_tokens.txt",
        help="Path to the Go+BPE validation corpus.",
    )
    compile_parser.add_argument(
        "--out",
        default="data/go/compiled_dataset.pt",
        help="Path to write the compiled dataset file.",
    )

    train_parser = subparsers.add_parser("train", help="Train the Go code model.")
    train_parser.add_argument(
        "--dataset",
        default="data/go/train_bpe_tokens.txt",
        help="Path to the Go+BPE training corpus.",
    )
    train_parser.add_argument(
        "--val-dataset",
        default="data/go/val_bpe_tokens.txt",
        help="Path to the Go+BPE validation corpus.",
    )
    train_parser.add_argument(
        "--compiled-dataset",
        default="data/go/compiled_dataset.pt",
        help="Path to a compiled dataset file to load if present.",
    )
    train_parser.add_argument(
        "--checkpoint",
        default="checkpoints/go.pt",
        help="Path to write the checkpoint file.",
    )
    train_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the checkpoint instead of starting over.",
    )
    train_parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to use, for example auto, cpu, or mps.",
    )
    train_parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of training steps to run.",
    )
    train_parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="How many training steps to run between evaluations.",
    )
    train_parser.add_argument(
        "--eval-batches",
        type=int,
        default=20,
        help="How many random batches to average for each evaluation pass.",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size.",
    )
    train_parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Context length in tokens.",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    train_parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Transformer model width.",
    )
    train_parser.add_argument(
        "--n-heads",
        type=int,
        default=4,
        help="Number of attention heads.",
    )
    train_parser.add_argument(
        "--n-layers",
        type=int,
        default=4,
        help="Number of transformer blocks.",
    )
    train_parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability.",
    )

    sample_parser = subparsers.add_parser("sample", help="Sample Go code from a trained model.")
    sample_parser.add_argument(
        "--checkpoint",
        default="checkpoints/go.pt",
        help="Path to the checkpoint file.",
    )
    sample_parser.add_argument(
        "--bpe-model",
        default="data/go/bpe_model.json",
        help="Path to the tokenizers BPE model JSON.",
    )
    sample_parser.add_argument(
        "--prompt",
        default="package main\n\nfunc main() {\n",
        help="Go prompt to seed generation.",
    )
    sample_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Number of tokens to generate.",
    )
    sample_parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    sample_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Restrict sampling to the top-k most likely next tokens.",
    )

    eval_parser = subparsers.add_parser(
        "eval-go",
        help="Generate Go files and evaluate them with gofmt and go build.",
    )
    eval_parser.add_argument(
        "--checkpoint",
        default="checkpoints/go.pt",
        help="Path to the trained checkpoint.",
    )
    eval_parser.add_argument(
        "--bpe-model",
        default="data/go/bpe_model.json",
        help="Path to the tokenizers BPE model JSON.",
    )
    eval_parser.add_argument(
        "--prompt-file",
        default="prompts/go.txt",
        help="Prompt file with blocks separated by a line containing only ---.",
    )
    eval_parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="How many prompts/samples to generate.",
    )
    eval_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Number of characters to generate per sample.",
    )
    eval_parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for evaluation.",
    )
    eval_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Restrict sampling to the top-k most likely next tokens.",
    )
    eval_parser.add_argument(
        "--keep-samples",
        action="store_true",
        help="Keep generated temporary Go files instead of deleting them.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build-go-corpus":
        from gpt.go_corpus import build_go_corpus

        build_go_corpus(
            GoCorpusConfig(
                root=args.root,
                output_path=args.out,
                include_file_headers=not args.no_file_headers,
            )
        )
        return

    if args.command == "fetch-go-repos":
        from gpt.go_repos import fetch_go_repos

        fetch_go_repos(
            GoRepoFetchConfig(
                manifest_path=args.manifest,
                output_dir=args.out_dir,
                shallow=not args.full_history,
                update_existing=not args.no_update,
            )
        )
        return

    if args.command == "tokenize-go":
        from gpt.go_tokenize import tokenize_go_corpus

        tokenize_go_corpus(
            GoTokenizeConfig(
                root=args.root,
                output_path=args.out,
                include_file_headers=not args.no_file_headers,
            )
        )
        return

    if args.command == "train-go-bpe":
        from gpt.go_bpe import train_go_bpe

        train_go_bpe(
            GoBPEConfig(
                root=args.root,
                model_path=args.out,
                vocab_size=args.vocab_size,
                min_frequency=args.min_frequency,
            )
        )
        return

    if args.command == "tokenize-go-bpe":
        from gpt.go_bpe import tokenize_go_bpe

        tokenize_go_bpe(
            GoBPETokenizeConfig(
                root=args.root,
                model_path=args.bpe_model,
                output_path=args.out,
                include_file_headers=not args.no_file_headers,
            )
        )
        return

    if args.command == "split-go-bpe":
        from gpt.go_bpe import split_go_bpe_corpus

        split_go_bpe_corpus(
            GoBPESplitConfig(
                root=args.root,
                model_path=args.bpe_model,
                train_output_path=args.train_out,
                val_output_path=args.val_out,
                val_fraction=args.val_fraction,
                seed=args.seed,
                include_file_headers=not args.no_file_headers,
            )
        )
        return

    if args.command == "compile-dataset":
        from gpt.runtime import compile_dataset

        compile_dataset(
            CompileDatasetConfig(
                train_dataset_path=args.dataset,
                val_dataset_path=args.val_dataset,
                output_path=args.out,
            )
        )
        return

    if args.command == "train":
        from gpt.runtime import train

        train(
            TrainingConfig(
                dataset_path=args.dataset,
                val_dataset_path=args.val_dataset,
                compiled_dataset_path=args.compiled_dataset,
                checkpoint_path=args.checkpoint,
                resume=args.resume,
                device=args.device,
                max_steps=args.steps,
                eval_interval=args.eval_interval,
                eval_batches=args.eval_batches,
                batch_size=args.batch_size,
                block_size=args.block_size,
                learning_rate=args.learning_rate,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                dropout=args.dropout,
            )
        )
        return

    if args.command == "sample":
        from gpt.runtime import sample

        sample(
            SamplingConfig(
                checkpoint_path=args.checkpoint,
                bpe_model_path=args.bpe_model,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        )
        return

    if args.command == "eval-go":
        from gpt.go_eval import eval_go

        eval_go(
            GoEvalConfig(
                checkpoint_path=args.checkpoint,
                bpe_model_path=args.bpe_model,
                prompt_file=args.prompt_file,
                sample_count=args.samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                keep_samples=args.keep_samples,
            )
        )
        return

    parser.error(f"unknown command: {args.command}")
