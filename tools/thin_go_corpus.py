#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SKIP_REPOS = {
    "aws__aws-sdk-go-v2",
    "googleapis__google-cloud-go",
}

SKIP_DIRS = {
    ".git",
    "vendor",
    "node_modules",
    "dist",
    "build",
    "docs",
    "doc",
    "tutorials",
    "_examples",
    "examples",
    "example",
    "bench",
    "benchmark",
    "benchmarks",
    "testdata",
}

SKIP_FILE_SUFFIXES = (
    ".pb.go",
    ".gen.go",
    "_gen.go",
    "_generated.go",
    "_mock.go",
    "_easyjson.go",
    "_string.go",
    "_test.go",
)

SKIP_FILE_PREFIXES = ("mock_",)

KEEP_EXTRA_FILES = {"go.mod", "go.sum"}
GENERATED_HEADER_MARKERS = ("Code generated", "DO NOT EDIT")


def should_skip_dir(path: Path) -> bool:
    return path.name in SKIP_DIRS


def should_skip_repo(path: Path) -> bool:
    return path.name in SKIP_REPOS


def looks_generated(path: Path) -> bool:
    try:
        header = path.read_text(encoding="utf-8", errors="ignore")[:2048]
    except OSError:
        return False
    return all(marker in header for marker in GENERATED_HEADER_MARKERS)


def should_copy_file(path: Path, max_size_bytes: int) -> bool:
    if path.name in KEEP_EXTRA_FILES:
        return True
    if path.suffix != ".go":
        return False
    if any(path.name.endswith(suffix) for suffix in SKIP_FILE_SUFFIXES):
        return False
    if any(path.name.startswith(prefix) for prefix in SKIP_FILE_PREFIXES):
        return False
    try:
        if path.stat().st_size > max_size_bytes:
            return False
    except OSError:
        return False
    if looks_generated(path):
        return False
    return True


def thin_corpus(root: Path, out: Path, max_size_bytes: int) -> tuple[int, int]:
    copied = 0
    skipped = 0

    for repo_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if should_skip_repo(repo_dir):
            skipped += 1
            continue
        for path in repo_dir.rglob("*"):
            relative = path.relative_to(root)

            if path.is_dir():
                if should_skip_dir(path):
                    skipped += 1
                continue

            if any(part in SKIP_DIRS for part in relative.parts[:-1]):
                skipped += 1
                continue

            if not should_copy_file(path, max_size_bytes):
                skipped += 1
                continue

            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied += 1

    return copied, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a thinner mirrored Go corpus by copying only relevant files."
    )
    parser.add_argument("--root", required=True, help="Root directory containing cloned Go repos.")
    parser.add_argument("--out", required=True, help="Output directory for the thinned corpus.")
    parser.add_argument(
        "--max-size-kb",
        type=int,
        default=256,
        help="Skip Go files larger than this many KB.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    copied, skipped = thin_corpus(root, out, args.max_size_kb * 1024)
    print(f"copied {copied} files into {out}")
    print(f"skipped {skipped} paths/files")


if __name__ == "__main__":
    main()
