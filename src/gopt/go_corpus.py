from __future__ import annotations

from pathlib import Path
import os

from gopt.config import GoCorpusConfig


SKIP_FILE_SUFFIXES = (
    ".pb.go",
    ".gen.go",
    "_gen.go",
    "_generated.go",
    "_mock.go",
    "_easyjson.go",
    "_string.go",
)

SKIP_FILE_PREFIXES = (
    "mock_",
)


def _iter_go_files(config: GoCorpusConfig) -> list[Path]:
    root = Path(config.root)
    if not root.exists():
        raise FileNotFoundError(f"go corpus root does not exist: {root}")

    files: list[Path] = []
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in config.skip_dirs)
        current_path = Path(current_root)
        for filename in sorted(filenames):
            path = current_path / filename
            if path.suffix in config.extensions and not _should_skip_file(path):
                files.append(path)
    return files


def _should_skip_file(path: Path) -> bool:
    name = path.name
    if any(name.endswith(suffix) for suffix in SKIP_FILE_SUFFIXES):
        return True
    if any(name.startswith(prefix) for prefix in SKIP_FILE_PREFIXES):
        return True

    try:
        header = path.read_text(encoding="utf-8", errors="ignore")[:2048]
    except OSError:
        return False
    return "Code generated" in header and "DO NOT EDIT." in header


def build_go_corpus(config: GoCorpusConfig) -> None:
    files = _iter_go_files(config)
    if not files:
        raise ValueError(f"no Go files found under {config.root}")

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks: list[str] = []
    root = Path(config.root).resolve()
    for path in files:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue

        if config.include_file_headers:
            rel = path.resolve().relative_to(root)
            chunks.append(f"// FILE: {rel.as_posix()}\n")
        chunks.append(text.rstrip())
        chunks.append("\n\n")

    corpus = "".join(chunks).rstrip() + "\n"
    output_path.write_text(corpus, encoding="utf-8")

    print(f"wrote {len(files)} Go files to {output_path}")
    print(f"corpus bytes={len(corpus.encode('utf-8'))}")
