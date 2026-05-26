from __future__ import annotations

from pathlib import Path
import os
import subprocess

from gopt.config import GoTokenizeConfig


def tokenize_go_corpus(config: GoTokenizeConfig) -> None:
    stdout = run_go_tokenizer(
        root=config.root,
        output_path=config.output_path,
        include_file_headers=config.include_file_headers,
    )
    if stdout:
        print(stdout)


def run_go_tokenizer(
    *,
    root: str,
    output_path: str,
    include_file_headers: bool,
    preserve_values: bool = False,
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    go_cache = Path(".gocache")
    go_cache.mkdir(exist_ok=True)

    cmd = [
        "go",
        "run",
        "./tools/go_tokenize/main.go",
        "--root",
        root,
        "--out",
        str(output_path),
    ]
    if not include_file_headers:
        cmd.append("--no-file-headers")
    if preserve_values:
        cmd.append("--preserve-values")

    env = dict(os.environ, GOCACHE=str(go_cache.resolve()))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "go tokenizer failed")

    return result.stdout.strip()
