from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile

from gopt.config import GoEvalConfig, SamplingConfig
from gopt.runtime import sample_text


PROMPT_SEPARATOR = "\n---\n"


@dataclass(slots=True)
class EvalResult:
    prompt_index: int
    gofmt_ok: bool
    gobuild_ok: bool
    sample_path: str
    error: str | None = None


def _load_prompts(path: str) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    prompts = [block.strip("\n") for block in text.split(PROMPT_SEPARATOR)]
    prompts = [prompt for prompt in prompts if prompt.strip()]
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def _run_command(args: list[str], cwd: str) -> bool:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    return result.returncode == 0


def _write_go_module(tmpdir: str, sample: str) -> Path:
    temp_path = Path(tmpdir)
    (temp_path / "go.mod").write_text("module sample\n\ngo 1.24.0\n", encoding="utf-8")
    sample_path = temp_path / "main.go"
    sample_path.write_text(sample, encoding="utf-8")
    return sample_path


def eval_go(config: GoEvalConfig) -> None:
    if shutil.which("gofmt") is None:
        raise RuntimeError("gofmt not found in PATH")
    if shutil.which("go") is None:
        raise RuntimeError("go not found in PATH")

    prompts = _load_prompts(config.prompt_file)
    results: list[EvalResult] = []

    for sample_index in range(config.sample_count):
        prompt = prompts[sample_index % len(prompts)]
        try:
            sample = sample_text(
                SamplingConfig(
                    checkpoint_path=config.checkpoint_path,
                    bpe_model_path=config.bpe_model_path,
                    prompt=prompt,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_k=config.top_k,
                )
            )
        except ValueError as exc:
            results.append(
                EvalResult(
                    prompt_index=sample_index % len(prompts),
                    gofmt_ok=False,
                    gobuild_ok=False,
                    sample_path="",
                    error=str(exc),
                )
            )
            print(
                f"sample {sample_index:03d} "
                f"prompt={sample_index % len(prompts)} "
                f"error={exc}"
            )
            continue

        if config.keep_samples:
            out_dir = Path("samples") / f"eval-{sample_index:03d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            sample_path = _write_go_module(str(out_dir), sample)
            workdir = str(out_dir)
        else:
            tmpdir = tempfile.TemporaryDirectory(prefix="gopt-go-eval-")
            sample_path = _write_go_module(tmpdir.name, sample)
            workdir = tmpdir.name

        gofmt_ok = _run_command(["gofmt", "-w", sample_path.name], workdir)
        gobuild_ok = gofmt_ok and _run_command(["go", "build", "."], workdir)

        results.append(
            EvalResult(
                prompt_index=sample_index % len(prompts),
                gofmt_ok=gofmt_ok,
                gobuild_ok=gobuild_ok,
                sample_path=str(sample_path),
            )
        )

        print(
            f"sample {sample_index:03d} "
            f"prompt={sample_index % len(prompts)} "
            f"gofmt={'ok' if gofmt_ok else 'fail'} "
            f"gobuild={'ok' if gobuild_ok else 'fail'} "
            f"path={sample_path}"
        )

        if not config.keep_samples:
            tmpdir.cleanup()

    gofmt_passes = sum(1 for result in results if result.gofmt_ok)
    gobuild_passes = sum(1 for result in results if result.gobuild_ok)
    print()
    print(f"gofmt pass rate:  {gofmt_passes}/{len(results)}")
    print(f"go build pass rate: {gobuild_passes}/{len(results)}")
