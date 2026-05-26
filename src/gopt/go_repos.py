from __future__ import annotations

from pathlib import Path
import subprocess

from gopt.config import GoRepoFetchConfig


def fetch_go_repos(config: GoRepoFetchConfig) -> None:
    manifest_path = Path(config.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"repo manifest does not exist: {manifest_path}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repos = _load_manifest(manifest_path)
    if not repos:
        raise ValueError(f"no repositories found in manifest: {manifest_path}")

    fetched = 0
    updated = 0
    skipped = 0

    for entry in repos:
        url = _normalize_repo_url(entry)
        owner_repo = _owner_repo_from_url(url)
        dest = output_dir / owner_repo.replace("/", "__")

        if dest.exists():
            if not config.update_existing:
                print(f"skip   {owner_repo} (already present)")
                skipped += 1
                continue
            print(f"update {owner_repo}")
            _run_git(["git", "-C", str(dest), "pull", "--ff-only"])
            updated += 1
            continue

        print(f"clone  {owner_repo}")
        clone_cmd = ["git", "clone"]
        if config.shallow:
            clone_cmd.extend(["--depth", "1"])
        clone_cmd.extend([url, str(dest)])
        _run_git(clone_cmd)
        fetched += 1

    print()
    print(f"fetched={fetched} updated={updated} skipped={skipped}")
    print(f"repos root: {output_dir}")


def _load_manifest(path: Path) -> list[str]:
    repos: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        repos.append(line)
    return repos


def _normalize_repo_url(entry: str) -> str:
    if entry.startswith(("https://", "http://", "git@")):
        return entry
    cleaned = entry.removeprefix("github.com/").strip("/")
    if cleaned.count("/") != 1:
        raise ValueError(
            f"manifest entry must be owner/repo, github.com/owner/repo, or a git URL: {entry}"
        )
    return f"https://github.com/{cleaned}.git"


def _owner_repo_from_url(url: str) -> str:
    trimmed = url
    if trimmed.endswith(".git"):
        trimmed = trimmed[:-4]
    if trimmed.startswith("git@github.com:"):
        trimmed = trimmed.removeprefix("git@github.com:")
    elif trimmed.startswith("https://github.com/"):
        trimmed = trimmed.removeprefix("https://github.com/")
    elif trimmed.startswith("http://github.com/"):
        trimmed = trimmed.removeprefix("http://github.com/")
    parts = trimmed.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"unable to derive owner/repo from URL: {url}")
    return f"{parts[-2]}/{parts[-1]}"


def _run_git(args: list[str]) -> None:
    subprocess.run(args, check=True)
