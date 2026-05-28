#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ALLOWED_LICENSES = {
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "MIT",
    "MPL-2.0",
}

DEFAULT_EXCLUDED_REPOS = {
    "aws/aws-sdk-go-v2",
    "googleapis/google-cloud-go",
}


@dataclass
class RepoRecord:
    repo: str
    stars: int
    license: str
    include: str
    notes: str


def parse_allowed_licenses(values: list[str]) -> set[str]:
    if not values:
        return set(DEFAULT_ALLOWED_LICENSES)
    allowed: set[str] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                allowed.add(part)
    return allowed


def parse_excluded_repos(values: list[str]) -> set[str]:
    excluded = set(DEFAULT_EXCLUDED_REPOS)
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                excluded.add(part)
    return excluded


def github_request(url: str, token: str | None) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "gopt-go-repo-discovery",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def fetch_search_page(
    *,
    min_stars: int,
    page: int,
    per_page: int,
    token: str | None,
) -> dict:
    query = f"language:Go stars:>={min_stars} archived:false fork:false"
    params = urllib.parse.urlencode(
        {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": str(per_page),
            "page": str(page),
        }
    )
    url = f"https://api.github.com/search/repositories?{params}"
    return github_request(url, token)


def fetch_repo_license(owner_repo: str, token: str | None) -> str:
    url = f"https://api.github.com/repos/{owner_repo}"
    payload = github_request(url, token)
    license_info = payload.get("license")
    if not license_info:
        return "NOASSERTION"
    spdx_id = (license_info.get("spdx_id") or "").strip()
    return spdx_id or "NOASSERTION"


def decide_include(
    owner_repo: str,
    license_name: str,
    allowed_licenses: set[str],
    excluded_repos: set[str],
) -> tuple[str, str]:
    if owner_repo in excluded_repos:
        return "n", "explicitly excluded"
    if not license_name or license_name == "NOASSERTION":
        return "n", "missing explicit license"
    if license_name not in allowed_licenses:
        return "n", f"license {license_name} not in allow list"
    return "y", ""


def discover_repos(
    *,
    min_stars: int,
    max_repos: int,
    per_page: int,
    token: str | None,
    allowed_licenses: set[str],
    excluded_repos: set[str],
    sleep_seconds: float,
) -> list[RepoRecord]:
    results: list[RepoRecord] = []
    seen: set[str] = set()
    page = 1

    while len(results) < max_repos:
        payload = fetch_search_page(
            min_stars=min_stars,
            page=page,
            per_page=per_page,
            token=token,
        )
        items = payload.get("items", [])
        if not items:
            break

        for item in items:
            owner_repo = item["full_name"]
            if owner_repo in seen:
                continue
            seen.add(owner_repo)

            stars = int(item.get("stargazers_count", 0))
            license_name = fetch_repo_license(owner_repo, token)
            include, notes = decide_include(
                owner_repo,
                license_name,
                allowed_licenses,
                excluded_repos,
            )
            results.append(
                RepoRecord(
                    repo=owner_repo,
                    stars=stars,
                    license=license_name,
                    include=include,
                    notes=notes,
                )
            )
            if len(results) >= max_repos:
                break
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        page += 1
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return results


def write_tsv(path: Path, rows: list[RepoRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["repo", "stars", "license", "include", "notes"])
        for row in rows:
            writer.writerow([row.repo, row.stars, row.license, row.include, row.notes])


def write_plain_manifest(path: Path, rows: list[RepoRecord]) -> None:
    repos = [row.repo for row in rows if row.include == "y"]
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Generated from GitHub Go repo discovery.",
        "# Only repos with include=y are listed below.",
        "",
    ]
    path.write_text("\n".join(header + repos) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover candidate Go repos from GitHub and emit a license-aware manifest."
    )
    parser.add_argument("--min-stars", type=int, default=500)
    parser.add_argument("--max-repos", type=int, default=100)
    parser.add_argument("--per-page", type=int, default=50)
    parser.add_argument(
        "--out",
        default="manifests/go_repos_discovered.tsv",
        help="TSV output path.",
    )
    parser.add_argument(
        "--out-plain",
        default="",
        help="Optional plain manifest output path containing only include=y repos.",
    )
    parser.add_argument(
        "--allow-license",
        action="append",
        default=[],
        help="Allowed SPDX license(s). Repeat or pass comma-separated values.",
    )
    parser.add_argument(
        "--exclude-repo",
        action="append",
        default=[],
        help="Repo(s) to always exclude, e.g. owner/repo. Repeat or pass comma-separated values.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Sleep between API requests to be polite and reduce rate-limit spikes.",
    )
    parser.add_argument(
        "--github-token",
        default=os.environ.get("GITHUB_TOKEN", ""),
        help="GitHub token. Defaults to GITHUB_TOKEN if set.",
    )
    args = parser.parse_args()

    token = args.github_token.strip() or None
    allowed_licenses = parse_allowed_licenses(args.allow_license)
    excluded_repos = parse_excluded_repos(args.exclude_repo)

    try:
        rows = discover_repos(
            min_stars=args.min_stars,
            max_repos=args.max_repos,
            per_page=args.per_page,
            token=token,
            allowed_licenses=allowed_licenses,
            excluded_repos=excluded_repos,
            sleep_seconds=args.sleep_seconds,
        )
    except urllib.error.HTTPError as exc:
        sys.exit(f"github api error: {exc.code} {exc.reason}")

    write_tsv(Path(args.out), rows)
    if args.out_plain:
        write_plain_manifest(Path(args.out_plain), rows)

    kept = sum(1 for row in rows if row.include == "y")
    print(f"discovered {len(rows)} repos")
    print(f"kept {kept} repos after license/exclusion filters")
    print(f"wrote tsv to {args.out}")
    if args.out_plain:
        print(f"wrote plain manifest to {args.out_plain}")


if __name__ == "__main__":
    main()
