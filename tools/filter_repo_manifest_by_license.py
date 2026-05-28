#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_ALLOWED_LICENSES = {
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "MIT",
    "MPL-2.0",
}


def normalize_license(value: str) -> str:
    return value.strip()


def parse_allowed_licenses(raw_values: list[str]) -> set[str]:
    if not raw_values:
        return set(DEFAULT_ALLOWED_LICENSES)
    allowed: set[str] = set()
    for raw in raw_values:
        for part in raw.split(","):
            part = normalize_license(part)
            if part:
                allowed.add(part)
    return allowed


def keep_row(row: dict[str, str], allowed_licenses: set[str]) -> bool:
    include = row.get("include", "").strip().lower()
    if include not in {"y", "yes", "true", "1"}:
        return False

    license_name = normalize_license(row.get("license", ""))
    if not license_name or license_name == "NOASSERTION":
        return False

    return license_name in allowed_licenses


def filter_manifest(input_path: Path, output_path: Path, allowed_licenses: set[str]) -> tuple[int, int]:
    kept: list[str] = []
    total = 0

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"repo", "license", "include"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest is missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            total += 1
            repo = row["repo"].strip()
            if not repo or repo.startswith("#"):
                continue
            if keep_row(row, allowed_licenses):
                kept.append(repo)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Generated from a license-aware manifest.",
        "# Safe default: explicit license + include=y + allowed license family.",
        "",
    ]
    output_path.write_text("\n".join(header + kept) + "\n", encoding="utf-8")
    return total, len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter a license-aware repo manifest into the plain fetch-go-repos format."
    )
    parser.add_argument("--input", required=True, help="Path to tab-separated repo/license manifest.")
    parser.add_argument("--out", required=True, help="Path to write the filtered plain manifest.")
    parser.add_argument(
        "--allow-license",
        action="append",
        default=[],
        help="Allowed license name(s). Repeat or pass a comma-separated list.",
    )
    args = parser.parse_args()

    allowed_licenses = parse_allowed_licenses(args.allow_license)
    total, kept = filter_manifest(Path(args.input), Path(args.out), allowed_licenses)
    print(f"scanned {total} rows")
    print(f"kept {kept} repos")
    print(f"wrote filtered manifest to {args.out}")


if __name__ == "__main__":
    main()
