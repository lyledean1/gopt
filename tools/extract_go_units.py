#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:
    from tree_sitter import Language, Parser
    import tree_sitter_go
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "missing dependency: install `tree-sitter` and `tree-sitter-go` to run this script"
    ) from exc


@dataclass(slots=True)
class Unit:
    kind: str
    package_name: str
    imports: list[str]
    source: str


@dataclass(slots=True)
class Decl:
    kind: str
    name: str | None
    receiver_type: str | None
    text: str


def _run(args: list[str], cwd: str | None = None) -> bool:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    return result.returncode == 0


def _node_text(node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8")


def _extract_package_name(root, source: bytes) -> str:
    for child in root.children:
        if child.type == "package_clause":
            text = _node_text(child, source).strip()
            parts = text.split()
            if len(parts) == 2:
                return parts[1]
    return "sample"


def _extract_imports(root, source: bytes) -> list[str]:
    imports: list[str] = []
    for child in root.children:
        if child.type == "import_declaration":
            imports.append(_node_text(child, source))
    return imports


def _build_file(unit: Unit, package_name: str) -> str:
    lines = [f"package {package_name}", ""]
    if unit.imports:
        lines.extend(unit.imports)
        lines.append("")
    lines.append(unit.source.strip())
    lines.append("")
    return "\n".join(lines)


def _validate_go(source_text: str, require_build: bool) -> bool:
    with tempfile.TemporaryDirectory(prefix="gopt-unit-") as tmp:
        tmpdir = Path(tmp)
        (tmpdir / "go.mod").write_text("module sample\n\ngo 1.24.0\n", encoding="utf-8")
        (tmpdir / "main.go").write_text(source_text, encoding="utf-8")

        formatter = ["goimports", "-w", "main.go"] if shutil.which("goimports") else ["gofmt", "-w", "main.go"]
        if not _run(formatter, cwd=tmp):
            return False
        if require_build and not _run(["go", "build", "."], cwd=tmp):
            return False
        return True


def _keep_unit(unit: Unit, max_lines: int) -> bool:
    if unit.kind not in {"func", "struct", "cluster"}:
        return False
    line_count = len(unit.source.splitlines())
    return 0 < line_count <= max_lines


def _extract_type_name(decl_text: str) -> str | None:
    match = re.search(r"\btype\s+([A-Za-z_][A-Za-z0-9_]*)\b", decl_text)
    return match.group(1) if match else None


def _extract_function_name(decl_text: str) -> str | None:
    match = re.search(r"\bfunc\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", decl_text)
    return match.group(1) if match else None


def _extract_method_name_and_receiver(decl_text: str) -> tuple[str | None, str | None]:
    match = re.search(
        r"\bfunc\s*\(\s*[^)]*\*?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        decl_text,
    )
    if not match:
        return None, None
    return match.group(2), match.group(1)


def _returns_type(function_text: str, type_name: str) -> bool:
    return f") {type_name}" in function_text or f") *{type_name}" in function_text


def _declarations_for_file(root, source: bytes) -> list[Decl]:
    declarations: list[Decl] = []
    for child in root.children:
        text = _node_text(child, source)
        if child.type == "type_declaration" and "struct" in text:
            declarations.append(
                Decl(kind="struct", name=_extract_type_name(text), receiver_type=None, text=text)
            )
        elif child.type == "function_declaration":
            declarations.append(
                Decl(kind="func", name=_extract_function_name(text), receiver_type=None, text=text)
            )
        elif child.type == "method_declaration":
            name, receiver_type = _extract_method_name_and_receiver(text)
            declarations.append(
                Decl(kind="method", name=name, receiver_type=receiver_type, text=text)
            )
    return declarations


def _cluster_units(
    declarations: list[Decl],
    *,
    package_name: str,
    imports: list[str],
    max_lines: int,
    require_build: bool,
) -> list[Unit]:
    units: list[Unit] = []
    used_names: set[str] = set()

    for decl in declarations:
        if decl.kind != "struct" or not decl.name or decl.name in used_names:
            continue

        parts = [decl.text.strip()]
        used_names.add(decl.name)

        for candidate in declarations:
            if candidate.kind == "method" and candidate.receiver_type == decl.name:
                parts.append(candidate.text.strip())
            elif (
                candidate.kind == "func"
                and candidate.name
                and candidate.name.startswith("New")
                and _returns_type(candidate.text, decl.name)
            ):
                parts.append(candidate.text.strip())

        unit = Unit(
            kind="cluster",
            package_name=package_name,
            imports=imports,
            source="\n\n".join(parts),
        )
        if not _keep_unit(unit, max_lines):
            continue
        if _validate_go(_build_file(unit, "sample"), require_build):
            units.append(unit)

    return units


def _extract_units(path: Path, parser: Parser, *, max_lines: int, require_build: bool) -> list[Unit]:
    source = path.read_bytes()
    tree = parser.parse(source)
    root = tree.root_node

    package_name = _extract_package_name(root, source)
    imports = _extract_imports(root, source)
    declarations = _declarations_for_file(root, source)

    units = _cluster_units(
        declarations,
        package_name=package_name,
        imports=imports,
        max_lines=max_lines,
        require_build=require_build,
    )

    for decl in declarations:
        if decl.kind == "func":
            unit = Unit(
                kind="func",
                package_name=package_name,
                imports=imports,
                source=decl.text,
            )
            if not _keep_unit(unit, max_lines):
                continue
            if _validate_go(_build_file(unit, "sample"), require_build):
                units.append(unit)
        elif decl.kind == "struct":
            unit = Unit(
                kind="struct",
                package_name=package_name,
                imports=imports,
                source=decl.text,
            )
            if not _keep_unit(unit, max_lines):
                continue
            if _validate_go(_build_file(unit, "sample"), require_build):
                units.append(unit)

    return units


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract small self-contained Go functions and structs into standalone files."
    )
    parser.add_argument("--root", required=True, help="Root directory containing Go files.")
    parser.add_argument("--out", required=True, help="Directory to write extracted units into.")
    parser.add_argument(
        "--max-lines",
        type=int,
        default=40,
        help="Maximum number of lines per extracted unit.",
    )
    parser.add_argument(
        "--require-build",
        action="store_true",
        help="Only keep units that pass `go build` when wrapped into a tiny file.",
    )
    args = parser.parse_args()

    ts_language = Language(tree_sitter_go.language())
    ts_parser = Parser(ts_language)

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for path in root.rglob("*.go"):
        try:
            units = _extract_units(
                path,
                ts_parser,
                max_lines=args.max_lines,
                require_build=args.require_build,
            )
        except Exception:
            continue

        for unit in units:
            out_path = out_dir / f"{written:06d}_{unit.kind}.go"
            out_path.write_text(_build_file(unit, "main"), encoding="utf-8")
            written += 1

    print(f"wrote {written} extracted units to {out_dir}")


if __name__ == "__main__":
    main()
