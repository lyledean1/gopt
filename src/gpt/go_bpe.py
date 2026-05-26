from __future__ import annotations

from pathlib import Path
import json
import random
import tempfile

from gpt.config import GoBPEConfig, GoBPESplitConfig, GoBPETokenizeConfig
from gpt.go_tokenize import run_go_tokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


PAYLOAD_LABELS = {"IDENT", "INT", "FLOAT", "IMAG", "CHAR", "STRING"}
PUNCT_NO_SPACE_BEFORE = {".", ",", ";", ")", "]", "}", ":"}
PUNCT_NO_SPACE_AFTER = {"(", "[", "{", "."}


def train_go_bpe(config: GoBPEConfig) -> None:
    with tempfile.TemporaryDirectory(prefix="gpt-go-bpe-") as tmpdir:
        raw_tokens_path = Path(tmpdir) / "raw_tokens.txt"
        run_go_tokenizer(
            root=config.root,
            output_path=str(raw_tokens_path),
            include_file_headers=False,
            preserve_values=True,
        )
        texts = [value for label, value in iter_records(raw_tokens_path) if label in PAYLOAD_LABELS and value]

    model_path = Path(config.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path = model_path.with_suffix(".payloads.txt")
    payload_path.write_text("\n".join(texts) + "\n", encoding="utf-8")

    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
    trainer = BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=["<UNK>", "<EMPTY>"],
        show_progress=True,
    )
    tokenizer.train([str(payload_path)], trainer)
    tokenizer.save(str(model_path))
    vocab_size = tokenizer.get_vocab_size()
    print(f"trained tokenizers BPE model with vocab size {vocab_size} and wrote {model_path}")


def tokenize_go_bpe(config: GoBPETokenizeConfig) -> None:
    tokenizer = Tokenizer.from_file(config.model_path)

    with tempfile.TemporaryDirectory(prefix="gpt-go-bpe-") as tmpdir:
        raw_tokens_path = Path(tmpdir) / "raw_tokens.txt"
        run_go_tokenizer(
            root=config.root,
            output_path=str(raw_tokens_path),
            include_file_headers=config.include_file_headers,
            preserve_values=True,
        )
        records = list(iter_records(raw_tokens_path))

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for label, value in records:
            if label in PAYLOAD_LABELS and value is not None:
                f.write(f"{label}_START\n")
                encoded = tokenizer.encode(value)
                pieces = encoded.tokens or ["<EMPTY>"]
                for piece in pieces:
                    write_record(f, f"{label}_PIECE", piece)
                f.write(f"{label}_END\n")
                continue

            write_record(f, label, value)

    print(f"wrote BPE token stream to {output_path}")


def split_go_bpe_corpus(config: GoBPESplitConfig) -> None:
    tokenizer = Tokenizer.from_file(config.model_path)

    with tempfile.TemporaryDirectory(prefix="gpt-go-bpe-split-") as tmpdir:
        raw_tokens_path = Path(tmpdir) / "raw_tokens.txt"
        run_go_tokenizer(
            root=config.root,
            output_path=str(raw_tokens_path),
            include_file_headers=config.include_file_headers,
            preserve_values=True,
        )
        grouped_files = group_records_by_file(list(iter_records(raw_tokens_path)))

    rng = random.Random(config.seed)
    rng.shuffle(grouped_files)

    if not grouped_files:
        raise ValueError("no tokenized files found for split")

    val_count = max(1, int(len(grouped_files) * config.val_fraction))
    train_groups = grouped_files[val_count:]
    val_groups = grouped_files[:val_count]
    if not train_groups:
        raise ValueError("validation split consumed every file; reduce val_fraction")

    train_records = flatten_encoded_groups(train_groups, tokenizer)
    val_records = flatten_encoded_groups(val_groups, tokenizer)

    train_path = Path(config.train_output_path)
    val_path = Path(config.val_output_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text("\n".join(train_records) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(val_records) + "\n", encoding="utf-8")

    print(
        f"wrote train split to {train_path} ({len(train_groups)} files) "
        f"and val split to {val_path} ({len(val_groups)} files)"
    )


def tokenize_prompt(prompt: str, model_path: str) -> list[str]:
    tokenizer = Tokenizer.from_file(model_path)
    with tempfile.TemporaryDirectory(prefix="gpt-go-prompt-") as tmpdir:
        tmp_root = Path(tmpdir)
        prompt_path = tmp_root / "prompt.go"
        prompt_path.write_text(prompt, encoding="utf-8")
        raw_tokens_path = tmp_root / "raw_tokens.txt"
        run_go_tokenizer(
            root=str(tmp_root),
            output_path=str(raw_tokens_path),
            include_file_headers=False,
            preserve_values=True,
        )
        records = list(iter_records(raw_tokens_path))
    return encode_records_with_bpe(records, tokenizer)


def render_bpe_records(records: list[str]) -> str:
    output_parts: list[str] = []
    payload_label: str | None = None
    payload_pieces: list[str] = []
    prev_plain: str | None = None
    pending_semicolon = False

    def flush_payload() -> None:
        nonlocal payload_label, payload_pieces, prev_plain
        if payload_label is None:
            return
        text = "".join(payload_pieces) or _placeholder_for_payload(payload_label)
        _append_plain(text)
        payload_label = None
        payload_pieces = []

    def _append_plain(text: str) -> None:
        nonlocal prev_plain
        if not output_parts:
            output_parts.append(text)
            prev_plain = text
            return
        if prev_plain == "\n":
            output_parts.append(text)
            prev_plain = text
            return
        if text in PUNCT_NO_SPACE_BEFORE:
            output_parts.append(text)
            prev_plain = text
            return
        if prev_plain in PUNCT_NO_SPACE_AFTER:
            output_parts.append(text)
            prev_plain = text
            return
        output_parts.append(" ")
        output_parts.append(text)
        prev_plain = text

    def flush_pending_semicolon(next_label: str | None) -> None:
        nonlocal pending_semicolon
        if not pending_semicolon:
            return
        # The Go scanner inserts semicolons before newlines and closing delimiters.
        # Suppress those so rendered output looks like normal Go source, but keep
        # explicit semicolons that separate clauses such as in a for-header.
        if next_label in {"NEWLINE", "}", ")", None}:
            pending_semicolon = False
            return
        _append_plain(";")
        pending_semicolon = False

    for record in records:
        label, value = split_record(record)
        if label == "NEWLINE":
            flush_pending_semicolon(label)
            flush_payload()
            if output_parts and output_parts[-1] != "\n":
                output_parts.append("\n")
            prev_plain = "\n"
            continue
        if label == "FILE":
            flush_pending_semicolon(label)
            flush_payload()
            continue
        if label.endswith("_START"):
            flush_pending_semicolon(label)
            flush_payload()
            payload_label = label.removesuffix("_START")
            payload_pieces = []
            continue
        if label.endswith("_PIECE"):
            if value is not None:
                payload_pieces.append(value)
            continue
        if label.endswith("_END"):
            flush_pending_semicolon(label)
            flush_payload()
            continue

        if label == ";":
            flush_payload()
            pending_semicolon = True
            continue

        flush_pending_semicolon(label)
        flush_payload()
        _append_plain(label)

    flush_pending_semicolon(None)
    flush_payload()
    return "".join(output_parts).rstrip() + "\n"


def encode_records_with_bpe(records: list[tuple[str, str | None]], tokenizer: Tokenizer) -> list[str]:
    output: list[str] = []
    for label, value in records:
        if label in PAYLOAD_LABELS and value is not None:
            output.append(f"{label}_START")
            encoded = tokenizer.encode(value)
            pieces = encoded.tokens or ["<EMPTY>"]
            output.extend(_format_record(f"{label}_PIECE", piece) for piece in pieces)
            output.append(f"{label}_END")
            continue
        output.append(_format_record(label, value))
    return output


def iter_records(path: Path) -> list[tuple[str, str | None]]:
    records: list[tuple[str, str | None]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            records.append((line, None))
            continue
        label, raw_value = line.split("\t", 1)
        records.append((label, json.loads(raw_value)))
    return records


def group_records_by_file(records: list[tuple[str, str | None]]) -> list[list[tuple[str, str | None]]]:
    groups: list[list[tuple[str, str | None]]] = []
    current: list[tuple[str, str | None]] = []

    for record in records:
        label, _ = record
        if label == "FILE":
            if current:
                groups.append(current)
            current = [record]
            continue
        if not current:
            current = [record]
        else:
            current.append(record)

    if current:
        groups.append(current)
    return groups


def flatten_encoded_groups(
    groups: list[list[tuple[str, str | None]]],
    tokenizer: Tokenizer,
) -> list[str]:
    output: list[str] = []
    for group in groups:
        output.extend(encode_records_with_bpe(group, tokenizer))
    return output


def write_record(handle, label: str, value: str | None) -> None:
    if value is None:
        handle.write(f"{label}\n")
        return
    handle.write(f"{label}\t{json.dumps(value, ensure_ascii=False)}\n")


def split_record(record: str) -> tuple[str, str | None]:
    if "\t" not in record:
        return record, None
    label, raw_value = record.split("\t", 1)
    return label, json.loads(raw_value)


def _format_record(label: str, value: str | None) -> str:
    if value is None:
        return label
    return f"{label}\t{json.dumps(value, ensure_ascii=False)}"


def _placeholder_for_payload(label: str) -> str:
    return {
        "IDENT": "ident",
        "INT": "0",
        "FLOAT": "0.0",
        "IMAG": "0i",
        "CHAR": "'x'",
        "STRING": "\"\"",
    }.get(label, "x")
