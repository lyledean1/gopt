from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import torch


@dataclass(slots=True)
class RecordTokenizer:
    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def from_records(cls, records: list[str]) -> "RecordTokenizer":
        vocab = sorted(set(records))
        stoi = {record: i for i, record in enumerate(vocab)}
        itos = {i: record for record, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, records: list[str]) -> list[int]:
        missing = [record for record in records if record not in self.stoi]
        if missing:
            preview = ", ".join(repr(record) for record in missing[:5])
            raise ValueError(f"prompt contains token records not in vocabulary: {preview}")
        return [self.stoi[record] for record in records]

    def decode(self, token_ids: list[int]) -> list[str]:
        return [self.itos[token_id] for token_id in token_ids]


@dataclass(slots=True)
class Corpus:
    tokenizer: RecordTokenizer
    train_tokens: torch.Tensor
    val_tokens: torch.Tensor


def load_corpus(train_path: str, val_path: str) -> Corpus:
    train_records = read_records(train_path)
    val_records = read_records(val_path)
    tokenizer = RecordTokenizer.from_records(train_records + val_records)
    train_tokens = torch.tensor(tokenizer.encode(train_records), dtype=torch.long)
    val_tokens = torch.tensor(tokenizer.encode(val_records), dtype=torch.long)
    return Corpus(
        tokenizer=tokenizer,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
    )


def load_compiled_corpus(path: str) -> Corpus:
    data = torch.load(path, map_location="cpu")
    stoi = data["vocab"]
    train_ids = data["train_ids"]
    val_ids = data["val_ids"]
    if not isinstance(train_ids, torch.Tensor):
        train_ids = torch.tensor(train_ids, dtype=torch.long)
    if not isinstance(val_ids, torch.Tensor):
        val_ids = torch.tensor(val_ids, dtype=torch.long)
    tokenizer = RecordTokenizer(stoi=stoi, itos={i: record for record, i in stoi.items()})
    return Corpus(
        tokenizer=tokenizer,
        train_tokens=train_ids.to(dtype=torch.long),
        val_tokens=val_ids.to(dtype=torch.long),
    )


def compile_corpus(train_path: str, val_path: str, output_path: str) -> None:
    corpus = load_corpus(train_path, val_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vocab": corpus.tokenizer.stoi,
            "train_ids": corpus.train_tokens,
            "val_ids": corpus.val_tokens,
        },
        out,
    )


def read_records(path: str) -> list[str]:
    return [
        line
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sample_batch(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.numel() <= block_size:
        raise ValueError(
            f"dataset is too small for block size {block_size}: only {tokens.numel()} tokens"
        )

    upper = tokens.numel() - block_size - 1
    starts = torch.tensor(
        [rng.randint(0, upper) for _ in range(batch_size)],
        dtype=torch.long,
    )
    offsets = torch.arange(block_size + 1, dtype=torch.long)
    windows = tokens[starts[:, None] + offsets[None, :]]
    return windows[:, :-1], windows[:, 1:]
