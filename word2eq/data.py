"""Dataset loading, number masking, tokenisation and data lineage.

Training data (MAWPS / ASDiv-A / SVAMP CSVs) ships with numbers already masked as
``number0 .. numberK``. Raw text at serving time goes through :func:`mask_numbers`
so train and serve share exactly one preprocessing path (no train/serve skew).
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# Numbers as they appear in raw problems: 1,200  3.5  .75  42
_RAW_NUMBER_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\.\d+|\d+)(?![\w])")
_TOKEN_RE = re.compile(r"number\d+|[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[^\sa-z\d]")


@dataclass(frozen=True)
class Example:
    uid: str
    question: str  # masked, e.g. "... number0 kids ... number1 kids ..."
    numbers: tuple[float, ...]
    equation: tuple[str, ...]  # prefix tokens
    answer: float

    @property
    def src_tokens(self) -> list[str]:
        return tokenize(self.question)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def mask_numbers(text: str) -> tuple[str, list[float]]:
    """Replace every literal number with ``numberK`` (K = order of appearance)."""
    numbers: list[float] = []

    def repl(m: re.Match) -> str:
        numbers.append(float(m.group(1).replace(",", "")))
        return f" number{len(numbers) - 1} "

    masked = _RAW_NUMBER_RE.sub(repl, text)
    return re.sub(r"\s+", " ", masked).strip(), numbers


def load_csv(path: str | Path) -> list[Example]:
    path = Path(path)
    examples = []
    with path.open(encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            numbers = tuple(float(x) for x in row["Numbers"].split())
            examples.append(
                Example(
                    uid=f"{path.parent.name}/{path.stem}:{i}",
                    question=row["Question"].strip(),
                    numbers=numbers,
                    equation=tuple(row["Equation"].split()),
                    answer=float(row["Answer"]),
                )
            )
    return examples


def stable_split(examples: list[Example], val_fraction: float, salt: str = "word2eq") -> tuple[list, list]:
    """Deterministic train/val split keyed on content hash, independent of row
    order and of the RNG seed, so a re-shuffled CSV yields the same split."""
    train, val = [], []
    for ex in examples:
        h = hashlib.sha256(f"{salt}:{ex.question}:{' '.join(ex.equation)}".encode()).digest()
        bucket = int.from_bytes(h[:8], "big") / 2**64
        (val if bucket < val_fraction else train).append(ex)
    return train, val


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_manifest(paths: Iterable[str | Path]) -> dict[str, dict]:
    """Content hashes + row counts for every input file, recorded with each run."""
    out = {}
    for p in paths:
        p = Path(p)
        with p.open(encoding="utf-8") as f:
            rows = sum(1 for _ in csv.reader(f)) - 1
        out[p.as_posix()] = {"sha256": file_sha256(p), "rows": rows}
    return out
