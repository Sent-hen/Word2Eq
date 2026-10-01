"""Token vocabularies, persisted as plain JSON (never pickle) so artifacts are
safe to load from untrusted storage and diffable in review."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

PAD, BOS, EOS, UNK = "<pad>", "<s>", "</s>", "<unk>"
SPECIALS = (PAD, BOS, EOS, UNK)
PAD_ID, BOS_ID, EOS_ID, UNK_ID = range(4)
MAX_NUMBERS = 10  # number0 .. number9 are always in both vocabularies


class Vocab:
    def __init__(self, itos: list[str]):
        if tuple(itos[: len(SPECIALS)]) != SPECIALS:
            raise ValueError("vocab must start with the special tokens")
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, sequences: Iterable[Iterable[str]], min_freq: int = 1, reserved: Iterable[str] = ()):
        counts = Counter(tok for seq in sequences for tok in seq)
        itos = list(SPECIALS)
        for tok in reserved:
            if tok not in itos:
                itos.append(tok)
        seen = set(itos)
        for tok, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if c >= min_freq and tok not in seen:
                itos.append(tok)
                seen.add(tok)
        return cls(itos)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: Iterable[str], add_bos_eos: bool = True) -> list[int]:
        ids = [self.stoi.get(t, UNK_ID) for t in tokens]
        return [BOS_ID, *ids, EOS_ID] if add_bos_eos else ids

    def decode(self, ids: Iterable[int]) -> list[str]:
        out = []
        for i in ids:
            if i == EOS_ID:
                break
            if i in (PAD_ID, BOS_ID):
                continue
            out.append(self.itos[i])
        return out

    def to_dict(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, d: dict) -> Vocab:
        return cls(d["itos"])


def number_tokens() -> list[str]:
    return [f"number{i}" for i in range(MAX_NUMBERS)]


def save_vocabs(path: str | Path, src: Vocab, tgt: Vocab) -> None:
    Path(path).write_text(json.dumps({"src": src.to_dict(), "tgt": tgt.to_dict()}, indent=1))


def load_vocabs(path: str | Path) -> tuple[Vocab, Vocab]:
    d = json.loads(Path(path).read_text())
    return Vocab.from_dict(d["src"]), Vocab.from_dict(d["tgt"])
