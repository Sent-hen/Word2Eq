"""Offline evaluation: equation exact-match, verified answer accuracy and
grammar-validity rate, for constrained vs. unconstrained decoding."""

from __future__ import annotations

import time
from collections.abc import Sequence

import torch

from .data import Example
from .decoding import GrammarTables, greedy_decode
from .expr import answers_match, evaluate_prefix, is_valid_prefix
from .model import Seq2SeqTransformer
from .tensors import bucketed_batches, encode_sources
from .vocab import Vocab


def predict(
    model: Seq2SeqTransformer,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    token_lists: Sequence[list[str]],
    n_numbers: Sequence[int],
    grammar: GrammarTables | None,
    batch_size: int = 128,
    max_eq_len: int = 32,
    device: torch.device | str = "cpu",
) -> list[list[str]]:
    model.eval()
    preds: list[list[str]] = [[] for _ in token_lists]
    for idx in bucketed_batches([len(t) for t in token_lists], batch_size, generator=None):
        src = encode_sources([token_lists[i] for i in idx], src_vocab, model.cfg.max_len).to(device)
        nn_ = torch.tensor([n_numbers[i] for i in idx], dtype=torch.long, device=device)
        out = greedy_decode(model, src, nn_, grammar, max_len=max_eq_len).cpu().tolist()
        for i, ids in zip(idx, out, strict=True):
            preds[i] = tgt_vocab.decode(ids)
    return preds


def score(examples: Sequence[Example], preds: Sequence[list[str]]) -> dict:
    n = len(examples)
    eq = ans = valid = 0
    for ex, p in zip(examples, preds, strict=True):
        eq += list(ex.equation) == p
        valid += is_valid_prefix(p)
        ans += answers_match(evaluate_prefix(p, list(ex.numbers)), ex.answer)
    return {
        "n": n,
        "equation_acc": eq / n if n else 0.0,
        "answer_acc": ans / n if n else 0.0,
        "valid_rate": valid / n if n else 0.0,
    }


def evaluate(
    model: Seq2SeqTransformer,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    examples: Sequence[Example],
    constrained: bool = True,
    batch_size: int = 128,
    max_eq_len: int = 32,
    device: torch.device | str = "cpu",
) -> dict:
    grammar = GrammarTables.from_vocab(tgt_vocab, device) if constrained else None
    t0 = time.perf_counter()
    preds = predict(
        model,
        src_vocab,
        tgt_vocab,
        [ex.src_tokens for ex in examples],
        [len(ex.numbers) for ex in examples],
        grammar,
        batch_size=batch_size,
        max_eq_len=max_eq_len,
        device=device,
    )
    elapsed = time.perf_counter() - t0
    metrics = score(examples, preds)
    metrics["examples_per_sec"] = len(examples) / elapsed if elapsed > 0 else 0.0
    return metrics
