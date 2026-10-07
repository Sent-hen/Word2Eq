"""Model runner: raw text in, verified equations out. Framework-agnostic and
synchronous; the async batcher calls it from a dedicated worker thread."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from ..artifact import load_artifact
from ..data import mask_numbers, tokenize
from ..decoding import GrammarTables
from ..evaluate import predict
from ..expr import evaluate_prefix, to_infix
from ..vocab import MAX_NUMBERS


class InvalidProblem(ValueError):
    pass


@dataclass
class Prepared:
    tokens: list[str]
    numbers: list[float]


@dataclass
class Solution:
    equation_prefix: list[str]
    equation: str | None  # infix with concrete numbers
    answer: float | None
    numbers: list[float]
    verified: bool  # expression well-formed AND evaluated to a finite value


def prepare(problem: str, max_src_tokens: int) -> Prepared:
    masked, numbers = mask_numbers(problem)
    if not numbers:
        raise InvalidProblem("no numeric quantities found in problem")
    if len(numbers) > MAX_NUMBERS:
        raise InvalidProblem(f"at most {MAX_NUMBERS} numeric quantities are supported")
    tokens = tokenize(masked)
    if len(tokens) > max_src_tokens:
        raise InvalidProblem(f"problem too long: {len(tokens)} tokens > {max_src_tokens}")
    return Prepared(tokens, numbers)


class ModelRunner:
    def __init__(self, artifact_dir: str | Path, quantize: bool = False, num_threads: int | None = None,
                 constrained: bool = True):
        if num_threads:
            torch.set_num_threads(num_threads)
        model, self.src_vocab, self.tgt_vocab, self.card = load_artifact(artifact_dir)
        if quantize:
            # Dynamic int8 quantisation of Linear layers: smaller + faster on CPU.
            # The fused encoder fast path can't inspect quantized weights, so disable it.
            torch.backends.mha.set_fastpath_enabled(False)
            model =torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        self.model = model.eval()
        self.quantized = quantize
        self.version = self.card.get("version", "unknown")
        self.max_eq_len = int(self.card.get("max_eq_len", 32))
        self.max_src_tokens = self.model.cfg.max_len - 2
        self.grammar = GrammarTables.from_vocab(self.tgt_vocab) if constrained else None

    def solve_batch(self, items: list[Prepared]) -> list[Solution]:
        preds = predict(
            self.model,
            self.src_vocab,
            self.tgt_vocab,
            [it.tokens for it in items],
            [len(it.numbers) for it in items],
            self.grammar,
            batch_size=max(1, len(items)),
            max_eq_len=self.max_eq_len,
        )
        out = []
        for it, p in zip(items, preds, strict=True):
            ans = evaluate_prefix(p, it.numbers)
            out.append(Solution(p, to_infix(p, it.numbers), ans, it.numbers, verified=ans is not None))
        return out

    def warmup(self, rounds: int = 3) -> float:
        """Run a few inferences so lazy init / allocator warm-up doesn't hit the
        first real request. Returns the last round's latency in seconds."""
        item = prepare("Tom had 5 apples and bought 3 more. How many apples does Tom have now?",
                       self.max_src_tokens)
        dt = 0.0
        for _ in range(rounds):
            t0 = time.perf_counter()
            self.solve_batch([item])
            dt = time.perf_counter() - t0
        return dt
