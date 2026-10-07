"""Prefix-notation equation utilities: validation, evaluation and infix rendering.

The model emits equations in prefix (Polish) notation over *masked* operands, e.g.
``- number1 number0``. Keeping the target grammar this small is what lets us
constrain decoding and deterministically verify every answer the service returns.
"""

from __future__ import annotations

import math
import re

OPERATORS = ("+", "-", "*", "/")
NUMBER_RE = re.compile(r"^number(\d+)$")


def is_operator(tok: str) -> bool:
    return tok in OPERATORS


def number_index(tok: str) -> int | None:
    m = NUMBER_RE.match(tok)
    return int(m.group(1)) if m else None


def is_constant(tok: str) -> bool:
    try:
        float(tok)
    except ValueError:
        return False
    return True


def is_valid_prefix(tokens: list[str]) -> bool:
    """True iff ``tokens`` is exactly one complete binary prefix expression."""
    need = 1
    for tok in tokens:
        if need == 0:
            return False  # trailing tokens after a complete expression
        if is_operator(tok):
            need += 1
        elif number_index(tok) is not None or is_constant(tok):
            need -= 1
        else:
            return False
    return need == 0


def evaluate_prefix(tokens: list[str], numbers: list[float]) -> float | None:
    """Evaluate a prefix expression; returns None for malformed input, missing
    operands, division by zero or non-finite results."""
    if not is_valid_prefix(tokens):
        return None
    stack: list[float] = []
    for tok in reversed(tokens):
        if is_operator(tok):
            a, b = stack.pop(), stack.pop()
            if tok == "+":
                v = a + b
            elif tok == "-":
                v = a - b
            elif tok == "*":
                v = a * b
            else:
                if b == 0:
                    return None
                v = a / b
            stack.append(v)
        else:
            idx = number_index(tok)
            if idx is not None:
                if idx >= len(numbers):
                    return None
                stack.append(numbers[idx])
            else:
                stack.append(float(tok))
    result = stack[0]
    return result if math.isfinite(result) else None


def to_infix(tokens: list[str], numbers: list[float] | None = None) -> str | None:
    """Render a prefix expression as fully parenthesised infix, optionally
    substituting the concrete numbers for ``numberK`` placeholders."""
    if not is_valid_prefix(tokens):
        return None

    def fmt(tok: str) -> str:
        idx = number_index(tok)
        if idx is not None and numbers is not None and idx < len(numbers):
            v = numbers[idx]
            return str(int(v)) if float(v).is_integer() else str(v)
        return tok

    stack: list[str] = []
    for tok in reversed(tokens):
        if is_operator(tok):
            a, b = stack.pop(), stack.pop()
            stack.append(f"({a} {tok} {b})")
        else:
            stack.append(fmt(tok))
    out = stack[0]
    return out[1:-1] if out.startswith("(") and out.endswith(")") else out


def answers_match(pred: float | None, gold: float, rel_tol: float = 1e-4) -> bool:
    if pred is None:
        return False
    return abs(pred - gold) <= rel_tol * max(1.0, abs(gold))
