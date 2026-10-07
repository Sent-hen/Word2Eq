"""Batched greedy decoding with an optional prefix-grammar constraint.

The constraint is a tiny pushdown automaton tracked per sequence as ``need`` (the
number of operands still required to close the expression):

* start with ``need = 1``; an operator adds one, an operand removes one;
* ``</s>`` is legal only when ``need == 0``, and is then the *only* legal token;
* an operator is legal only if the remaining length budget can still close it;
* ``numberK`` is legal only if the problem actually contains K+1 numbers.

Illegal tokens get ``-inf`` logits before the argmax, so every decoded output is
a well-formed expression that references real operands -- by construction, not
by post-hoc filtering. Masks are built with vectorised tensor ops, so the
constraint adds O(batch x vocab) work per step and no Python loop over the batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .expr import is_constant, is_operator, number_index
from .model import Seq2SeqTransformer
from .vocab import BOS_ID, EOS_ID, PAD_ID, Vocab


@dataclass
class GrammarTables:
    is_op: torch.Tensor  # [V] bool
    is_const: torch.Tensor  # [V] bool
    num_idx: torch.Tensor  # [V] long, -1 for non-number tokens

    @classmethod
    def from_vocab(cls, vocab: Vocab, device: torch.device | str = "cpu") -> GrammarTables:
        v = len(vocab)
        is_op = torch.zeros(v, dtype=torch.bool)
        is_const = torch.zeros(v, dtype=torch.bool)
        num_idx = torch.full((v,), -1, dtype=torch.long)
        for i, tok in enumerate(vocab.itos):
            if tok.startswith("<"):
                continue
            if is_operator(tok):
                is_op[i] = True
            elif (k := number_index(tok)) is not None:
                num_idx[i] = k
            elif is_constant(tok):
                is_const[i] = True
        return cls(is_op.to(device), is_const.to(device), num_idx.to(device))


def _allowed_mask(
    g: GrammarTables, need: torch.Tensor, length: torch.Tensor, n_nums: torch.Tensor, max_len: int
) -> torch.Tensor:
    """[B, V] bool mask of legal next tokens."""
    operand = g.is_const.unsqueeze(0) | (
        (g.num_idx.unsqueeze(0) >= 0) & (g.num_idx.unsqueeze(0) < n_nums.unsqueeze(1))
    )
    open_ = (need > 0).unsqueeze(1)
    # After an operator: length+1 tokens used, need+1 operands still owed.
    op_ok = (length + need + 2 <= max_len).unsqueeze(1)
    allowed = open_ & (operand | (g.is_op.unsqueeze(0) & op_ok))
    eos = torch.zeros_like(allowed)
    eos[:, EOS_ID] = True
    return torch.where(open_, allowed, eos)


@torch.inference_mode()
def greedy_decode(
    model: Seq2SeqTransformer,
    src: torch.Tensor,
    n_nums: torch.Tensor,
    grammar: GrammarTables | None,
    max_len: int = 32,
) -> torch.Tensor:
    """Decode a batch. Returns [B, <=max_len] token ids (without BOS), PAD after EOS.

    ``grammar=None`` gives plain unconstrained greedy decoding (used as the
    baseline in evaluation)."""
    device = src.device
    b = src.size(0)
    memory, src_pad = model.encode(src)
    ys = torch.full((b, 1), BOS_ID, dtype=torch.long, device=device)
    need = torch.ones(b, dtype=torch.long, device=device)
    length = torch.zeros(b, dtype=torch.long, device=device)
    done = torch.zeros(b, dtype=torch.bool, device=device)

    for _ in range(max_len + 1):
        logits = model.decode(ys, memory, src_pad)[:, -1]
        logits[:, PAD_ID] = float("-inf")
        logits[:, BOS_ID] = float("-inf")
        if grammar is not None:
            logits = logits.masked_fill(~_allowed_mask(grammar, need, length, n_nums, max_len), float("-inf"))
        nxt = logits.argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, PAD_ID), nxt)
        ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)

        if grammar is not None:
            live = ~done & nxt.ne(EOS_ID)
            is_op = grammar.is_op[nxt]
            need = need + torch.where(live, torch.where(is_op, 1, -1), 0)
        length = length + (~done).long()
        done = done | nxt.eq(EOS_ID)
        if bool(done.all()):
            break
    return ys[:, 1:]
