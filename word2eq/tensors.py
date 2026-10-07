"""Turning examples into padded tensors, with length bucketing to cut padding."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .vocab import PAD_ID, Vocab


def pad_batch(seqs: Sequence[Sequence[int]], max_len: int | None = None) -> torch.Tensor:
    width = max(len(s) for s in seqs)
    if max_len is not None:
        width = min(width, max_len)
    out = torch.full((len(seqs), width), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        s = list(s)[:width]
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def encode_sources(token_lists: Sequence[list[str]], vocab: Vocab, max_len: int) -> torch.Tensor:
    return pad_batch([vocab.encode(t) for t in token_lists], max_len=max_len)


def bucketed_batches(
    lengths: Sequence[int], batch_size: int, generator: torch.Generator | None, bucket_mult: int = 50
) -> list[list[int]]:
    """Shuffle, then sort by length within large chunks and cut into batches.

    Keeps batches length-homogeneous (less padding, less wasted attention
    compute) while preserving randomness across batches. Fully determined by
    ``generator``, so resumed runs replay the exact same batch order."""
    n = len(lengths)
    if generator is None:
        order = sorted(range(n), key=lambda i: lengths[i])
        return [order[i : i + batch_size] for i in range(0, n, batch_size)]
    perm = torch.randperm(n, generator=generator).tolist()
    chunk = batch_size * bucket_mult
    batches = []
    for start in range(0, n, chunk):
        part = sorted(perm[start : start + chunk], key=lambda i: lengths[i])
        batches.extend(part[i : i + batch_size] for i in range(0, len(part), batch_size))
    idx = torch.randperm(len(batches), generator=generator).tolist()
    return [batches[i] for i in idx]
