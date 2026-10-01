"""Encoder-decoder Transformer built on PyTorch's fused ``nn.Transformer*`` layers.

Replaces the original hand-rolled Annotated-Transformer port: same architecture
family, but batch-first, pre-LayerNorm (stable without long warmup), fused SDPA
attention kernels, and tied decoder input/output embeddings.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .vocab import PAD_ID


@dataclass
class ModelConfig:
    src_vocab: int
    tgt_vocab: int
    d_model: int = 256
    nhead: int = 8
    enc_layers: int = 3
    dec_layers: int = 3
    ff_dim: int = 1024
    dropout: float = 0.1
    max_len: int = 256

    def to_dict(self) -> dict:
        return asdict(self)


class SinusoidalPositions(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class Seq2SeqTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.scale = math.sqrt(cfg.d_model)
        self.src_emb = nn.Embedding(cfg.src_vocab, cfg.d_model, padding_idx=PAD_ID)
        self.tgt_emb = nn.Embedding(cfg.tgt_vocab, cfg.d_model, padding_idx=PAD_ID)
        self.pos = SinusoidalPositions(cfg.d_model, cfg.max_len)
        self.drop = nn.Dropout(cfg.dropout)
        enc_layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.nhead, cfg.ff_dim, cfg.dropout, batch_first=True, norm_first=True
        )
        dec_layer = nn.TransformerDecoderLayer(
            cfg.d_model, cfg.nhead, cfg.ff_dim, cfg.dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, cfg.enc_layers, norm=nn.LayerNorm(cfg.d_model), enable_nested_tensor=False
        )
        self.decoder = nn.TransformerDecoder(dec_layer, cfg.dec_layers, norm=nn.LayerNorm(cfg.d_model))
        self.out = nn.Linear(cfg.d_model, cfg.tgt_vocab, bias=False)
        self.out.weight = self.tgt_emb.weight  # weight tying
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for name, p in self.named_parameters():
            if p.dim() > 1 and "emb" not in name:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.src_emb.weight, std=self.cfg.d_model**-0.5)
        nn.init.normal_(self.tgt_emb.weight, std=self.cfg.d_model**-0.5)
        with torch.no_grad():
            self.src_emb.weight[PAD_ID].zero_()
            self.tgt_emb.weight[PAD_ID].zero_()

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src_pad = src.eq(PAD_ID)
        x = self.drop(self.pos(self.src_emb(src) * self.scale))
        return self.encoder(x, src_key_padding_mask=src_pad), src_pad

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, src_pad: torch.Tensor) -> torch.Tensor:
        t = tgt.size(1)
        causal = torch.ones(t, t, dtype=torch.bool, device=tgt.device).triu_(1)
        y = self.drop(self.pos(self.tgt_emb(tgt) * self.scale))
        h = self.decoder(
            y,
            memory,
            tgt_mask=causal,
            tgt_is_causal=True,
            tgt_key_padding_mask=tgt.eq(PAD_ID),
            memory_key_padding_mask=src_pad,
        )
        return self.out(h)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        memory, src_pad = self.encode(src)
        return self.decode(tgt_in, memory, src_pad)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
