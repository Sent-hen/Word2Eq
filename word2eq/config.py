"""Run configuration: a single YAML file fully determines a training run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    train_files: list[str] = field(default_factory=lambda: ["data/mawps-asdiv-a_svamp/train.csv"])
    test_files: list[str] = field(default_factory=lambda: ["data/mawps-asdiv-a_svamp/dev.csv"])
    val_fraction: float = 0.1
    out_dir: str = "runs/default"
    seed: int = 1234

    # model
    d_model: int = 256
    nhead: int = 8
    enc_layers: int = 3
    dec_layers: int = 3
    ff_dim: int = 1024
    dropout: float = 0.2
    src_min_freq: int = 1
    max_eq_len: int = 32

    # optimisation
    epochs: int = 60
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 400
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    amp: bool = True  # bf16/fp16 autocast when a CUDA device is present
    patience: int = 15  # early stopping on validation answer accuracy
    limit_train: int | None = None  # cap examples (smoke tests)
    num_threads: int | None = None

    @classmethod
    def load(cls, path: str | Path, **overrides) -> TrainConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        raw.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        """Hash of everything that affects the trained weights (not out_dir)."""
        d = self.to_dict()
        d.pop("out_dir")
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
