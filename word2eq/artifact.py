"""Versioned model artifacts: weights + vocab + model card, written atomically.

An artifact directory is the unit of deployment::

    artifact/
      model.pt          # state_dict only, loaded with weights_only=True
      vocab.json        # plain JSON vocabularies
      model_card.json   # config, data hashes, git SHA, metrics, version id
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import torch

from .model import ModelConfig, Seq2SeqTransformer
from .vocab import Vocab, load_vocabs, save_vocabs


def atomic_write_bytes(path: str | Path, write_fn) -> None:
    """Write via a temp file in the same directory + os.replace, so a crash or
    preemption mid-write never leaves a truncated file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            write_fn(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_torch_save(obj, path: str | Path) -> None:
    atomic_write_bytes(path, lambda f: torch.save(obj, f))


def atomic_write_json(obj, path: str | Path) -> None:
    atomic_write_bytes(path, lambda f: f.write(json.dumps(obj, indent=2, sort_keys=True).encode()))


def git_revision() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"sha": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"sha": "unknown", "dirty": None}


def save_artifact(
    out_dir: str | Path, model: Seq2SeqTransformer, src_vocab: Vocab, tgt_vocab: Vocab, card: dict
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_torch_save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out_dir / "model.pt")
    save_vocabs(out_dir / "vocab.json", src_vocab, tgt_vocab)
    atomic_write_json({**card, "model_config": model.cfg.to_dict()}, out_dir / "model_card.json")


def load_artifact(
    art_dir: str | Path, device: torch.device | str = "cpu"
) -> tuple[Seq2SeqTransformer, Vocab, Vocab, dict]:
    art_dir = Path(art_dir)
    card = json.loads((art_dir / "model_card.json").read_text())
    src_vocab, tgt_vocab = load_vocabs(art_dir / "vocab.json")
    model = Seq2SeqTransformer(ModelConfig(**card["model_config"]))
    state = torch.load(art_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, src_vocab, tgt_vocab, card
