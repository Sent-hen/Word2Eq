"""Reproducible, preemption-safe training.

* Deterministic: seeded RNGs, deterministic kernels, content-hashed data split,
  batch order derived from ``(seed, epoch)`` -- two runs of one config match.
* Preemption-safe: SIGTERM/SIGINT finish the current step, write an atomic
  checkpoint (including the position inside the epoch and RNG state) and exit
  cleanly; ``--resume`` continues to a bit-identical result.
* Traceable: every run writes a JSONL metric log and an artifact whose model card
  records config fingerprint, input-data SHA-256s and the git revision.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from . import __version__
from .artifact import atomic_torch_save, atomic_write_json, git_revision, save_artifact
from .config import TrainConfig
from .data import Example, data_manifest, load_csv, stable_split
from .evaluate import evaluate
from .expr import OPERATORS
from .model import ModelConfig, Seq2SeqTransformer, count_parameters
from .tensors import bucketed_batches, pad_batch
from .vocab import PAD_ID, Vocab, number_tokens

log = logging.getLogger("word2eq.train")


class StopFlag:
    """Set by SIGTERM/SIGINT; checked between optimizer steps."""

    def __init__(self) -> None:
        self.requested = False

    def install(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(ValueError, OSError):  # not main thread / unsupported platform
                signal.signal(sig, self._handle)

    def _handle(self, signum, _frame) -> None:
        log.warning("signal %s received: checkpointing after current step", signum)
        self.requested = True


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


def build_vocabs(train: list[Example], min_freq: int) -> tuple[Vocab, Vocab]:
    src = Vocab.build((ex.src_tokens for ex in train), min_freq=min_freq, reserved=number_tokens())
    tgt = Vocab.build((ex.equation for ex in train), reserved=[*OPERATORS, *number_tokens()])
    return src, tgt


def lr_lambda(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))  # cosine to 10%

    return f


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return torch.autocast("cpu", enabled=False)


def train(cfg: TrainConfig, resume: bool = False, stop: StopFlag | None = None) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.num_threads:
        torch.set_num_threads(cfg.num_threads)
    seed_everything(cfg.seed)

    all_train = [ex for f in cfg.train_files for ex in load_csv(f)]
    train_set, val_set = stable_split(all_train, cfg.val_fraction)
    if cfg.limit_train:
        train_set = train_set[: cfg.limit_train]
        val_set = val_set[: max(1, cfg.limit_train // 5)]
    test_set = [ex for f in cfg.test_files for ex in load_csv(f)]
    src_vocab, tgt_vocab = build_vocabs(train_set, cfg.src_min_freq)

    src_ids = [src_vocab.encode(ex.src_tokens) for ex in train_set]
    tgt_ids = [tgt_vocab.encode(ex.equation) for ex in train_set]
    lengths = [len(s) for s in src_ids]

    mcfg = ModelConfig(
        src_vocab=len(src_vocab),
        tgt_vocab=len(tgt_vocab),
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        enc_layers=cfg.enc_layers,
        dec_layers=cfg.dec_layers,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
    )
    model = Seq2SeqTransformer(mcfg).to(device)
    steps_per_epoch = math.ceil(len(train_set) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg.warmup_steps, total_steps))
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=cfg.label_smoothing)
    use_scaler = cfg.amp and device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    state = {"epoch": 0, "batch_in_epoch": 0, "step": 0, "best_val": -1.0, "bad_epochs": 0,
             "ep_loss": 0.0, "ep_tokens": 0}
    ckpt_path = out / "checkpoint.pt"
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        if ckpt["fingerprint"] != cfg.fingerprint():
            raise RuntimeError("checkpoint was produced by a different config; refusing to resume")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        torch.set_rng_state(ckpt["rng_cpu"])
        if device.type == "cuda" and ckpt.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(ckpt["rng_cuda"])
        state = ckpt["state"]
        log.info("resumed from epoch %d batch %d (step %d)", state["epoch"], state["batch_in_epoch"],
                 state["step"])

    def save_checkpoint() -> None:
        atomic_torch_save(
            {
                "fingerprint": cfg.fingerprint(),
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None,
                "state": state,
            },
            ckpt_path,
        )

    log.info("device=%s params=%s train=%d val=%d test=%d src_vocab=%d tgt_vocab=%d", device,
             f"{count_parameters(model):,}", len(train_set), len(val_set), len(test_set),
             len(src_vocab), len(tgt_vocab))
    metrics_log = (out / "metrics.jsonl").open("a")
    stop = stop or StopFlag()
    t_start = time.perf_counter()
    interrupted = False

    while state["epoch"] < cfg.epochs:
        epoch = state["epoch"]
        gen = torch.Generator().manual_seed(cfg.seed * 1000 + epoch)
        batches = bucketed_batches(lengths, cfg.batch_size, gen)
        model.train()
        t0 = time.perf_counter()
        for bi in range(state["batch_in_epoch"], len(batches)):
            idx = batches[bi]
            src = pad_batch([src_ids[i] for i in idx], mcfg.max_len).to(device)
            tgt = pad_batch([tgt_ids[i] for i in idx]).to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            with _autocast(device, cfg.amp):
                logits = model(src, tgt_in)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)).float(), tgt_out.reshape(-1))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ntok = int(tgt_out.ne(PAD_ID).sum())
            state["ep_loss"] += loss.item() * ntok
            state["ep_tokens"] += ntok
            state["step"] += 1
            state["batch_in_epoch"] = bi + 1
            if stop.requested:
                interrupted = True
                break
        if interrupted:
            save_checkpoint()
            log.warning("checkpoint written at epoch %d batch %d; exiting", epoch, state["batch_in_epoch"])
            break

        val = evaluate(model, src_vocab, tgt_vocab, val_set, constrained=True, device=device,
                       max_eq_len=cfg.max_eq_len)
        record = {
            "epoch": epoch,
            "step": state["step"],
            "train_loss": state["ep_loss"] / max(1, state["ep_tokens"]),
            "lr": sched.get_last_lr()[0],
            "epoch_sec": round(time.perf_counter() - t0, 3),
            "val_answer_acc": val["answer_acc"],
            "val_equation_acc": val["equation_acc"],
        }
        metrics_log.write(json.dumps(record) + "\n")
        metrics_log.flush()
        log.info("epoch %d loss %.4f val_ans %.4f val_eq %.4f (%.1fs)", epoch, record["train_loss"],
                 val["answer_acc"], val["equation_acc"], record["epoch_sec"])

        if val["answer_acc"] > state["best_val"]:
            state["best_val"] = val["answer_acc"]
            state["bad_epochs"] = 0
            atomic_torch_save(model.state_dict(), out / "best.pt")
        else:
            state["bad_epochs"] += 1
        state["epoch"] += 1
        state["batch_in_epoch"] = 0
        state["ep_loss"], state["ep_tokens"] = 0.0, 0
        save_checkpoint()
        if state["bad_epochs"] >= cfg.patience:
            log.info("early stopping: no val improvement for %d epochs", cfg.patience)
            state["epoch"] = cfg.epochs
            save_checkpoint()
            break

    metrics_log.close()
    if interrupted:
        return {"status": "interrupted", "state": state}

    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True))
    results = {"status": "completed", "best_val_answer_acc": state["best_val"], "test": {}}
    for constrained in (True, False):
        key = "constrained" if constrained else "unconstrained"
        results["test"][key] = evaluate(model, src_vocab, tgt_vocab, test_set, constrained=constrained,
                                        device=device, max_eq_len=cfg.max_eq_len)
    results["train_wall_sec"] = round(time.perf_counter() - t_start, 1)

    created = datetime.now(timezone.utc)
    card = {
        "version": f"{created:%Y%m%d%H%M%S}-{cfg.fingerprint()[:8]}",
        "created_at": created.isoformat(),
        "word2eq_version": __version__,
        "torch_version": torch.__version__,
        "config_fingerprint": cfg.fingerprint(),
        "train_config": cfg.to_dict(),
        "git": git_revision(),
        "data": data_manifest([*cfg.train_files, *cfg.test_files]),
        "parameters": count_parameters(model),
        "metrics": results,
        "max_eq_len": cfg.max_eq_len,
    }
    save_artifact(out / "artifact", model.cpu(), src_vocab, tgt_vocab, card)
    atomic_write_json(results, out / "metrics.json")
    log.info("test (constrained): %s", results["test"]["constrained"])
    return results
