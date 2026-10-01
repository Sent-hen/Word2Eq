"""Command-line entry point: ``word2eq {train,eval,cv,bench,gate,serve}``."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def cmd_train(a) -> int:
    from .config import TrainConfig
    from .train import StopFlag, train

    cfg = TrainConfig.load(a.config, out_dir=a.out_dir, epochs=a.epochs, seed=a.seed)
    stop = StopFlag()
    stop.install()
    res = train(cfg, resume=a.resume, stop=stop)
    print(json.dumps(res, indent=2))
    return 0 if res["status"] == "completed" else 3


def cmd_eval(a) -> int:
    from .artifact import load_artifact
    from .data import load_csv
    from .evaluate import evaluate

    model, sv, tv, card = load_artifact(a.artifact)
    examples = [ex for f in a.data for ex in load_csv(f)]
    res = {
        "model_version": card.get("version"),
        "constrained": evaluate(model, sv, tv, examples, constrained=True, max_eq_len=card["max_eq_len"]),
        "unconstrained": evaluate(model, sv, tv, examples, constrained=False, max_eq_len=card["max_eq_len"]),
    }
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
    return 0


def cmd_cv(a) -> int:
    """k-fold cross-validation over data/<dataset>/fold*/{train,dev}.csv."""
    from .config import TrainConfig
    from .train import train

    folds = sorted(Path(a.dataset).glob("fold*"))
    if not folds:
        print(f"no folds under {a.dataset}", file=sys.stderr)
        return 2
    base = TrainConfig.load(a.config, epochs=a.epochs)
    accs = []
    for fold in folds:
        cfg = TrainConfig.load(a.config, epochs=a.epochs)
        cfg.train_files = [str(fold / "train.csv")]
        cfg.test_files = [str(fold / "dev.csv")]
        cfg.out_dir = str(Path(base.out_dir) / fold.name)
        res = train(cfg)
        accs.append(res["test"]["constrained"]["answer_acc"])
        print(f"{fold.name}: answer_acc={accs[-1]:.4f}", flush=True)
    summary = {"folds": len(accs), "answer_acc_mean": statistics.mean(accs),
               "answer_acc_stdev": statistics.stdev(accs) if len(accs) > 1 else 0.0, "per_fold": accs}
    print(json.dumps(summary, indent=2))
    Path(base.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(base.out_dir) / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    return 0


def cmd_bench(a) -> int:
    """Offline latency/throughput of the runner: fp32 vs int8 across batch sizes."""
    import torch

    from .data import load_csv
    from .serving.runner import ModelRunner, Prepared

    examples = load_csv(a.data)
    results = []
    for quant in (False, True):
        runner = ModelRunner(a.artifact, quantize=quant, num_threads=a.threads)
        # Datasets are pre-masked, so build Prepared items directly.
        items = [Prepared(ex.src_tokens, list(ex.numbers)) for ex in examples]
        runner.warmup()
        for bs in a.batch_sizes:
            lat = []
            n = 0
            t_all = time.perf_counter()
            for i in range(0, min(len(items), a.max_items), bs):
                chunk = items[i : i + bs]
                t0 = time.perf_counter()
                runner.solve_batch(chunk)
                lat.append((time.perf_counter() - t0) * 1000)
                n += len(chunk)
            wall = time.perf_counter() - t_all
            lat.sort()
            results.append({
                "quantized": quant, "batch_size": bs, "items": n,
                "throughput_per_s": round(n / wall, 1),
                "batch_p50_ms": round(lat[len(lat) // 2], 2),
                "batch_p95_ms": round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 2),
            })
            print(json.dumps(results[-1]), flush=True)
    size = {}
    for quant in (False, True):
        runner = ModelRunner(a.artifact, quantize=quant)
        buf = __import__("io").BytesIO()
        torch.save(runner.model.state_dict(), buf)
        size["int8" if quant else "fp32"] = round(buf.tell() / 1e6, 2)
    out = {"threads": torch.get_num_threads(), "results": results, "model_mb": size}
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({"model_mb": size}))
    return 0


def cmd_gate(a) -> int:
    """Release gate: fail (exit 1) if the candidate regresses vs. the baseline."""
    cand = json.loads(Path(a.candidate).read_text())
    base = json.loads(Path(a.baseline).read_text())

    def pick(d):
        d = d.get("metrics", d)
        return d["test"]["constrained"] if "test" in d else d["constrained"]

    c, b = pick(cand), pick(base)
    failures = []
    if c["valid_rate"] < 1.0:
        failures.append(f"valid_rate {c['valid_rate']:.4f} < 1.0 (grammar constraint broken)")
    drop = b["answer_acc"] - c["answer_acc"]
    if drop > a.max_drop:
        failures.append(f"answer_acc dropped {drop:.4f} > {a.max_drop} ({b['answer_acc']:.4f} -> "
                        f"{c['answer_acc']:.4f})")
    report = {"baseline_answer_acc": b["answer_acc"], "candidate_answer_acc": c["answer_acc"],
              "candidate_valid_rate": c["valid_rate"], "passed": not failures, "failures": failures}
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


def cmd_serve(a) -> int:
    import os

    import uvicorn

    # Settings travel via env so each worker process builds its own app + model.
    if a.artifact:
        os.environ["WORD2EQ_ARTIFACT"] = a.artifact
    if a.quantize:
        os.environ["WORD2EQ_QUANTIZE"] = "1"
    # Separate processes: decode is a Python loop that holds the GIL between ops,
    # so one process can't run inference and its event loop at full speed.
    uvicorn.run("word2eq.serving.app:create_app", factory=True, host=a.host, port=a.port,
                workers=a.workers, log_level="info", access_log=False, timeout_graceful_shutdown=15)
    return 0


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    p = argparse.ArgumentParser(prog="word2eq")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train a model from a YAML config")
    t.add_argument("--config", default="configs/base.yaml")
    t.add_argument("--out-dir")
    t.add_argument("--epochs", type=int)
    t.add_argument("--seed", type=int)
    t.add_argument("--resume", action="store_true", help="continue from out_dir/checkpoint.pt")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval", help="evaluate an artifact on CSV data")
    e.add_argument("--artifact", required=True)
    e.add_argument("--data", nargs="+", required=True)
    e.add_argument("--out")
    e.set_defaults(fn=cmd_eval)

    c = sub.add_parser("cv", help="k-fold cross-validation")
    c.add_argument("--dataset", default="data/cv_asdiv-a")
    c.add_argument("--config", default="configs/base.yaml")
    c.add_argument("--epochs", type=int)
    c.set_defaults(fn=cmd_cv)

    b = sub.add_parser("bench", help="offline latency / throughput benchmark")
    b.add_argument("--artifact", required=True)
    b.add_argument("--data", default="data/mawps-asdiv-a_svamp/dev.csv")
    b.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    b.add_argument("--max-items", type=int, default=512)
    b.add_argument("--threads", type=int)
    b.add_argument("--out")
    b.set_defaults(fn=cmd_bench)

    g = sub.add_parser("gate", help="regression gate for model promotion")
    g.add_argument("--candidate", required=True)
    g.add_argument("--baseline", required=True)
    g.add_argument("--max-drop", type=float, default=0.01)
    g.set_defaults(fn=cmd_gate)

    s = sub.add_parser("serve", help="run the HTTP inference service")
    s.add_argument("--artifact")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--quantize", action="store_true")
    s.add_argument("--workers", type=int, default=1, help="worker processes, each with its own model")
    s.set_defaults(fn=cmd_serve)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
