"""HTTP inference service.

    POST /v1/solve        {"problem": "..."}      -> verified equation + answer
    POST /v1/solve/batch  {"problems": [...]}     -> list of results
    GET  /v1/model                                -> model card summary
    GET  /healthz                                 -> liveness (process is up)
    GET  /readyz                                  -> readiness (model loaded + warmed)
    GET  /metrics                                 -> Prometheus exposition
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from .batcher import DeadlineExceeded, MicroBatcher, Overloaded, ShuttingDown
from .metrics import Metrics
from .runner import InvalidProblem, ModelRunner, prepare

log = logging.getLogger("word2eq.serve")


@dataclass
class Settings:
    artifact_dir: str = "runs/default/artifact"
    max_batch: int = 32
    max_wait_ms: float = 5.0
    max_queue: int = 256
    max_inflight: int = 512  # edge admission cap across all handlers in this process
    request_timeout_s: float = 2.0
    quantize: bool = False
    num_threads: int | None = None
    drain_timeout_s: float = 10.0

    @classmethod
    def from_env(cls) -> Settings:
        e = os.environ.get
        threads = e("WORD2EQ_THREADS")
        return cls(
            artifact_dir=e("WORD2EQ_ARTIFACT", cls.artifact_dir),
            max_batch=int(e("WORD2EQ_MAX_BATCH", cls.max_batch)),
            max_wait_ms=float(e("WORD2EQ_MAX_WAIT_MS", cls.max_wait_ms)),
            max_queue=int(e("WORD2EQ_MAX_QUEUE", cls.max_queue)),
            max_inflight=int(e("WORD2EQ_MAX_INFLIGHT", cls.max_inflight)),
            request_timeout_s=float(e("WORD2EQ_TIMEOUT_S", cls.request_timeout_s)),
            quantize=e("WORD2EQ_QUANTIZE", "0").lower() in ("1", "true", "yes"),
            num_threads=int(threads) if threads else None,
            drain_timeout_s=float(e("WORD2EQ_DRAIN_TIMEOUT_S", cls.drain_timeout_s)),
        )


class SolveRequest(BaseModel):
    problem: str = Field(..., min_length=1, max_length=2000)


class BatchSolveRequest(BaseModel):
    problems: list[str] = Field(..., min_length=1, max_length=64)


class SolveResponse(BaseModel):
    equation: str | None
    equation_prefix: list[str]
    answer: float | None
    numbers: list[float]
    verified: bool
    model_version: str
    latency_ms: float


def create_app(settings: Settings | None = None, runner_factory: Callable[[Settings], object] | None = None):
    settings = settings or Settings.from_env()
    runner_factory = runner_factory or (
        lambda s: ModelRunner(s.artifact_dir, quantize=s.quantize, num_threads=s.num_threads)
    )
    metrics = Metrics()
    state: dict = {"runner": None, "batcher": None, "ready": False, "error": None, "inflight": 0}

    async def load() -> None:
        try:
            runner = await asyncio.to_thread(runner_factory, settings)
            warm = await asyncio.to_thread(runner.warmup)
            batcher = MicroBatcher(runner.solve_batch, settings.max_batch, settings.max_wait_ms,
                                   settings.max_queue, observer=metrics)
            await batcher.start()
            state.update(runner=runner, batcher=batcher, ready=True)
            metrics.ready.set(1)
            metrics.model_info.labels(runner.version, str(getattr(runner, "quantized", False))).set(1)
            log.info("model %s ready (warm latency %.1f ms)", runner.version, warm * 1000)
        except Exception as exc:  # stay alive but unready; /readyz reports why
            state["error"] = repr(exc)
            log.exception("model load failed")

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        loader = asyncio.create_task(load())
        yield
        state["ready"] = False
        metrics.ready.set(0)
        loader.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await loader
        if state["batcher"] is not None:
            await state["batcher"].stop(settings.drain_timeout_s)

    app = FastAPI(title="Word2Eq", version="1.0.0", lifespan=lifespan)
    app.state.metrics = metrics
    app.state.internal = state

    async def solve_one(problem: str) -> SolveResponse:
        # Edge admission control: reject before doing any work once this process
        # holds max_inflight requests, so overload yields fast 503s rather than
        # an unbounded backlog in front of the batcher.
        if state["inflight"] >= settings.max_inflight:
            metrics.shed("inflight_limit")
            metrics.requests.labels("shed").inc()
            raise HTTPException(503, "overloaded", headers={"Retry-After": "1"})
        state["inflight"] += 1
        try:
            return await _solve_admitted(problem)
        finally:
            state["inflight"] -= 1

    async def _solve_admitted(problem: str) -> SolveResponse:
        t0 = time.perf_counter()
        if not state["ready"]:
            metrics.requests.labels("unavailable").inc()
            raise HTTPException(503, "model not ready", headers={"Retry-After": "1"})
        runner = state["runner"]
        try:
            item = prepare(problem, runner.max_src_tokens)
        except InvalidProblem as exc:
            metrics.requests.labels("invalid").inc()
            raise HTTPException(422, str(exc)) from None
        try:
            sol = await state["batcher"].submit(item, settings.request_timeout_s)
        except Overloaded:
            metrics.requests.labels("shed").inc()
            raise HTTPException(503, "overloaded", headers={"Retry-After": "1"}) from None
        except ShuttingDown:
            metrics.requests.labels("unavailable").inc()
            raise HTTPException(503, "shutting down", headers={"Retry-After": "1"}) from None
        except DeadlineExceeded:
            metrics.requests.labels("timeout").inc()
            raise HTTPException(504, "deadline exceeded") from None
        except Exception:
            metrics.requests.labels("error").inc()
            log.exception("inference failed")
            raise HTTPException(500, "inference failed") from None
        elapsed = time.perf_counter() - t0
        metrics.latency.observe(elapsed)
        metrics.requests.labels("ok").inc()
        if not sol.verified:
            metrics.unverified.inc()
        return SolveResponse(
            equation=sol.equation,
            equation_prefix=sol.equation_prefix,
            answer=sol.answer,
            numbers=sol.numbers,
            verified=sol.verified,
            model_version=runner.version,
            latency_ms=round(elapsed * 1000, 3),
        )

    @app.post("/v1/solve", response_model=SolveResponse)
    async def solve(req: SolveRequest) -> SolveResponse:
        return await solve_one(req.problem)

    @app.post("/v1/solve/batch", response_model=list[SolveResponse])
    async def solve_batch(req: BatchSolveRequest) -> list[SolveResponse]:
        return list(await asyncio.gather(*(solve_one(p) for p in req.problems)))

    @app.get("/v1/model")
    async def model_info() -> dict:
        if not state["ready"]:
            raise HTTPException(503, "model not ready")
        card = getattr(state["runner"], "card", {})
        return {
            "version": state["runner"].version,
            "quantized": getattr(state["runner"], "quantized", False),
            "created_at": card.get("created_at"),
            "git": card.get("git"),
            "config_fingerprint": card.get("config_fingerprint"),
            "metrics": card.get("metrics", {}).get("test"),
        }

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(response: Response) -> dict:
        if state["ready"]:
            return {"status": "ready", "queue_depth": state["batcher"].depth}
        response.status_code = 503
        return {"status": "loading" if state["error"] is None else "failed", "error": state["error"]}

    @app.get("/metrics")
    async def prom() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app
