"""Async dynamic micro-batcher with admission control.

Requests are queued and coalesced into one forward pass when either
``max_batch`` items are waiting or the oldest has waited ``max_wait_ms``. That
trades a bounded amount of latency for much higher throughput under load, and
degrades to batch size 1 when idle.

Reliability properties:

* **Load shedding** -- the queue is bounded; when full, ``submit`` fails fast
  with :class:`Overloaded` (-> HTTP 503 + Retry-After) instead of letting
  latency grow without bound.
* **Deadline propagation** -- each item carries a deadline; items that expired
  while queued are dropped *before* inference so no compute is spent on
  requests the client has already abandoned.
* **Fault isolation** -- an exception in one batch fails only that batch's
  futures; the worker loop keeps serving.
* **Graceful drain** -- ``stop()`` stops admission, lets in-flight and queued
  work finish (bounded by a timeout), then exits.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any


class Overloaded(Exception):
    pass


class DeadlineExceeded(Exception):
    pass


class ShuttingDown(Exception):
    pass


@dataclass
class _Item:
    payload: Any
    deadline: float
    future: asyncio.Future
    enqueued: float = field(default_factory=time.monotonic)


class Observer:
    """Metric hooks; the default is a no-op so the batcher has no hard
    dependency on a metrics library."""

    def queue_depth(self, depth: int) -> None: ...
    def batch(self, size: int, seconds: float) -> None: ...
    def shed(self, reason: str) -> None: ...
    def queue_wait(self, seconds: float) -> None: ...


class MicroBatcher:
    def __init__(
        self,
        fn: Callable[[list[Any]], Sequence[Any]],
        max_batch: int = 32,
        max_wait_ms: float = 5.0,
        max_queue: int = 256,
        observer: Observer | None = None,
    ):
        self.fn = fn
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000.0
        self.max_queue = max_queue
        self.obs = observer or Observer()
        self._queue: asyncio.Queue[_Item] | None = None
        self._task: asyncio.Task | None = None
        # One thread: the model is the serialisation point; torch parallelises within the op.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="word2eq-infer")
        self._accepting = False

    @property
    def depth(self) -> int:
        return self._queue.qsize() if self._queue else 0

    async def start(self) -> None:
        self._queue = asyncio.Queue()
        self._accepting = True
        self._task = asyncio.create_task(self._run(), name="word2eq-batcher")

    async def stop(self, drain_timeout: float = 10.0) -> None:
        self._accepting = False
        if self._task is None:
            return
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._drained(), timeout=drain_timeout)
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        while self._queue and not self._queue.empty():
            item = self._queue.get_nowait()
            if not item.future.done():
                item.future.set_exception(ShuttingDown())
        self._pool.shutdown(wait=True)

    async def _drained(self) -> None:
        assert self._queue is not None
        await self._queue.join()

    async def submit(self, payload: Any, timeout: float) -> Any:
        if not self._accepting or self._queue is None:
            self.obs.shed("shutting_down")
            raise ShuttingDown()
        if self._queue.qsize() >= self.max_queue:
            self.obs.shed("queue_full")
            raise Overloaded()
        fut = asyncio.get_running_loop().create_future()
        self._queue.put_nowait(_Item(payload, time.monotonic() + timeout, fut))
        self.obs.queue_depth(self._queue.qsize())
        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=timeout)
        except asyncio.TimeoutError:
            fut.cancel()  # lets the worker skip it if it is still queued
            raise DeadlineExceeded() from None

    async def _collect(self) -> list[_Item]:
        assert self._queue is not None
        first = await self._queue.get()
        batch = [first]
        window_end = time.monotonic() + self.max_wait
        while len(batch) < self.max_batch:
            remaining = window_end - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break
        return batch

    async def _run(self) -> None:
        assert self._queue is not None
        loop = asyncio.get_running_loop()
        while True:
            batch = await self._collect()
            try:
                now = time.monotonic()
                live = []
                for it in batch:
                    if it.future.done():  # caller already timed out / cancelled
                        continue
                    if it.deadline <= now:
                        self.obs.shed("deadline_expired")
                        it.future.set_exception(DeadlineExceeded())
                        continue
                    self.obs.queue_wait(now - it.enqueued)
                    live.append(it)
                self.obs.queue_depth(self._queue.qsize())
                if not live:
                    continue
                t0 = time.perf_counter()
                try:
                    results = await loop.run_in_executor(self._pool, self.fn, [it.payload for it in live])
                except Exception as exc:  # fault isolation: fail this batch only
                    for it in live:
                        if not it.future.done():
                            it.future.set_exception(exc)
                    continue
                self.obs.batch(len(live), time.perf_counter() - t0)
                for it, res in zip(live, results, strict=True):
                    if not it.future.done():
                        it.future.set_result(res)
            finally:
                for _ in batch:
                    self._queue.task_done()
