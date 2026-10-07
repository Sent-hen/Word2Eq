import asyncio
import threading
import time

import pytest

from word2eq.serving.batcher import DeadlineExceeded, MicroBatcher, Overloaded, ShuttingDown


def run(coro):
    return asyncio.run(coro)


def test_concurrent_requests_are_coalesced_into_batches():
    sizes = []

    def fn(xs):
        sizes.append(len(xs))
        time.sleep(0.01)
        return [x * 2 for x in xs]

    async def main():
        b = MicroBatcher(fn, max_batch=16, max_wait_ms=20)
        await b.start()
        out = await asyncio.gather(*(b.submit(i, timeout=5) for i in range(40)))
        await b.stop()
        return out

    assert run(main()) == [i * 2 for i in range(40)]
    assert max(sizes) > 1 and max(sizes) <= 16
    assert len(sizes) < 40


def test_full_queue_sheds_load_instead_of_queueing_forever():
    gate = threading.Event()

    def fn(xs):
        gate.wait(5)
        return xs

    async def main():
        b = MicroBatcher(fn, max_batch=1, max_wait_ms=0, max_queue=2)
        await b.start()
        tasks = [asyncio.create_task(b.submit(0, timeout=5))]
        await asyncio.sleep(0.05)  # item 0 is now held by the (blocked) worker
        tasks += [asyncio.create_task(b.submit(i, timeout=5)) for i in (1, 2)]
        await asyncio.sleep(0.01)  # items 1 and 2 fill the queue
        with pytest.raises(Overloaded):
            await b.submit(99, timeout=5)
        gate.set()
        res = await asyncio.gather(*tasks)
        await b.stop()
        return res

    assert run(main()) == [0, 1, 2]


def test_expired_requests_are_dropped_before_inference():
    seen = []
    gate = threading.Event()

    def fn(xs):
        seen.extend(xs)
        gate.wait(5)
        return xs

    async def main():
        b = MicroBatcher(fn, max_batch=1, max_wait_ms=0)
        await b.start()
        blocker = asyncio.create_task(b.submit("blocker", timeout=5))
        await asyncio.sleep(0.02)
        with pytest.raises(DeadlineExceeded):
            await b.submit("stale", timeout=0.05)  # expires while blocker holds the worker
        gate.set()
        await blocker
        await asyncio.sleep(0.05)
        await b.stop()

    run(main())
    assert "stale" not in seen


def test_failing_batch_does_not_kill_the_worker():
    calls = {"n": 0}

    def fn(xs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return xs

    async def main():
        b = MicroBatcher(fn, max_batch=1, max_wait_ms=0)
        await b.start()
        with pytest.raises(RuntimeError):
            await b.submit(1, timeout=2)
        ok = await b.submit(2, timeout=2)
        await b.stop()
        return ok

    assert run(main()) == 2


def test_stop_drains_queued_work_then_rejects_new_requests():
    def fn(xs):
        time.sleep(0.02)
        return xs

    async def main():
        b = MicroBatcher(fn, max_batch=2, max_wait_ms=1)
        await b.start()
        tasks = [asyncio.create_task(b.submit(i, timeout=5)) for i in range(6)]
        await asyncio.sleep(0)
        await b.stop(drain_timeout=5)
        with pytest.raises(ShuttingDown):
            await b.submit(7, timeout=1)
        return await asyncio.gather(*tasks)

    assert run(main()) == list(range(6))
