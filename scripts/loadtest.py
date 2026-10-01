"""Open-loop HTTP load generator for the Word2Eq service.

Requests are fired on a fixed schedule (``--rps``) regardless of whether earlier
ones have completed, and latency is measured from the *scheduled* send time.
That avoids coordinated omission -- a closed-loop client quietly slows down when
the server does, which hides exactly the tail latency an SLO cares about.

    python scripts/loadtest.py --url http://localhost:8000 --rps 200 --duration 30
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import time
from collections import Counter

import httpx


def load_problems(path: str, limit: int = 1000) -> list[str]:
    """Rebuild raw text from the masked dataset (numberK -> its real value)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            nums = row["Numbers"].split()
            text = re.sub(r"number(\d+)", lambda m, nums=nums: nums[int(m.group(1))], row["Question"])
            out.append(text)
            if len(out) >= limit:
                break
    return out


def pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    return sorted_vals[min(len(sorted_vals) - 1, int(p / 100 * len(sorted_vals)))]


async def run(url: str, rps: float, duration: float, problems: list[str], timeout: float) -> dict:
    latencies: list[float] = []
    status: Counter = Counter()
    total = int(rps * duration)
    limits = httpx.Limits(max_connections=1000, max_keepalive_connections=1000)
    async with httpx.AsyncClient(base_url=url, timeout=timeout, limits=limits) as client:
        start = time.perf_counter() + 0.1

        async def one(i: int) -> None:
            scheduled = start + i / rps
            delay = scheduled - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                r = await client.post("/v1/solve", json={"problem": problems[i % len(problems)]})
                status[r.status_code] += 1
                if r.status_code == 200:
                    latencies.append(time.perf_counter() - scheduled)
            except httpx.HTTPError as e:
                status[type(e).__name__] += 1

        await asyncio.gather(*(one(i) for i in range(total)))
        wall = time.perf_counter() - start
    latencies.sort()
    ok = status.get(200, 0)
    return {
        "target_rps": rps,
        "duration_s": duration,
        "requests": total,
        "achieved_ok_rps": round(ok / wall, 1),
        "success_rate": round(ok / total, 4) if total else 0.0,
        "status": {str(k): v for k, v in status.items()},
        "p50_ms": round(pct(latencies, 50) * 1000, 2),
        "p95_ms": round(pct(latencies, 95) * 1000, 2),
        "p99_ms": round(pct(latencies, 99) * 1000, 2),
        "max_ms": round(latencies[-1] * 1000, 2) if latencies else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--rps", type=float, nargs="+", default=[50.0])
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--data", default="data/mawps-asdiv-a_svamp/dev.csv")
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--out")
    a = p.parse_args()
    problems = load_problems(a.data)
    results = []
    for rps in a.rps:
        res = asyncio.run(run(a.url, rps, a.duration, problems, a.timeout))
        print(json.dumps(res), flush=True)
        results.append(res)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
