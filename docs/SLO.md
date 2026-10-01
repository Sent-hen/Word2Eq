# Word2Eq service level objectives

## SLIs and SLOs (30-day window)

| SLI | Definition | SLO |
|---|---|---|
| Availability | `ok` / all requests except `invalid` (422s are client errors) | 99.9% |
| Latency | p99 of `word2eq_request_latency_seconds` | < 250 ms |
| Answer integrity | share of `ok` responses with `verified=true` | ≥ 98% |

Alerting uses multi-window burn rates; see [`deploy/prometheus/alerts.yml`](../deploy/prometheus/alerts.yml)
and the [runbook](RUNBOOK.md).

## Capacity

Measured on one laptop (Windows 11, 22 logical cores, CPU only). The base model has 6.4M parameters
and serving ran in a single process with 4 torch threads. The load generator was open-loop,
running on the same host.

| Offered load | Achieved | p50 | p95 | p99 |
|---|---|---|---|---|
| 50 rps | 50 rps | 33 ms | 57 ms | 329 ms |
| 150 rps | 150 rps | 46 ms | 73 ms | 336 ms |
| 300 rps | ~80 rps | saturated (seconds) | | |

Offline, with no HTTP layer (`word2eq bench`, 4 threads):

| Batch size | fp32 throughput | int8 throughput |
|---|---|---|
| 1 | 93 /s | 70 /s |
| 8 | 279 /s | 271 /s |
| 32 | 402 /s | 423 /s |

Dynamic int8 cuts the model from 25.8 MB to 16.3 MB but isn't faster on this CPU, so it is off by
default (`WORD2EQ_QUANTIZE=0`).

**Open issues**

* **Saturation point.** Above about 150 rps, client-observed latency grows without bound, while
  server-side request latency stays around 40–60 ms. Throughput stays near 80 ok/s at 300 rps with
  either 1 or 4 workers, so the open-loop generator (one Python/httpx process) is at least part of
  the limit. GIL contention between the decode loop and the event loop is the other suspect. This
  needs re-measuring on Linux with a separate load-generator host before setting HPA targets.
* **p99 of about 330 ms at low load** is above the 250 ms target. Profile it (GC, allocator, or
  thread oversubscription) before treating the latency SLO as met.
* **Multi-worker serving** (`word2eq serve --workers N`) is meant for Linux containers. Uvicorn's
  multi-process mode fails on Windows with WinError 10022.

Until those are resolved, size for **≤ 150 rps per replica**.
