# Word2Eq on-call runbook

Every alert in [`deploy/prometheus/alerts.yml`](../deploy/prometheus/alerts.yml) links to a section here.
Metric names are defined in [`word2eq/serving/metrics.py`](../word2eq/serving/metrics.py).

## Triage in 60 seconds

```promql
sum by (outcome) (rate(word2eq_requests_total[5m]))          # what is failing?
histogram_quantile(0.99, sum by (le) (rate(word2eq_request_latency_seconds_bucket[5m])))
histogram_quantile(0.99, sum by (le) (rate(word2eq_queue_wait_seconds_bucket[5m])))   # queueing vs compute
histogram_quantile(0.50, sum by (le) (rate(word2eq_batch_size_bucket[5m])))
max(word2eq_queue_depth)
sum by (version) (word2eq_model_info)                          # mixed versions mid-rollout?
```

Request outcomes: `ok`, `invalid` (422, client error, excluded from the SLO), `shed` (503 queue full),
`timeout` (504 deadline), `unavailable` (503 not ready / draining), `error` (500).

## Error budget burn

**Signal:** `Word2EqErrorBudgetFastBurn` (page) / `SlowBurn` (ticket).

1. Break down by `outcome` (query above).
   * Mostly `shed` → go to [Load shedding](#load-shedding).
   * Mostly `timeout` → go to [Latency](#latency).
   * Mostly `unavailable` → replicas are unready or restarting; go to [Not ready](#not-ready).
   * Mostly `error` → check logs for `inference failed`. If it started with a deploy, **roll back first**:
     `kubectl rollout undo deploy/word2eq`.
2. Correlate with `word2eq_model_info` version changes and recent deploys.

## Latency

**Signal:** `Word2EqLatencySLO` (p99 > 250 ms for 10 min).

* **Queue wait p99 high, inference time normal** → saturation. Scale out
  (`kubectl scale deploy/word2eq --replicas=N`) and check the HPA is not at `maxReplicas`.
* **Inference time high at normal batch sizes** → CPU contention. Look for throttling
  (`container_cpu_cfs_throttled_periods_total`) and noisy neighbours, and check that
  `WORD2EQ_THREADS`/`OMP_NUM_THREADS` do not exceed the CPU request.
* **Batch size p50 near `WORD2EQ_MAX_BATCH`** → the batcher is saturated. Add replicas rather than
  raising `MAX_BATCH`, since bigger batches raise per-request latency.
* Mitigation knob: lowering `WORD2EQ_MAX_WAIT_MS` cuts idle-time latency and costs throughput.

## Load shedding

**Signal:** `Word2EqLoadShedding` (`word2eq_shed_total{reason="queue_full"}` > 0).

Shedding is doing its job: it keeps latency bounded for the requests it admits. The fix is capacity:

1. Scale out. Per-replica capacity is in [SLO.md](SLO.md#capacity).
2. If traffic is a retry storm (sheds and request rate rising together), confirm clients honour
   `Retry-After` and use jittered backoff.
3. Don't raise `WORD2EQ_MAX_QUEUE` to hide the problem. That turns fast 503s into slow 504s.

`reason="deadline_expired"` means requests sat in the queue past their deadline. It is the same
saturation signal, arriving later.

## Unverified answers

**Signal:** `Word2EqUnverifiedAnswers` (> 2% of OK responses have `verified=false`).

Constrained decoding makes malformed equations impossible, so an unverified answer means the
expression evaluated to a non-finite value (for example, division by zero). A rising rate points to:

* **Input drift**: new problem styles (units, fractions, numbers written as words). Sample recent
  inputs and compare them with the training distribution.
* **Bad model promotion**: check whether it lines up with a `word2eq_model_info` version change.
  Roll back, then re-run `word2eq eval` on the candidate artifact.

## Not ready

**Signal:** `Word2EqNotReady`, or pods stuck failing their `startupProbe`.

```bash
kubectl logs deploy/word2eq | grep -E "model load failed|ready"
curl -s http://<pod>:8000/readyz      # {"status": "failed", "error": "..."}
```

* `FileNotFoundError` means the artifact is missing from the image. Check `WORD2EQ_ARTIFACT` and the
  `ARTIFACT` build arg.
* `RuntimeError ... state_dict` means the model code and artifact versions don't match. Roll back the
  image.
* Liveness (`/healthz`) deliberately stays green while the model is unready, so Kubernetes won't
  crash-loop a pod that is only slow to load.

## Model promotion

1. `word2eq train --config configs/base.yaml --out-dir runs/<candidate>`
2. `word2eq gate --candidate runs/<candidate>/metrics.json --baseline baselines/svamp.json`.
   This fails if answer accuracy drops by more than 1 point or grammar validity falls below 100%.
3. Build the image with `--build-arg ARTIFACT=runs/<candidate>/artifact`, then roll out
   (`maxUnavailable: 0`).
4. Watch `sum by (version) (word2eq_model_info)` and the burn-rate alerts for 30 minutes. If anything
   degrades, run `kubectl rollout undo`.

## Training jobs

* Preemption-safe: SIGTERM checkpoints atomically, and `word2eq train --resume` continues from the
  checkpoint to bit-identical weights.
* `--resume` refuses a checkpoint from a different config (fingerprint mismatch) instead of
  silently mixing runs.
