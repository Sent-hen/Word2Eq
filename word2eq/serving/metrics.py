"""Prometheus instrumentation. Each app instance owns a registry so tests and
multiple apps in one process don't collide on global metric names."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from .batcher import Observer

LATENCY_BUCKETS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0, 2.5)


class Metrics(Observer):
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter("word2eq_requests_total", "Solve requests by outcome", ["outcome"],
                                registry=r)
        self.latency = Histogram("word2eq_request_latency_seconds", "End-to-end solve latency",
                                 buckets=LATENCY_BUCKETS, registry=r)
        self.inference = Histogram("word2eq_batch_inference_seconds", "Model time per batch",
                                   buckets=LATENCY_BUCKETS, registry=r)
        self.batch_size = Histogram("word2eq_batch_size", "Requests per forward pass",
                                    buckets=(1, 2, 4, 8, 16, 32, 64), registry=r)
        self.queue_wait_s = Histogram("word2eq_queue_wait_seconds", "Time spent queued before inference",
                                      buckets=LATENCY_BUCKETS, registry=r)
        self.depth = Gauge("word2eq_queue_depth", "Requests waiting for a batch slot", registry=r)
        self.shed_total = Counter("word2eq_shed_total", "Requests rejected before inference", ["reason"],
                                  registry=r)
        self.unverified = Counter("word2eq_unverified_answers_total",
                                  "Answers that failed symbolic verification", registry=r)
        self.ready = Gauge("word2eq_ready", "1 when the model is loaded and warmed up", registry=r)
        self.model_info = Gauge("word2eq_model_info", "Loaded model version", ["version", "quantized"],
                                registry=r)

    # Observer hooks called by the batcher
    def queue_depth(self, depth: int) -> None:
        self.depth.set(depth)

    def batch(self, size: int, seconds: float) -> None:
        self.batch_size.observe(size)
        self.inference.observe(seconds)

    def shed(self, reason: str) -> None:
        self.shed_total.labels(reason).inc()

    def queue_wait(self, seconds: float) -> None:
        self.queue_wait_s.observe(seconds)
