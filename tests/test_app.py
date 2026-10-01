import time

from fastapi.testclient import TestClient

from word2eq.serving.app import Settings, create_app
from word2eq.serving.runner import Solution


class FakeRunner:
    version = "test-1"
    quantized = False
    max_src_tokens = 254
    card = {}

    def __init__(self, delay: float = 0.0):
        self.delay = delay

    def warmup(self):
        return 0.0

    def solve_batch(self, items):
        time.sleep(self.delay)
        return [Solution(["+", "number0", "number1"], "a + b", sum(it.numbers[:2]), it.numbers, True)
                for it in items]


def _client(runner, **kw):
    return TestClient(create_app(Settings(**kw), runner_factory=lambda s: runner))


def _wait_ready(c):
    for _ in range(200):
        if c.get("/readyz").status_code == 200:
            return
        time.sleep(0.01)
    raise AssertionError("never became ready")


def test_solve_and_probes():
    with _client(FakeRunner()) as c:
        assert c.get("/healthz").status_code == 200
        _wait_ready(c)
        r = c.post("/v1/solve", json={"problem": "Tom has 5 apples and gets 3 more. How many?"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == 8.0 and body["verified"] and body["model_version"] == "test-1"
        assert body["numbers"] == [5.0, 3.0]
        r = c.post("/v1/solve/batch", json={"problems": ["1 and 2", "3 and 4"]})
        assert [x["answer"] for x in r.json()] == [3.0, 7.0]


def test_input_validation():
    with _client(FakeRunner()) as c:
        _wait_ready(c)
        assert c.post("/v1/solve", json={"problem": "no quantities here"}).status_code == 422
        assert c.post("/v1/solve", json={"problem": ""}).status_code == 422
        assert c.post("/v1/solve", json={"problem": "x" * 5000}).status_code == 422


def test_timeout_maps_to_504_and_is_counted():
    with _client(FakeRunner(delay=0.3), request_timeout_s=0.05) as c:
        _wait_ready(c)
        assert c.post("/v1/solve", json={"problem": "1 and 2"}).status_code == 504
        metrics = c.get("/metrics").text
        assert 'word2eq_requests_total{outcome="timeout"} 1.0' in metrics


def test_inflight_cap_sheds_fast():
    import threading

    with _client(FakeRunner(delay=0.3), max_inflight=1, request_timeout_s=5) as c:
        _wait_ready(c)
        codes = []
        threads = [threading.Thread(target=lambda: codes.append(
            c.post("/v1/solve", json={"problem": "1 and 2"}).status_code)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert codes.count(200) >= 1 and codes.count(503) >= 1
        assert 'word2eq_shed_total{reason="inflight_limit"}' in c.get("/metrics").text


def test_readiness_reports_failed_load():
    def boom(_s):
        raise FileNotFoundError("no artifact")

    with TestClient(create_app(Settings(), runner_factory=boom)) as c:
        time.sleep(0.1)
        r = c.get("/readyz")
        assert r.status_code == 503 and r.json()["status"] == "failed"
        assert c.post("/v1/solve", json={"problem": "1 and 2"}).status_code == 503
        assert c.get("/healthz").status_code == 200  # alive, just not ready


def test_real_model_end_to_end(tiny_artifact):
    with TestClient(create_app(Settings(artifact_dir=str(tiny_artifact)))) as c:
        _wait_ready(c)
        r = c.post("/v1/solve", json={"problem": "Paco had 26 cookies. He ate 9 of them. How many are left?"})
        assert r.status_code == 200
        body = r.json()
        assert body["numbers"] == [26.0, 9.0]
        assert body["verified"] in (True, False)  # tiny model: output is well-formed, maybe not right
        assert body["equation_prefix"]
        assert "word2eq_batch_size_bucket" in c.get("/metrics").text
