import json

import torch

from word2eq.artifact import load_artifact
from word2eq.train import StopFlag, train

from .conftest import tiny_config


class StopAfter(StopFlag):
    """Simulates a SIGTERM arriving after ``n`` optimizer steps."""

    def __init__(self, n: int):
        super().__init__()
        self._n = n
        self._reads = 0

    @property
    def requested(self) -> bool:
        self._reads += 1
        return self._reads >= self._n

    @requested.setter
    def requested(self, _v) -> None:
        pass


def _weights(path):
    return torch.load(path / "artifact" / "model.pt", weights_only=True)


def test_training_is_deterministic(tmp_path):
    r1 = train(tiny_config(tmp_path / "a"))
    r2 = train(tiny_config(tmp_path / "b"))
    w1, w2 = _weights(tmp_path / "a"), _weights(tmp_path / "b")
    assert all(torch.equal(w1[k], w2[k]) for k in w1)
    assert r1["test"]["constrained"]["answer_acc"] == r2["test"]["constrained"]["answer_acc"]


def test_preempted_run_resumes_to_identical_weights(tmp_path):
    train(tiny_config(tmp_path / "ref"))

    # 64 examples / batch 16 = 4 steps per epoch: stop mid-way through epoch 0.
    res = train(tiny_config(tmp_path / "pre"), stop=StopAfter(3))
    assert res["status"] == "interrupted"
    assert res["state"]["step"] == 3
    assert (tmp_path / "pre" / "checkpoint.pt").exists()

    res = train(tiny_config(tmp_path / "pre"), resume=True)
    assert res["status"] == "completed"
    w_ref, w_res = _weights(tmp_path / "ref"), _weights(tmp_path / "pre")
    assert all(torch.equal(w_ref[k], w_res[k]) for k in w_ref)


def test_resume_refuses_mismatched_config(tmp_path):
    train(tiny_config(tmp_path / "x"), stop=StopAfter(1))
    try:
        train(tiny_config(tmp_path / "x", lr=1e-2), resume=True)
    except RuntimeError as e:
        assert "different config" in str(e)
    else:
        raise AssertionError("expected refusal")


def test_artifact_has_lineage_and_loads_safely(tiny_artifact):
    model, sv, tv, card = load_artifact(tiny_artifact)
    assert card["config_fingerprint"] and card["git"]["sha"]
    assert all(v["sha256"] and v["rows"] > 0 for v in card["data"].values())
    assert card["metrics"]["test"]["constrained"]["valid_rate"] == 1.0
    json.loads((tiny_artifact / "vocab.json").read_text())  # plain JSON, not pickle
