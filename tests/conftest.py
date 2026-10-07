import pytest

from word2eq.config import TrainConfig

SMALL_TEST = "data/cv_asdiv-a/fold0/dev.csv"


def tiny_config(out_dir, **kw) -> TrainConfig:
    base = dict(
        train_files=["data/mawps-asdiv-a_svamp/train.csv"],
        test_files=[SMALL_TEST],
        out_dir=str(out_dir),
        seed=3,
        d_model=32,
        nhead=2,
        enc_layers=1,
        dec_layers=1,
        ff_dim=64,
        dropout=0.1,
        epochs=2,
        batch_size=16,
        warmup_steps=5,
        limit_train=64,
        num_threads=1,
    )
    base.update(kw)
    return TrainConfig(**base)


@pytest.fixture(scope="session")
def tiny_artifact(tmp_path_factory):
    from word2eq.train import train

    out = tmp_path_factory.mktemp("tiny")
    train(tiny_config(out))
    return out / "artifact"
