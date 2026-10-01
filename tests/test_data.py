import random

from word2eq.data import load_csv, mask_numbers, stable_split, tokenize
from word2eq.vocab import Vocab


def test_mask_numbers_order_and_formats():
    text = "Dan had $3 left. He paid 1,200 dollars and 2.5 more, then .75 on the 3rd day."
    masked, nums = mask_numbers(text)
    assert nums == [3.0, 1200.0, 2.5, 0.75]
    assert "number0" in masked and "number3" in masked
    assert "3rd" in masked  # ordinals are not quantities


def test_serving_tokenisation_matches_training_format():
    # A raw problem, once masked, must tokenise exactly like the pre-masked dataset text.
    raw = "Julia played tag with 18 kids on Monday. She played tag with 10 kids on Tuesday."
    dataset = "julia played tag with number0 kids on monday . she played tag with number1 kids on tuesday ."
    masked, _ = mask_numbers(raw)
    assert tokenize(masked) == dataset.split()


def test_stable_split_is_deterministic_and_order_independent():
    exs = load_csv("data/cv_asdiv-a/fold0/dev.csv")
    tr1, va1 = stable_split(exs, 0.2)
    shuffled = exs[:]
    random.Random(0).shuffle(shuffled)
    tr2, va2 = stable_split(shuffled, 0.2)
    assert {e.question for e in va1} == {e.question for e in va2}
    assert len(tr1) + len(va1) == len(exs)
    assert 0.1 < len(va1) / len(exs) < 0.3


def test_vocab_roundtrip():
    v = Vocab.build([["a", "b", "a"], ["c"]], reserved=["number0"])
    ids = v.encode(["a", "zzz"])
    assert v.decode(ids) == ["a", "<unk>"]
    assert Vocab.from_dict(v.to_dict()).itos == v.itos
