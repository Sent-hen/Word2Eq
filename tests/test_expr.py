import pytest

from word2eq.expr import answers_match, evaluate_prefix, is_valid_prefix, to_infix


@pytest.mark.parametrize(
    "tokens,valid",
    [
        (["number0"], True),
        (["-", "number1", "number0"], True),
        (["/", "+", "number0", "number1", "number2"], True),
        (["+", "number0"], False),  # missing operand
        (["number0", "number1"], False),  # trailing token
        ([], False),
        (["+", "number0", "apple"], False),  # unknown token
        (["*", "number0", "100.0"], True),  # constant operand
    ],
)
def test_is_valid_prefix(tokens, valid):
    assert is_valid_prefix(tokens) is valid


def test_evaluate_prefix():
    assert evaluate_prefix(["-", "number1", "number0"], [3.0, 10.0]) == 7.0
    assert evaluate_prefix(["/", "+", "number0", "number1", "number2"], [4.0, 2.0, 3.0]) == 2.0
    assert evaluate_prefix(["*", "number0", "0.01"], [50.0]) == pytest.approx(0.5)


def test_evaluate_prefix_rejects_bad_input():
    assert evaluate_prefix(["/", "number0", "number1"], [1.0, 0.0]) is None  # div by zero
    assert evaluate_prefix(["+", "number0", "number3"], [1.0, 2.0]) is None  # missing number
    assert evaluate_prefix(["+", "number0"], [1.0]) is None  # malformed


def test_to_infix():
    assert to_infix(["-", "number1", "number0"], [3.0, 10.0]) == "10 - 3"
    assert to_infix(["*", "+", "number0", "number1", "number2"]) == "(number0 + number1) * number2"
    assert to_infix(["number0"], [2.5]) == "2.5"
    assert to_infix(["+"]) is None


def test_answers_match():
    assert answers_match(51.00001, 51.0)
    assert not answers_match(50.0, 51.0)
    assert not answers_match(None, 1.0)
