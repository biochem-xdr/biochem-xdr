import sys
sys.path.insert(0, ".")

from evaluation.score_mcq import parse_answer, score


def test_direct_letter():
    assert parse_answer("A") == "A"
    assert parse_answer("b") == "B"
    assert parse_answer("C") == "C"
    assert parse_answer("d") == "D"


def test_letter_with_punctuation():
    assert parse_answer("A.") == "A"
    assert parse_answer("B) some text") == "B"


def test_answer_prefix():
    assert parse_answer("Answer: C") == "C"
    assert parse_answer("Answer is D") == "D"
    assert parse_answer("The answer is B.") == "B"


def test_thinking_model_tail():
    long = "A lot of reasoning here... " * 50 + "\n\nD"
    assert parse_answer(long) == "D"


def test_none_on_empty():
    assert parse_answer("") is None
    assert parse_answer(None) is None


def test_score():
    assert score("A", "A") is True
    assert score("B", "A") is False
    assert score(None, "A") is None


if __name__ == "__main__":
    test_direct_letter()
    test_letter_with_punctuation()
    test_answer_prefix()
    test_thinking_model_tail()
    test_none_on_empty()
    test_score()
    print("All scoring tests passed.")