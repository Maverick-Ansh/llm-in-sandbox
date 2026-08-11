"""Grading and answer-extraction tests.

Both modes share this code, so a bug here shows up as a fake effect size rather
than as an obvious crash. The regressions pinned below were all observed in real
model output, not invented.
"""

from __future__ import annotations

import pytest

from sandbox_lab.agent import extract_final_answer
from sandbox_lab.evals.scoring import (
    grade_free_form,
    grade_multiple_choice,
    normalize,
)

# ------------------------------------------------------------------ extraction


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("Some working.\nFINAL ANSWER: 849", "849"),
        ("FINAL ANSWER: 849\n", "849"),
        ("**FINAL ANSWER:** B", "B"),
        ("final answer: 1/3", "1/3"),
        ("The final answer is 42", "42"),
        # Observed regression: the label leaked into the captured answer and
        # graded a correct response as wrong.
        ("FINAL ANSWER: FINAL ANSWER: 849", "849"),
        ("blah \\boxed{17} blah", "17"),
        ("no answer here", None),
        ("", None),
    ],
)
def test_extract_final_answer(content, expected):
    assert extract_final_answer(content) == expected


def test_last_answer_wins():
    """A model that revises itself means the final statement, not the first."""
    assert extract_final_answer("FINAL ANSWER: 3\nwait\nFINAL ANSWER: 4") == "4"


# ------------------------------------------- terse answers without the label
#
# Observed on a real run: 17 of 99 baseline episodes answered in 2-15 tokens,
# obeying the benchmark's own "answer with the letter only" instruction rather
# than the system prompt's FINAL ANSWER format, and were scored as having no
# answer. That penalises the baseline specifically - the sandbox arm answers
# through the `finish` tool and is never affected.


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("C", "C"),
        ("  B  ", "B"),
        ("42", "42"),
        ("F. The population of SAT scores from each group is normally distributed.", "F"),
        ("(D) 3.14 metres per second", "D"),
        ("A: the first option", "A"),
    ],
)
def test_terse_answers_are_accepted(content, expected):
    assert extract_final_answer(content) == expected


def test_prose_is_not_mistaken_for_a_lettered_answer():
    """The delimiter requirement is what makes the fallback safe.

    Without it, any reasoning line beginning with a capital A-J word ("A more
    careful analysis...") would be harvested as that letter, silently scoring
    long completions against a letter they never chose.
    """
    assert extract_final_answer("A more careful analysis shows the value grows") is None
    assert extract_final_answer("Before concluding, check the units") is None
    assert (
        extract_final_answer("\n".join(f"step {i} of the derivation" for i in range(30)))
        is None
    )


def test_long_unlabelled_reasoning_yields_no_answer():
    """A long chain with no marker and no terse ending is genuinely unanswered."""
    body = "We integrate by parts and obtain a messy expression " * 20
    assert extract_final_answer(body) is None


# ------------------------------------------------------------- multiple choice


@pytest.mark.parametrize(
    ("pred", "gold", "ok"),
    [
        ("B", "B", True),
        ("(B)", "B", True),
        ("B.", "B", True),
        ("B) the second option", "B", True),
        ("Answer: B", "B", True),
        ("b", "B", True),
        ("C", "B", False),
        ("", "B", False),
        (None, "B", False),
    ],
)
def test_grade_multiple_choice(pred, gold, ok):
    assert grade_multiple_choice(pred, gold) is ok


def test_multiple_choice_does_not_credit_option_text_alone():
    """Reproducing the option without naming the letter is not an answer.

    Crediting it would behave differently between modes: the prose baseline
    restates options far more often than the tool-calling agent does.
    """
    assert grade_multiple_choice("the acceleration is 9.8 m/s^2", "B") is False


# ------------------------------------------------------------------ free form


@pytest.mark.parametrize(
    ("pred", "gold", "ok"),
    [
        ("849", "849", True),
        (" 849 ", "849", True),
        ("849.", "849", True),
        ("1,234", "1234", True),
        ("$42$", "42", True),
        ("\\boxed{42}", "42", True),
        ("42 units", "42", True),
        ("0.5", "1/2", True),
        ("50%", "1/2", True),
        ("849", "850", False),
        ("", "849", False),
    ],
)
def test_grade_free_form(pred, gold, ok):
    assert grade_free_form(pred, gold) is ok


def test_computed_decimal_matches_symbolic_reference():
    """A computed 0.333... must match a reference of 1/3.

    This is exactly what the sandbox mode produces, so marking it wrong would
    penalise the behaviour under study and manufacture a negative result.
    """
    assert grade_free_form("0.3333333333", "1/3") is True


def test_tolerance_does_not_swallow_genuinely_wrong_answers():
    assert grade_free_form("0.34", "1/3") is False
    assert grade_free_form("1000000", "1000001") is False


def test_normalize_is_idempotent():
    for text in ["849", "**FINAL ANSWER: 42**", "$x^2$", "1,234 units"]:
        once = normalize(text)
        assert normalize(once) == once
