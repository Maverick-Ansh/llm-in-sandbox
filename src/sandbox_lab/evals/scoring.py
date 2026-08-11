"""Deterministic grading.

Grading is where a sandbox-vs-baseline comparison quietly dies. Two rules:

1. **The same grader runs on both modes.** Any asymmetry becomes indistinguishable
   from the effect being measured.
2. **No LLM judge.** A judge introduces its own variance and its own bias, and
   the sandbox mode tends to produce tidier, more machine-like answers (``42``)
   than the prose baseline (``the answer is 42 units``). A judge that rewards
   tidiness would credit the sandbox for formatting.

So: normalise hard, compare exactly, and fall back to symbolic equivalence for
maths. Every normalisation below is applied to both sides.
"""

from __future__ import annotations

import re
from fractions import Fraction

# Chatter models wrap answers in. Stripped from both prediction and reference.
_PREAMBLE = re.compile(
    r"^\s*(the\s+)?(final\s+)?answer\s*(is|:)?\s*", re.IGNORECASE
)
_LATEX_WRAPPERS = [
    (re.compile(r"\\boxed\s*\{(.+)\}", re.DOTALL), r"\1"),
    (re.compile(r"\\text\s*\{(.+?)\}", re.DOTALL), r"\1"),
    (re.compile(r"\\mathrm\s*\{(.+?)\}", re.DOTALL), r"\1"),
    (re.compile(r"\$+(.+?)\$+", re.DOTALL), r"\1"),
]
_UNITS = re.compile(
    r"\s*(units?|meters?|metres?|m/s\^?2?|seconds?|s|kg|grams?|g|joules?|J|"
    r"newtons?|N|degrees?|percent|%|dollars?|cm|mm|km|mol|moles?|kelvin|K)\s*$",
    re.IGNORECASE,
)


def normalize(text: str | None) -> str:
    """Canonicalise an answer string for comparison."""
    if text is None:
        return ""
    out = str(text).strip()
    out = _PREAMBLE.sub("", out)
    for pattern, repl in _LATEX_WRAPPERS:
        out = pattern.sub(repl, out)
    out = out.strip().strip(".").strip()
    out = _UNITS.sub("", out)
    out = out.replace("\\!", "").replace("\\,", "").replace("\\ ", " ")
    out = out.replace("\\left", "").replace("\\right", "")
    out = out.replace("^{\\circ}", "").replace("^\\circ", "")
    # 1,234 -> 1234, but leave "1, 2, 3" tuples alone.
    out = re.sub(r"(?<=\d),(?=\d{3}\b)", "", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out.rstrip(".").strip()


def _as_number(text: str) -> float | None:
    """Parse a scalar, accepting fractions, percentages and LaTeX \\frac."""
    text = text.strip().replace(" ", "")
    frac = re.fullmatch(r"\\d?frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", text)
    if frac:
        try:
            return float(frac.group(1)) / float(frac.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    if text.endswith("%"):
        try:
            return float(text[:-1]) / 100.0
        except ValueError:
            return None
    try:
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        pass
    try:
        return float(text)
    except ValueError:
        return None


def numeric_match(pred: str, gold: str, *, rel_tol: float = 1e-4) -> bool:
    """Compare as numbers when both sides parse as numbers.

    The tolerance exists because the sandbox mode returns *computed* values
    (``0.3333333333``) where the reference is symbolic (``1/3``). Marking that
    wrong would penalise the very behaviour under study. The tolerance is
    relative so it behaves sanely across magnitudes.
    """
    a, b = _as_number(pred), _as_number(gold)
    if a is None or b is None:
        return False
    if a == b:
        return True
    scale = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / scale < rel_tol


def symbolic_match(pred: str, gold: str) -> bool:
    """Fall back to sympy for expressions numeric comparison cannot settle.

    Optional: if sympy is missing, or the expression does not parse, we simply
    decline rather than guessing.
    """
    try:
        from sympy import simplify
        from sympy.parsing.latex import parse_latex  # noqa: F401
        from sympy.parsing.sympy_parser import parse_expr
    except ImportError:
        return False
    try:
        lhs, rhs = parse_expr(pred), parse_expr(gold)
        return bool(simplify(lhs - rhs) == 0)
    except Exception:  # noqa: BLE001 - sympy raises a wide variety on bad input
        return False


def grade_multiple_choice(pred: str | None, gold: str) -> bool:
    """Grade a lettered multiple-choice answer.

    Accepts ``B``, ``(B)``, ``B.``, ``B) text`` and ``Answer: B``. Deliberately
    does *not* try to match the option's text: a model that reproduces the
    option verbatim without naming the letter has not followed the instruction,
    and letting that pass differs between modes.
    """
    text = normalize(pred).upper()
    if not text:
        return False
    match = re.match(r"^\(?([A-J])\)?(?:[.):]|\s|$)", text)
    letter = match.group(1) if match else (text if len(text) == 1 else "")
    return letter == normalize(gold).upper().strip("()")


def grade_free_form(pred: str | None, gold: str) -> bool:
    """Grade a numeric or short symbolic answer."""
    p, g = normalize(pred), normalize(gold)
    if not p:
        return False
    if p == g or p.lower() == g.lower():
        return True
    if numeric_match(p, g):
        return True
    return symbolic_match(p, g)


def grade(pred: str | None, gold: str, kind: str) -> bool:
    if kind == "multiple_choice":
        return grade_multiple_choice(pred, gold)
    if kind == "free_form":
        return grade_free_form(pred, gold)
    raise ValueError(f"unknown answer kind {kind!r}")
