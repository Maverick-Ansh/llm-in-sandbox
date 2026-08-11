"""Benchmark loading.

Domain coverage is chosen to line up with the paper's Table 2 columns (maths,
physics, chemistry, biomedicine) while staying on openly downloadable datasets -
GPQA is gated behind an access agreement, so MMLU-Pro carries the science
domains instead. It has the useful property of being a *single* dataset with a
category field, so cross-domain comparisons are not confounded by differences in
question style, difficulty calibration, or answer format between sources.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

# MMLU-Pro categories mapped onto the paper's domains. "biomedicine" merges
# biology and health, which is what the paper's own category covers.
MMLU_PRO_DOMAINS: dict[str, tuple[str, ...]] = {
    "math": ("math",),
    "physics": ("physics",),
    "chemistry": ("chemistry",),
    "biomedicine": ("biology", "health"),
}


@dataclass
class Task:
    """One benchmark item, in the form the agent loop consumes."""

    task_id: str
    question: str
    answer: str
    kind: str  # "multiple_choice" | "free_form"
    domain: str
    benchmark: str
    documents: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


def _mc_prompt(question: str, options: list[str]) -> str:
    letters = [chr(ord("A") + i) for i in range(len(options))]
    body = "\n".join(
        f"{letter}. {text}" for letter, text in zip(letters, options, strict=True)
    )
    return (
        f"{question}\n\n{body}\n\n"
        f"Answer with the letter of the correct option ({letters[0]}-{letters[-1]}) only."
    )


def load_mmlu_pro(
    domains: list[str] | None = None,
    *,
    n_per_domain: int = 50,
    seed: int = 0,
    split: str = "test",
) -> list[Task]:
    """Sample MMLU-Pro items per domain.

    Sampling is seeded and done *after* filtering by category, so the same seed
    yields the same items for every model and every mode. Comparing runs drawn
    from different samples would swamp the effect size with sampling noise -
    MMLU-Pro items vary enormously in difficulty.
    """
    from datasets import load_dataset

    domains = domains or list(MMLU_PRO_DOMAINS)
    data = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    tasks: list[Task] = []

    for domain in domains:
        categories = MMLU_PRO_DOMAINS.get(domain)
        if categories is None:
            raise ValueError(f"unknown domain {domain!r}; expected {sorted(MMLU_PRO_DOMAINS)}")
        pool = [row for row in data if row["category"] in categories]
        rng = random.Random(f"{seed}:{domain}")
        rng.shuffle(pool)
        for row in pool[:n_per_domain]:
            options = [o for o in row["options"] if o != "N/A"]
            tasks.append(
                Task(
                    task_id=f"mmlu_pro/{domain}/{row['question_id']}",
                    question=_mc_prompt(row["question"], options),
                    answer=row["answer"],
                    kind="multiple_choice",
                    domain=domain,
                    benchmark="mmlu_pro",
                    meta={"category": row["category"], "n_options": len(options)},
                )
            )
    return tasks


def load_math500(*, n: int = 100, seed: int = 0) -> list[Task]:
    """MATH-500: free-form competition maths, the paper's strongest domain."""
    from datasets import load_dataset

    data = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    rng = random.Random(seed)
    rng.shuffle(data)
    return [
        Task(
            task_id=f"math500/{row.get('unique_id', i)}",
            question=(
                f"{row['problem']}\n\n"
                "Give the final answer in simplest exact form."
            ),
            answer=str(row["answer"]),
            kind="free_form",
            domain="math",
            benchmark="math500",
            meta={"level": row.get("level"), "subject": row.get("subject")},
        )
        for i, row in enumerate(data[:n])
    ]


def load_aime(*, year: str = "2025", n: int | None = None) -> list[Task]:
    """AIME: integer answers in [0, 999]. Small but very high signal.

    Worth running despite n=30: these problems are hard enough that the
    "compute it rather than reason about it" behaviour the sandbox encourages
    either clearly helps or clearly does not.
    """
    from datasets import load_dataset

    repo = {"2024": "HuggingFaceH4/aime_2024", "2025": "yentinglin/aime_2025"}[year]
    data = list(load_dataset(repo, split="train"))
    rows = data if n is None else data[:n]
    return [
        Task(
            task_id=f"aime{year}/{row.get('id', i)}",
            question=(
                f"{row['problem']}\n\n"
                "The answer is an integer between 0 and 999 inclusive."
            ),
            answer=str(row["answer"]).strip(),
            kind="free_form",
            domain="math",
            benchmark=f"aime{year}",
            meta={},
        )
        for i, row in enumerate(rows)
    ]


LOADERS = {
    "mmlu_pro": load_mmlu_pro,
    "math500": load_math500,
    "aime": load_aime,
}


def load_suite(name: str, **kwargs: Any) -> list[Task]:
    if name not in LOADERS:
        raise ValueError(f"unknown benchmark {name!r}; expected {sorted(LOADERS)}")
    return LOADERS[name](**kwargs)  # type: ignore[operator]
