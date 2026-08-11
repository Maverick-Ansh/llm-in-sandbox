#!/usr/bin/env python
"""Turn a results.jsonl into a markdown report with honest uncertainty.

The paper reports bare deltas. At this scale (tens of items per domain) a bare
delta is not interpretable: a +4 point difference on 25 paired items is two
questions changing hands, which is well inside noise. So every delta here comes
with a paired bootstrap CI and an exact McNemar test.

Both statistics are *paired* because the design is paired - the same items are
run in both conditions. Treating the two arms as independent samples throws away
the pairing and inflates the variance, which for small n is the difference
between "no detectable effect" and a misleading claim in either direction.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # partial final line from a killed run
    # Later rows supersede earlier attempts at the same task/condition.
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique[f"{row['task_id']}|{row['mode']}|{row['caps']}"] = row
    return list(unique.values())


def pair(rows: list[dict], a: str, b: str) -> list[tuple[dict, dict]]:
    """Match rows from two conditions by task id. Conditions are 'mode/caps'."""
    index: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        index[row["task_id"]][f"{row['mode']}/{row['caps']}"] = row
    return [
        (per[a], per[b]) for per in index.values() if a in per and b in per
    ]


def mcnemar_exact(pairs: list[tuple[dict, dict]]) -> tuple[int, int, float]:
    """Exact McNemar test on discordant pairs.

    Only pairs where the two conditions disagree carry information about which
    is better; concordant pairs cancel. Under the null the number of wins among
    n discordant pairs is Binomial(n, 0.5), so the two-sided p is exact rather
    than a chi-square approximation - which matters because at this scale the
    discordant count is often under 10, where the approximation is poor.
    """
    b_wins = sum(1 for x, y in pairs if not x["correct"] and y["correct"])
    a_wins = sum(1 for x, y in pairs if x["correct"] and not y["correct"])
    n = a_wins + b_wins
    if n == 0:
        return a_wins, b_wins, 1.0
    k = min(a_wins, b_wins)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return a_wins, b_wins, min(1.0, 2 * tail)


def bootstrap_ci(
    pairs: list[tuple[dict, dict]],
    stat,
    *,
    iterations: int = 5000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile CI, resampling *pairs* so the pairing is preserved."""
    if not pairs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(pairs)
    samples = []
    for _ in range(iterations):
        draw = [pairs[rng.randrange(n)] for _ in range(n)]
        try:
            samples.append(stat(draw))
        except ZeroDivisionError:
            continue
    if not samples:
        return (float("nan"), float("nan"))
    samples.sort()
    lo = samples[int(0.025 * len(samples))]
    hi = samples[min(len(samples) - 1, int(0.975 * len(samples)))]
    return lo, hi


def acc_delta_pp(pairs: list[tuple[dict, dict]]) -> float:
    """Sandbox accuracy minus baseline accuracy, in percentage points."""
    base = sum(x["correct"] for x, _ in pairs) / len(pairs)
    sand = sum(y["correct"] for _, y in pairs) / len(pairs)
    return (sand - base) * 100


def token_ratio(pairs: list[tuple[dict, dict]]) -> float:
    """Mean sandbox tokens / mean baseline tokens (the paper's Table 5 form)."""
    base = sum(x["generated_tokens"] for x, _ in pairs) / len(pairs)
    sand = sum(y["generated_tokens"] for _, y in pairs) / len(pairs)
    return sand / base


def fmt(value: float, places: int = 1) -> str:
    return "n/a" if value != value else f"{value:.{places}f}"  # NaN check


def report(rows: list[dict], baseline: str, treatment: str) -> str:
    pairs = pair(rows, baseline, treatment)
    if not pairs:
        return f"No paired items between {baseline!r} and {treatment!r}.\n"

    lines: list[str] = []
    add = lines.append
    add(f"## {treatment} vs {baseline}\n")
    add(f"Paired items: **{len(pairs)}**\n")

    # ---- accuracy, overall and per domain
    add("### Accuracy\n")
    add("| domain | n | baseline | sandbox | delta (pp) | 95% CI | McNemar p |")
    add("|---|---:|---:|---:|---:|---:|---:|")

    groups: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for x, y in pairs:
        groups[x["domain"]].append((x, y))
    for domain in [*sorted(groups), "ALL"]:
        subset = pairs if domain == "ALL" else groups[domain]
        base = sum(x["correct"] for x, _ in subset) / len(subset)
        sand = sum(y["correct"] for _, y in subset) / len(subset)
        delta = acc_delta_pp(subset)
        lo, hi = bootstrap_ci(subset, acc_delta_pp)
        _, _, p = mcnemar_exact(subset)
        flag = " **" if p < 0.05 else ""
        add(
            f"| {domain} | {len(subset)} | {base * 100:.1f} | {sand * 100:.1f} | "
            f"{delta:+.1f}{flag} | [{fmt(lo)}, {fmt(hi)}] | {p:.3f} |"
        )

    # ---- tokens
    add("\n### Generated tokens\n")
    add("| domain | n | baseline | sandbox | ratio | 95% CI |")
    add("|---|---:|---:|---:|---:|---:|")
    for domain in [*sorted(groups), "ALL"]:
        subset = pairs if domain == "ALL" else groups[domain]
        base = sum(x["generated_tokens"] for x, _ in subset) / len(subset)
        sand = sum(y["generated_tokens"] for _, y in subset) / len(subset)
        lo, hi = bootstrap_ci(subset, token_ratio)
        add(
            f"| {domain} | {len(subset)} | {base:.0f} | {sand:.0f} | "
            f"{sand / base:.2f}x | [{fmt(lo, 2)}, {fmt(hi, 2)}] |"
        )

    # ---- behaviour: what the agent actually did
    add("\n### Agent behaviour (sandbox arm)\n")
    stops: dict[str, int] = defaultdict(int)
    turns = commands = timeouts = edits = restarts = 0
    for _, y in pairs:
        stops[y["stop_reason"]] += 1
        turns += y["n_turns"]
        stats = y.get("sandbox_stats") or {}
        commands += stats.get("commands", 0)
        timeouts += stats.get("timeouts", 0)
        edits += stats.get("edits", 0)
        restarts += stats.get("shell_restarts", 0)
    n = len(pairs)
    add(f"- mean turns: **{turns / n:.1f}**")
    add(f"- mean bash commands: **{commands / n:.1f}**, file edits: **{edits / n:.1f}**")
    add(f"- command timeouts: **{timeouts}** total, shell restarts: **{restarts}**")
    add(f"- stop reasons: {dict(sorted(stops.items(), key=lambda kv: -kv[1]))}")

    # A high "stalled"/"max_turns" share means the ceiling is the harness, not
    # the model, and the accuracy delta should be read with that in mind.
    unfinished = stops.get("max_turns", 0) + stops.get("stalled", 0)
    if unfinished:
        add(
            f"\n> {unfinished}/{n} sandbox episodes ended without calling `finish`. "
            "That is a harness/turn-budget ceiling as much as a capability result."
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", help="path to results.jsonl")
    p.add_argument("--baseline", default="direct/full")
    p.add_argument("--treatment", default="sandbox/full")
    p.add_argument("--out", default=None, help="write markdown here as well as stdout")
    args = p.parse_args(argv)

    rows = load_rows(Path(args.results))
    conditions = sorted({f"{r['mode']}/{r['caps']}" for r in rows})
    header = f"# Results\n\n{len(rows)} episodes across conditions: {conditions}\n\n"
    text = header + report(rows, args.baseline, args.treatment)

    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
