#!/usr/bin/env python
"""Re-derive answers and grades from stored trajectories.

Grading changed after episodes had already run. Re-running them on the GPU would
be slow *and* wrong: the two arms would then have been produced under different
code, and any difference between them would be partly the code change. Since the
full trajectory is stored, the honest fix is to re-derive every answer offline
with one version of the extractor and grader, so both arms are treated
identically.

This does not re-run the model, so it cannot repair an episode that errored
before producing output (a timeout has no trajectory). Those need a real re-run;
`run_sweep.py --retry-stop-reasons` handles them.

    python scripts/regrade.py runs/pilot_qwen3-4b
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from sandbox_lab.agent import extract_final_answer
from sandbox_lab.evals.scoring import grade


def answer_from_trajectory(traj: dict) -> tuple[str | None, str]:
    """Recover the final answer, preferring an explicit `finish` call.

    Returns ``(answer, source)``. The source is recorded so a regrade can be
    audited: if a large number of episodes suddenly resolve through the terse
    fallback, that is worth knowing rather than silently absorbing.
    """
    for turn in reversed(traj.get("turns", [])):
        for call in turn.get("tool_calls", []):
            if call.get("name") == "finish":
                answer = (call.get("arguments") or {}).get("answer")
                if answer:
                    return str(answer).strip(), "finish"

    # No finish call: fall back to parsing assistant prose, newest turn first.
    for turn in reversed(traj.get("turns", [])):
        parsed = extract_final_answer(turn.get("content") or "")
        if parsed:
            return parsed, "parsed"
    return None, "none"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", help="directory containing results.jsonl and trajectories/")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    results_path = run_dir / "results.jsonl"
    traj_dir = run_dir / "trajectories"
    if not results_path.exists():
        raise SystemExit(f"no results.jsonl in {run_dir}")

    rows = []
    with results_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    changed, sources, missing = 0, Counter(), 0
    flips = Counter()
    for row in rows:
        safe = row["task_id"].replace("/", "_")
        path = traj_dir / f"{row['mode']}_{row['caps']}_{safe}.json"
        if not path.exists():
            missing += 1
            continue
        traj = json.loads(path.read_text(encoding="utf-8"))
        answer, source = answer_from_trajectory(traj)
        sources[source] += 1

        # Look up the item's answer kind the same way the runner did: multiple
        # choice for mmlu_pro, free form otherwise.
        kind = "multiple_choice" if row["benchmark"] == "mmlu_pro" else "free_form"
        correct = grade(answer, row["gold"], kind)

        if correct != row["correct"] or answer != row["predicted"]:
            changed += 1
            flips[f"{row['correct']}->{correct}"] += 1
        row["predicted"] = answer
        row["correct"] = correct
        if answer and row["stop_reason"] == "no_answer_parsed":
            row["stop_reason"] = "answered"

    print(f"episodes: {len(rows)}  changed: {changed}  no trajectory: {missing}")
    print(f"answer source: {dict(sources)}")
    print(f"grade flips: {dict(flips)}")
    for mode in sorted({r["mode"] for r in rows}):
        subset = [r for r in rows if r["mode"] == mode]
        acc = sum(r["correct"] for r in subset) / len(subset)
        print(f"  {mode:8s} n={len(subset):3d} acc={acc:.3f}")

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0

    backup = results_path.with_suffix(".jsonl.bak")
    shutil.copy2(results_path, backup)
    with results_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"\nrewrote {results_path} (backup at {backup})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
