"""Run a benchmark suite across modes and write results.

Two properties this needs that a naive loop does not have:

**Resumability.** A full sweep is hours of wall clock on a T4, and a Colab
runtime will disconnect before it finishes. Every trajectory is appended to
JSONL the moment it completes, and a rerun skips whatever is already there. The
alternative - losing four hours to a dropped websocket - is not a hypothetical.

**Bounded concurrency.** vLLM serves several sequences at once, but each sandbox
episode also forks shells on a 4-core box. Concurrency is capped and each
episode gets its own sandbox root so parallel episodes cannot see each other's
files.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..agent import AgentConfig, SandboxAgent
from ..sandbox import Capabilities
from .datasets import Task
from .scoring import grade


@dataclass
class RunResult:
    task_id: str
    benchmark: str
    domain: str
    mode: str
    model: str
    caps: str
    correct: bool
    predicted: str | None
    gold: str
    generated_tokens: int
    prompt_tokens: int
    n_turns: int
    wall_s: float
    stop_reason: str
    error: str | None = None
    sandbox_stats: dict[str, Any] | None = None


def _completed_keys(path: Path) -> set[str]:
    """Keys already present in the results file, so a rerun can skip them."""
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A partial final line from a killed run: ignore it, the task
                # will simply be redone.
                continue
            done.add(f"{row['task_id']}|{row['mode']}|{row['caps']}")
    return done


def run_suite(
    tasks: Iterable[Task],
    *,
    client: Any,
    config: AgentConfig,
    caps: Capabilities | None = None,
    out_dir: str | Path,
    sandbox_root: str | Path = "/content/runs/sandboxes",
    backend: str = "auto",
    workers: int = 4,
    resume: bool = True,
    progress: bool = True,
) -> list[RunResult]:
    """Run every task and append results to ``out_dir/results.jsonl``."""
    caps = caps or Capabilities()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)

    done = _completed_keys(results_path) if resume else set()
    todo = [t for t in tasks if f"{t.task_id}|{config.mode}|{caps.slug}" not in done]
    if progress:
        print(f"[{config.mode}/{caps.slug}] {len(todo)} to run, {len(done)} already done")

    write_lock = threading.Lock()
    results: list[RunResult] = []
    started = time.monotonic()

    def one(task: Task) -> tuple[RunResult, Any]:
        agent = SandboxAgent(client, config)
        safe_id = task.task_id.replace("/", "_")
        traj = agent.run(
            task.question,
            task_id=task.task_id,
            caps=caps,
            documents=task.documents,
            sandbox_root=str(Path(sandbox_root) / f"{config.mode}_{caps.slug}_{safe_id}"),
            backend=backend,
        )
        correct = grade(traj.final_answer, task.answer, task.kind)
        return RunResult(
            task_id=task.task_id,
            benchmark=task.benchmark,
            domain=task.domain,
            mode=config.mode,
            model=config.model,
            caps=caps.slug,
            correct=correct,
            predicted=traj.final_answer,
            gold=task.answer,
            generated_tokens=traj.generated_tokens,
            prompt_tokens=traj.prompt_tokens,
            n_turns=traj.n_turns,
            wall_s=traj.wall_s,
            stop_reason=traj.stop_reason,
            error=traj.error,
            sandbox_stats=traj.sandbox.get("stats"),
        ), traj

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, task): task for task in todo}
        for n, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                result, traj = future.result()
            except Exception as exc:  # noqa: BLE001 - never lose a sweep to one task
                result = RunResult(
                    task_id=task.task_id,
                    benchmark=task.benchmark,
                    domain=task.domain,
                    mode=config.mode,
                    model=config.model,
                    caps=caps.slug,
                    correct=False,
                    predicted=None,
                    gold=task.answer,
                    generated_tokens=0,
                    prompt_tokens=0,
                    n_turns=0,
                    wall_s=0.0,
                    stop_reason="harness_error",
                    error=f"{type(exc).__name__}: {exc}",
                )
                traj = None

            with write_lock:
                # Append-and-flush per task: the whole point is that a runtime
                # dying mid-sweep costs one episode, not the sweep.
                with results_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(asdict(result)) + "\n")
                    handle.flush()
                if traj is not None:
                    safe = task.task_id.replace("/", "_")
                    name = f"{config.mode}_{caps.slug}_{safe}.json"
                    (traj_dir / name).write_text(
                        json.dumps(traj.as_dict(), indent=1), encoding="utf-8"
                    )
                results.append(result)

            if progress and (n % 5 == 0 or n == len(todo)):
                acc = sum(r.correct for r in results) / max(len(results), 1)
                rate = n / max(time.monotonic() - started, 1e-9)
                print(
                    f"  {n}/{len(todo)}  acc={acc:.3f}  "
                    f"{rate * 60:.1f}/min  last={result.task_id}",
                    flush=True,
                )
    return results


def summarize(results_path: str | Path) -> dict[str, Any]:
    """Aggregate a results file into the paper's two headline numbers.

    Accuracy delta (Table 2) and the token-consumption ratio (Table 5), both
    broken down by domain.
    """
    rows = []
    with Path(results_path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Later rows win, so a re-run of a task supersedes its earlier attempt.
    unique: dict[str, dict] = {}
    for row in rows:
        unique[f"{row['task_id']}|{row['mode']}|{row['caps']}"] = row
    rows = list(unique.values())

    def agg(subset: list[dict]) -> dict[str, Any]:
        if not subset:
            return {"n": 0}
        return {
            "n": len(subset),
            "accuracy": sum(r["correct"] for r in subset) / len(subset),
            "mean_generated_tokens": sum(r["generated_tokens"] for r in subset) / len(subset),
            "mean_turns": sum(r["n_turns"] for r in subset) / len(subset),
            "mean_wall_s": sum(r["wall_s"] for r in subset) / len(subset),
            "errors": sum(1 for r in subset if r.get("error")),
        }

    out: dict[str, Any] = {"overall": {}, "by_domain": {}, "by_condition": {}}
    conditions = sorted({(r["mode"], r["caps"]) for r in rows})
    for mode, cap in conditions:
        key = f"{mode}/{cap}"
        subset = [r for r in rows if r["mode"] == mode and r["caps"] == cap]
        out["by_condition"][key] = agg(subset)
        for domain in sorted({r["domain"] for r in subset}):
            out["by_domain"].setdefault(domain, {})[key] = agg(
                [r for r in subset if r["domain"] == domain]
            )

    # The paper's headline: sandbox/full versus the direct baseline, on the
    # tasks both actually attempted. Comparing over different task sets would
    # confound the delta with which subset each mode happened to finish.
    base_key, sb_key = "direct/full", "sandbox/full"
    base = {r["task_id"] for r in rows if f"{r['mode']}/{r['caps']}" == base_key}
    sand = {r["task_id"] for r in rows if f"{r['mode']}/{r['caps']}" == sb_key}
    shared = base & sand
    if shared:
        b = [r for r in rows if f"{r['mode']}/{r['caps']}" == base_key and r["task_id"] in shared]
        s = [r for r in rows if f"{r['mode']}/{r['caps']}" == sb_key and r["task_id"] in shared]
        ba, sa = agg(b), agg(s)
        out["overall"] = {
            "n_paired": len(shared),
            "direct_accuracy": ba["accuracy"],
            "sandbox_accuracy": sa["accuracy"],
            "delta_pp": (sa["accuracy"] - ba["accuracy"]) * 100,
            "direct_mean_tokens": ba["mean_generated_tokens"],
            "sandbox_mean_tokens": sa["mean_generated_tokens"],
            "token_ratio_sandbox_over_direct": (
                sa["mean_generated_tokens"] / ba["mean_generated_tokens"]
                if ba["mean_generated_tokens"]
                else None
            ),
        }
    return out
