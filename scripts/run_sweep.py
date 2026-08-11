#!/usr/bin/env python
"""Run the main experiment: sandbox vs direct, across domains.

Both conditions run over the *same* sampled tasks with the same seed, so the
paired comparison in `summarize` is over identical items. Example:

    python scripts/run_sweep.py --benchmark mmlu_pro --n 40 \
        --model qwen3-4b --out runs/qwen3-4b-mmlu

Re-running the same command resumes: completed episodes are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI

from sandbox_lab.agent import AgentConfig
from sandbox_lab.evals import load_suite, run_suite, summarize
from sandbox_lab.sandbox import Capabilities


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", default="mmlu_pro", choices=["mmlu_pro", "math500", "aime"])
    p.add_argument("--n", type=int, default=40, help="items per domain (mmlu_pro) or total")
    p.add_argument("--domains", nargs="*", default=None)
    p.add_argument("--model", default="qwen3-4b")
    p.add_argument(
        "--base-url",
        nargs="+",
        default=["http://127.0.0.1:8000/v1"],
        help="one or more model servers; episodes are assigned stickily across them",
    )
    p.add_argument("--modes", nargs="+", default=["direct", "sandbox"])
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--max-tokens-per-turn", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--backend", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--caps",
        default="full",
        help="full | noexec | nofile | nonet - which meta-capabilities to enable",
    )
    p.add_argument("--out", default="runs/latest")
    p.add_argument("--sandbox-root", default="/content/runs/sandboxes")
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args(argv)


def caps_from_slug(slug: str) -> Capabilities:
    caps = Capabilities()
    for token in slug.split("+"):
        token = token.strip().lower()
        if token in ("", "full"):
            continue
        if token == "noexec":
            caps.code_execution = False
        elif token == "nofile":
            caps.file_management = False
        elif token == "nonet":
            caps.external_resources = False
        else:
            raise SystemExit(f"unknown capability token {token!r}")
    return caps


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {"seed": args.seed}
    if args.benchmark == "mmlu_pro":
        kwargs["n_per_domain"] = args.n
        if args.domains:
            kwargs["domains"] = args.domains
    elif args.benchmark == "math500":
        kwargs["n"] = args.n
    else:
        kwargs = {"n": args.n}

    tasks = load_suite(args.benchmark, **kwargs)
    print(f"loaded {len(tasks)} tasks from {args.benchmark}")

    clients = [OpenAI(base_url=url, api_key="none") for url in args.base_url]
    print(f"using {len(clients)} model server(s): {', '.join(args.base_url)}")
    caps = caps_from_slug(args.caps)

    (out_dir / "config.json").write_text(
        json.dumps({**vars(args), "n_tasks": len(tasks)}, indent=2), encoding="utf-8"
    )

    for mode in args.modes:
        config = AgentConfig(
            model=args.model,
            mode=mode,
            max_turns=args.max_turns,
            max_tokens_per_turn=args.max_tokens_per_turn,
            temperature=args.temperature,
        )
        # The baseline has no sandbox, so capability ablations do not apply to
        # it; labelling it "full" keeps the paired comparison keys aligned.
        run_caps = caps if mode == "sandbox" else Capabilities()
        run_suite(
            tasks,
            client=clients,
            config=config,
            caps=run_caps,
            out_dir=out_dir,
            sandbox_root=args.sandbox_root,
            backend=args.backend,
            workers=args.workers,
            resume=not args.no_resume,
        )

    summary = summarize(out_dir / "results.jsonl")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
