"""The agent loop: model <-> sandbox, until an answer or the turn budget runs out.

Two run modes live here on purpose, sharing every line they can:

``sandbox``
    The paper's setting. The model gets bash + file_editor + finish and runs a
    multi-turn episode.
``direct``
    The control. Same model, same question, same answer extraction, no tools -
    one long chain of thought.

Sharing the prompt template and the answer parser between them is what makes
the comparison mean anything. If the baseline used a different extraction path,
any measured delta would partly be "we parse tool output better than prose",
which is not the claim under test.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..sandbox import Capabilities, Sandbox, ToolDispatcher, tools_for

SANDBOX_SYSTEM_PROMPT = """You are working inside a Linux computer environment.

You have a persistent bash shell and a file editor. Use them. You are expected \
to *compute* answers rather than derive them in your head: write a script, run \
it, and read the result. Verify numerically wherever a check is possible.

Guidance that matters:
- Prefer running code over long mental arithmetic. Python with numpy/scipy is \
available, and you may pip install more.
- Files persist. Write intermediate results to disk instead of holding them in \
your reasoning.
- Commands time out after 10 seconds. That kills the command, not your session. \
For slow work use: nohup <cmd> > /tmp/out.log 2>&1 &  then poll the log.
- Check your work. A result you have verified twice is worth more than three \
you have not.

Ending the episode: the moment you have the answer, call the `finish` tool with \
it. Do not restate it in prose and stop - an episode only counts as answered if \
`finish` was called. Every turn you spend after you already know the answer is \
wasted."""

DIRECT_SYSTEM_PROMPT = """You are a careful expert problem solver.

Think through the problem step by step, then give your final answer on the last \
line in exactly this form:

FINAL ANSWER: <answer>

The answer must stand alone - no working, no restatement of the question. For \
multiple choice, give the letter only."""

# Recognises the baseline's answer line, tolerating the formatting the model
# actually produces (bold, colons, spacing) rather than only the ideal form.
_FINAL_ANSWER_RE = re.compile(
    # The optional "is" matters: "The final answer is 42" is at least as common
    # as "FINAL ANSWER: 42", and without it the capture starts at "is 42".
    r"final\s*answer\s*(?:is)?\s*[:\-]?\s*\**\s*(.+?)\s*\**\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# Fallback for models that emit tool calls as text when native parsing misfires.
_TEXT_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# Small models frequently *write* `finish(849)` in prose instead of emitting a
# tool call, having clearly decided to finish. Recovering that is standard agent
# harness behaviour, and not recovering it would charge the sandbox mode extra
# turns for a formatting slip - a harness artefact that would show up as a
# capability difference. Deliberately narrow: only `finish`, only at the end of
# the message, because bash/file_editor arguments cannot be parsed unambiguously
# out of prose.
_PROSE_FINISH_RE = re.compile(
    r"\bfinish\s*\(\s*(?:answer\s*=\s*)?(.*?)\s*\)\s*\.?\s*$",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class AgentConfig:
    model: str = "qwen3-4b"
    mode: str = "sandbox"  # "sandbox" | "direct"
    max_turns: int = 30
    max_tokens_per_turn: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0
    # Observations older than this many turns are collapsed to a stub. Keeps a
    # 30-turn episode inside a 32k window without dropping the reasoning thread.
    keep_full_observations: int = 6
    max_observation_chars: int = 6000
    # Generous, because the baseline is deliberately given a large one-shot
    # budget: 8192 tokens on a T4 can exceed a 240s deadline, and 7 of 99
    # baseline episodes died that way on the first run. A timeout scores as
    # wrong, so a tight deadline silently penalises the arm that generates most.
    request_timeout_s: float = 900.0


@dataclass
class TurnRecord:
    index: int
    completion_tokens: int
    prompt_tokens: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    content: str = ""
    observations: list[str] = field(default_factory=list)
    latency_s: float = 0.0


@dataclass
class Trajectory:
    task_id: str
    mode: str
    model: str
    final_answer: str | None = None
    turns: list[TurnRecord] = field(default_factory=list)
    stop_reason: str = "unknown"
    wall_s: float = 0.0
    sandbox: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def generated_tokens(self) -> int:
        """Tokens the model *produced*.

        This is the number the paper's efficiency claim is about. Prompt tokens
        are tracked separately because they are dominated by re-sending the
        transcript each turn, which prefix caching makes nearly free in compute
        but which still shows up in a naive total.
        """
        return sum(t.completion_tokens for t in self.turns)

    @property
    def prompt_tokens(self) -> int:
        return sum(t.prompt_tokens for t in self.turns)

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "mode": self.mode,
            "model": self.model,
            "final_answer": self.final_answer,
            "stop_reason": self.stop_reason,
            "n_turns": self.n_turns,
            "generated_tokens": self.generated_tokens,
            "prompt_tokens": self.prompt_tokens,
            "wall_s": round(self.wall_s, 2),
            "sandbox": self.sandbox,
            "error": self.error,
            "turns": [
                {
                    "index": t.index,
                    "completion_tokens": t.completion_tokens,
                    "prompt_tokens": t.prompt_tokens,
                    "latency_s": round(t.latency_s, 2),
                    "content": t.content,
                    "tool_calls": t.tool_calls,
                    "observations": t.observations,
                }
                for t in self.turns
            ],
        }


class SandboxAgent:
    """Runs one task to completion in one of the two modes."""

    def __init__(self, client: Any, config: AgentConfig) -> None:
        self.client = client
        self.cfg = config

    # ------------------------------------------------------------------- run

    def run(
        self,
        question: str,
        *,
        task_id: str = "task",
        caps: Capabilities | None = None,
        documents: dict[str, str] | None = None,
        sandbox_root: str = "/testbed",
        backend: str = "auto",
    ) -> Trajectory:
        if self.cfg.mode == "direct":
            return self._run_direct(question, task_id=task_id)
        return self._run_sandbox(
            question,
            task_id=task_id,
            caps=caps or Capabilities(),
            documents=documents or {},
            sandbox_root=sandbox_root,
            backend=backend,
        )

    # ---------------------------------------------------------------- direct

    def _run_direct(self, question: str, *, task_id: str) -> Trajectory:
        traj = Trajectory(task_id=task_id, mode="direct", model=self.cfg.model)
        started = time.monotonic()
        messages = [
            {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        try:
            # The baseline gets the full remaining window in one shot: capping it
            # at the per-turn budget would handicap the control and manufacture
            # the paper's efficiency result for free.
            reply, usage, latency = self._complete(
                messages, tools=None, max_tokens=self.cfg.max_tokens_per_turn * 4
            )
            content = reply.get("content") or ""
            traj.turns.append(
                TurnRecord(
                    index=0,
                    completion_tokens=usage["completion_tokens"],
                    prompt_tokens=usage["prompt_tokens"],
                    content=content,
                    latency_s=latency,
                )
            )
            traj.final_answer = extract_final_answer(content)
            traj.stop_reason = "answered" if traj.final_answer else "no_answer_parsed"
        except Exception as exc:  # noqa: BLE001 - one bad task must not kill a sweep
            traj.error = f"{type(exc).__name__}: {exc}"
            traj.stop_reason = "error"
        traj.wall_s = time.monotonic() - started
        return traj

    # --------------------------------------------------------------- sandbox

    def _run_sandbox(
        self,
        question: str,
        *,
        task_id: str,
        caps: Capabilities,
        documents: dict[str, str],
        sandbox_root: str,
        backend: str,
    ) -> Trajectory:
        traj = Trajectory(task_id=task_id, mode="sandbox", model=self.cfg.model)
        started = time.monotonic()
        tools = tools_for(caps)

        sandbox = Sandbox(sandbox_root, caps=caps, backend=backend)
        try:
            sandbox.start()
            for name, content in documents.items():
                sandbox.write_document(name, content)

            dispatcher = ToolDispatcher(sandbox)
            stalled_turns = 0
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SANDBOX_SYSTEM_PROMPT},
                {"role": "user", "content": self._user_message(question, documents, sandbox)},
            ]

            for turn in range(self.cfg.max_turns):
                reply, usage, latency = self._complete(
                    self._compress(messages), tools=tools, max_tokens=self.cfg.max_tokens_per_turn
                )
                calls = self._extract_calls(reply)
                record = TurnRecord(
                    index=turn,
                    completion_tokens=usage["completion_tokens"],
                    prompt_tokens=usage["prompt_tokens"],
                    content=reply.get("content") or "",
                    tool_calls=[{"name": c["name"], "arguments": c["arguments"]} for c in calls],
                    latency_s=latency,
                )

                if not calls:
                    # No tool call: either it answered in prose, or it stalled.
                    # Small models routinely solve the task, narrate the answer,
                    # and then forget `finish` - left alone they burn every
                    # remaining turn repeating themselves, which both wastes
                    # compute and inflates the very token count we are measuring.
                    # So: accept a well-formed answer, nudge once, and give up on
                    # the second consecutive miss.
                    messages.append({"role": "assistant", "content": record.content})
                    parsed = extract_final_answer(record.content)
                    if parsed:
                        traj.turns.append(record)
                        traj.final_answer = parsed
                        traj.stop_reason = "answered_in_prose"
                        break

                    stalled_turns += 1
                    record.observations.append("[no tool call]")
                    traj.turns.append(record)
                    if stalled_turns >= 2:
                        traj.final_answer = self._force_answer(messages)
                        traj.stop_reason = "stalled"
                        break
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You did not call a tool. If you already have the "
                                "answer, call `finish` with it now. Otherwise call "
                                "`bash` or `file_editor` to continue."
                            ),
                        }
                    )
                    continue
                stalled_turns = 0

                messages.append(self._assistant_message(record.content, calls))
                for call in calls:
                    observation = dispatcher.dispatch(call["name"], call["arguments"])
                    observation = self._clip(observation)
                    record.observations.append(observation)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": observation,
                        }
                    )
                traj.turns.append(record)

                if dispatcher.finished:
                    traj.final_answer = dispatcher.final_answer
                    traj.stop_reason = "finished"
                    break
            else:
                # Out of turns. Ask once for the answer instead of scoring a
                # zero: "ran out of turns" and "got it wrong" are different
                # failures and collapsing them hides where the loss comes from.
                traj.final_answer = self._force_answer(messages)
                traj.stop_reason = "max_turns"

            traj.sandbox = sandbox.manifest()
        except Exception as exc:  # noqa: BLE001
            traj.error = f"{type(exc).__name__}: {exc}"
            traj.stop_reason = "error"
            try:
                traj.sandbox = sandbox.manifest()
            except Exception:  # noqa: BLE001, S110
                pass
        finally:
            sandbox.close()

        traj.wall_s = time.monotonic() - started
        return traj

    def _force_answer(self, messages: list[dict[str, Any]]) -> str | None:
        try:
            reply, _, _ = self._complete(
                [
                    *self._compress(messages),
                    {
                        "role": "user",
                        "content": (
                            "You are out of turns. State your single best answer now, "
                            "on one line, as: FINAL ANSWER: <answer>"
                        ),
                    },
                ],
                tools=None,
                max_tokens=256,
            )
            return extract_final_answer(reply.get("content") or "")
        except Exception:  # noqa: BLE001
            return None

    # -------------------------------------------------------------- plumbing

    def _user_message(self, question: str, documents: dict[str, str], sandbox: Sandbox) -> str:
        parts = [f"Your working directory is {sandbox.root}."]
        if documents:
            names = ", ".join(sorted(documents))
            parts.append(
                f"Task materials are in {sandbox.root}/documents/ ({names}). "
                "Read them from disk - they are not reproduced here."
            )
        parts.append("")
        parts.append(question)
        return "\n".join(parts)

    def _complete(
        self, messages: list[dict[str, Any]], *, tools: list[dict] | None, max_tokens: int
    ) -> tuple[dict[str, Any], dict[str, int], float]:
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "timeout": self.cfg.request_timeout_s,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        started = time.monotonic()
        response = self.client.chat.completions.create(**kwargs)
        latency = time.monotonic() - started

        choice = response.choices[0].message
        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
        }
        reply = {
            "content": choice.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in (choice.tool_calls or [])
            ],
        }
        return reply, usage, latency

    def _extract_calls(self, reply: dict[str, Any]) -> list[dict[str, Any]]:
        """Normalise tool calls, falling back to text parsing.

        vLLM's hermes parser occasionally misses a call and leaves the raw
        ``<tool_call>`` block in the content - more often with small models. A
        run that silently scores those as "no tool call" measures the parser,
        not the model, so we recover them.
        """
        calls: list[dict[str, Any]] = []
        for index, raw in enumerate(reply.get("tool_calls") or []):
            args = raw["arguments"]
            calls.append(
                {
                    "id": raw["id"] or f"call_{index}",
                    "name": raw["name"],
                    "arguments": _coerce_args(args),
                }
            )
        if calls:
            return calls

        content = reply.get("content") or ""
        for index, blob in enumerate(_TEXT_TOOL_CALL_RE.findall(content)):
            try:
                parsed = json.loads(blob)
            except json.JSONDecodeError:
                continue
            name = parsed.get("name")
            if not name:
                continue
            calls.append(
                {
                    "id": f"textcall_{index}",
                    "name": name,
                    "arguments": _coerce_args(parsed.get("arguments", {})),
                }
            )
        if calls:
            return calls

        prose = _PROSE_FINISH_RE.search(content.strip())
        if prose:
            answer = prose.group(1).strip().strip("\"'").strip()
            if answer:
                calls.append(
                    {"id": "prosefinish_0", "name": "finish", "arguments": {"answer": answer}}
                )
        return calls

    @staticmethod
    def _assistant_message(content: str, calls: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": content or None,
            "tool_calls": [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {
                        "name": c["name"],
                        "arguments": json.dumps(c["arguments"]),
                    },
                }
                for c in calls
            ],
        }

    def _clip(self, observation: str) -> str:
        limit = self.cfg.max_observation_chars
        if len(observation) <= limit:
            return observation
        head, tail = observation[: limit // 2], observation[-limit // 2 :]
        dropped = len(observation) - limit
        return f"{head}\n[... {dropped} chars elided ...]\n{tail}"

    def _compress(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Stub out old tool observations to keep the window from overflowing.

        Assistant reasoning is never touched - that is the thread the model is
        following. Old *observations* are the compressible part: a directory
        listing from turn 2 is rarely load-bearing at turn 25, while the
        decision it prompted is already recorded in the assistant turn.
        """
        keep = self.cfg.keep_full_observations
        tool_positions = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        if len(tool_positions) <= keep:
            return messages
        stale = set(tool_positions[:-keep])
        out = []
        for i, message in enumerate(messages):
            if i in stale and len(message.get("content") or "") > 200:
                content = message["content"]
                out.append(
                    {
                        **message,
                        "content": content[:200] + f"\n[... {len(content) - 200} chars elided from an earlier turn ...]",
                    }
                )
            else:
                out.append(message)
        return out


def _coerce_args(args: Any) -> dict[str, Any]:
    """Tool arguments arrive as a JSON string, a dict, or malformed JSON."""
    if isinstance(args, dict):
        return args
    if not isinstance(args, str):
        return {}
    try:
        parsed = json.loads(args)
    except json.JSONDecodeError:
        # Small models sometimes emit a bare command string instead of JSON.
        # Treating it as the `command` argument recovers the turn.
        return {"command": args} if args.strip() else {}
    return parsed if isinstance(parsed, dict) else {}


def extract_final_answer(content: str) -> str | None:
    """Pull the answer out of a prose completion.

    Used identically by both modes so neither is advantaged by parsing.
    """
    if not content:
        return None
    matches = _FINAL_ANSWER_RE.findall(content)
    if matches:
        return _strip_answer_label(matches[-1])
    boxed = re.findall(r"\\boxed\{([^}]*)\}", content)
    if boxed:
        return _strip_answer_label(boxed[-1])
    return _terse_answer(content)


# An option letter delivered as an answer: "(D) ...", "F. ...", "A: ...", or a
# bare "C". Every form requires a bracket or delimiter, or that the letter is
# the entire line. That requirement is the whole safety property - without it
# any reasoning line starting with a capital A-J word ("Before concluding...")
# would be harvested as that letter.
_TERSE_CHOICE_RE = re.compile(r"^(?:\(\s*([A-J])\s*\)|([A-J])\s*[.):]|([A-J])\s*$)")

# A short answer-shaped token: no whitespace, so numbers, fractions and simple
# expressions qualify while prose does not.
_TERSE_VALUE_RE = re.compile(r"^[^\s]{1,40}$")


def _terse_answer(content: str) -> str | None:
    """Accept an answer given without the requested label.

    Measured on a real run: 17 of 99 baseline episodes produced completions like
    ``C`` or ``F. The population of SAT scores...`` - two to fifteen tokens - and
    were scored as having no answer at all. The model was not failing; it was
    obeying the benchmark's own instruction ("Answer with the letter of the
    correct option only"), which overrides the system prompt's request for a
    ``FINAL ANSWER:`` line.

    Refusing those penalises the baseline specifically: the sandbox arm delivers
    its answer through the ``finish`` tool and is never affected. Left in place
    the bug would have manufactured a large sandbox advantage out of formatting.

    The fallback is deliberately conservative - it recognises answer *shapes*,
    not "short text". Accepting any short line would turn "Before concluding,
    check the units" into an answer, which is a worse failure than the one being
    fixed because it grades silently rather than visibly.
    """
    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
    if not lines:
        return None
    last = lines[-1].strip("*").strip()
    if len(last) > 200:
        return None

    choice = _TERSE_CHOICE_RE.match(last)
    if choice:
        return next(group for group in choice.groups() if group)

    # A bare value, only when it is the model's entire reply - a value on the
    # last line of a long derivation is more likely an intermediate result.
    if len(lines) == 1 and _TERSE_VALUE_RE.match(last):
        return last
    return None


def _strip_answer_label(text: str) -> str:
    """Remove any leftover 'FINAL ANSWER:' label from a captured answer.

    Models re-state the label inside the answer often enough ("FINAL ANSWER:
    FINAL ANSWER: 849", or a bolded label the capture group swallowed) that
    leaving it in silently grades correct answers as wrong. Applied to both
    modes so neither is advantaged.
    """
    out = text.strip().strip("*").strip()
    for _ in range(3):  # bounded: strip repeated labels without looping forever
        stripped = re.sub(
            r"^\**\s*(the\s+)?final\s*answer\s*(is)?\s*[:\-]?\s*\**\s*",
            "",
            out,
            flags=re.IGNORECASE,
        ).strip()
        if stripped == out:
            break
        out = stripped
    return out.strip().strip("*").strip()
