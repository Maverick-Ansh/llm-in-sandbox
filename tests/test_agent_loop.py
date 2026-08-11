"""Agent loop tests against a stub model server.

The loop's bugs so far have all been in *accounting* rather than logic - tokens
not counted, turns not recorded, answers not stored - and none of them made
anything crash. They showed up only as slightly wrong numbers in a results
table, which is the hardest kind of bug to notice and the most damaging to a
measurement. These tests assert the accounting directly.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from sandbox_lab.agent import AgentConfig, SandboxAgent

pytestmark = pytest.mark.skipif(os.name != "posix", reason="sandbox needs Linux")


class StubClient:
    """Minimal stand-in for the OpenAI client, replaying scripted replies."""

    def __init__(self, replies: list[dict]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        spec = self.replies.pop(0) if self.replies else {"content": "no more replies"}
        tool_calls = [
            SimpleNamespace(
                id=f"c{i}",
                function=SimpleNamespace(name=t["name"], arguments=json.dumps(t["arguments"])),
            )
            for i, t in enumerate(spec.get("tool_calls", []))
        ]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=spec.get("content"), tool_calls=tool_calls or None
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=spec.get("prompt_tokens", 100),
                completion_tokens=spec.get("completion_tokens", 10),
            ),
        )


def run_sandbox(replies, tmp_path, **cfg):
    agent = SandboxAgent(StubClient(replies), AgentConfig(mode="sandbox", **cfg))
    return agent.run("What is 2+2?", task_id="t", sandbox_root=str(tmp_path), backend="local")


def test_finish_ends_the_episode_and_records_the_answer(tmp_path):
    traj = run_sandbox(
        [{"tool_calls": [{"name": "finish", "arguments": {"answer": "4"}}]}], tmp_path
    )
    assert traj.final_answer == "4"
    assert traj.stop_reason == "finished"
    assert traj.n_turns == 1


def test_bash_result_reaches_the_next_turn(tmp_path):
    traj = run_sandbox(
        [
            {"tool_calls": [{"name": "bash", "arguments": {"command": "echo 4"}}]},
            {"tool_calls": [{"name": "finish", "arguments": {"answer": "4"}}]},
        ],
        tmp_path,
    )
    assert traj.turns[0].observations == ["4"]
    assert traj.final_answer == "4"


def test_forced_answer_is_recorded_as_a_turn_and_its_tokens_counted(tmp_path):
    """The bug this exists to prevent.

    When the turn budget runs out the loop makes one more model call asking for
    an answer. Consuming that silently dropped its completion tokens from the
    episode total, understating sandbox token use and biasing the efficiency
    comparison in the sandbox's favour.
    """
    replies = [
        {"tool_calls": [{"name": "bash", "arguments": {"command": "true"}}],
         "completion_tokens": 10}
        for _ in range(3)
    ]
    replies.append({"content": "FINAL ANSWER: 4", "completion_tokens": 7})

    traj = run_sandbox(replies, tmp_path, max_turns=3)
    assert traj.stop_reason == "max_turns"
    assert traj.final_answer == "4"
    # 3 loop turns + the forced-answer turn.
    assert traj.n_turns == 4
    assert traj.generated_tokens == 3 * 10 + 7
    assert traj.turns[-1].observations == ["[forced final answer]"]


def test_stall_stops_early_instead_of_burning_the_budget(tmp_path):
    """Two consecutive no-tool-call turns end the episode.

    Measured on Qwen3-4B: it solved a task on turn 0 then emitted seven more
    turns of nothing. Those wasted tokens land in the sandbox arm's total.
    """
    replies = [{"content": "I am thinking about it.", "completion_tokens": 5}] * 10
    traj = run_sandbox(replies, tmp_path, max_turns=20)
    assert traj.stop_reason == "stalled"
    assert traj.n_turns <= 4, "must not run to the turn limit"


def test_top_level_tool_arguments_still_execute(tmp_path):
    """End-to-end version of the parser fix: the command must actually run."""
    replies = [
        {"content": '<tool_call>\n{"name": "bash", "command": "echo recovered"}\n</tool_call>'},
        {"tool_calls": [{"name": "finish", "arguments": {"answer": "ok"}}]},
    ]
    traj = run_sandbox(replies, tmp_path)
    assert traj.turns[0].observations == ["recovered"]


def test_direct_mode_gets_a_larger_one_shot_budget(tmp_path):
    """The baseline must not be handicapped by the agent's per-turn cap.

    Capping it at max_tokens_per_turn would manufacture the paper's efficiency
    result for free.
    """
    client = StubClient([{"content": "FINAL ANSWER: 4", "completion_tokens": 300}])
    cfg = AgentConfig(mode="direct", max_tokens_per_turn=2048)
    traj = SandboxAgent(client, cfg).run("What is 2+2?", task_id="t")
    assert traj.final_answer == "4"
    assert client.calls[0]["max_tokens"] > cfg.max_tokens_per_turn
    assert client.calls[0].get("tools") is None, "the baseline must have no tools"


def test_both_modes_count_tokens_the_same_way(tmp_path):
    """generated_tokens must mean the same thing in both arms, or the ratio lies."""
    direct = SandboxAgent(
        StubClient([{"content": "FINAL ANSWER: 4", "completion_tokens": 250}]),
        AgentConfig(mode="direct"),
    ).run("q", task_id="t")

    sandbox = run_sandbox(
        [
            {"tool_calls": [{"name": "bash", "arguments": {"command": "true"}}],
             "completion_tokens": 100},
            {"tool_calls": [{"name": "finish", "arguments": {"answer": "4"}}],
             "completion_tokens": 150},
        ],
        tmp_path,
    )
    assert direct.generated_tokens == 250
    assert sandbox.generated_tokens == 250


# --------------------------------------------- the prompt-variant experiment


def test_neutral_prompt_differs_only_in_the_reasoning_instruction():
    """The separating arm must change exactly one thing.

    The -32pp maths result could be the environment or the prompt. This variant
    isolates it, which only works if everything else is held constant: same
    tools, same timeout guidance, same `finish` instruction. Anything else that
    drifts becomes a second variable and the arm stops separating anything.
    """
    from sandbox_lab.agent.loop import (
        SANDBOX_NEUTRAL_SYSTEM_PROMPT as NEUTRAL,
    )
    from sandbox_lab.agent.loop import (
        SANDBOX_SYSTEM_PROMPT as ORIGINAL,
    )

    # The instruction under test is present in one and absent from the other.
    assert "rather than derive them in your head" in ORIGINAL
    assert "rather than derive them in your head" not in NEUTRAL
    assert "Prefer running code over long mental arithmetic" in ORIGINAL
    assert "Prefer running code over long mental arithmetic" not in NEUTRAL

    # Everything load-bearing is held constant.
    for shared in [
        "You are working inside a Linux computer environment.",
        "Commands time out after 10 seconds. That kills the command, not your session.",
        "nohup <cmd> > /tmp/out.log 2>&1 &",
        "call the `finish` tool with",
        "only counts as answered if",
        "Python with numpy/scipy is",
    ]:
        assert shared in ORIGINAL, shared
        assert shared in NEUTRAL, shared


def test_mode_selects_the_prompt(tmp_path):
    from sandbox_lab.agent.loop import SANDBOX_NEUTRAL_SYSTEM_PROMPT

    client = StubClient([{"tool_calls": [{"name": "finish", "arguments": {"answer": "4"}}]}])
    agent = SandboxAgent(client, AgentConfig(mode="sandbox_neutral"))
    traj = agent.run("q", task_id="t", sandbox_root=str(tmp_path), backend="local")

    assert traj.mode == "sandbox_neutral", "results must key on the arm, or it merges with sandbox"
    assert client.calls[0]["messages"][0]["content"] == SANDBOX_NEUTRAL_SYSTEM_PROMPT
    # The tools must be identical to the sandbox arm - only the prompt varies.
    assert {t["function"]["name"] for t in client.calls[0]["tools"]} == {
        "bash", "file_editor", "finish"
    }


def test_unknown_mode_is_rejected(tmp_path):
    agent = SandboxAgent(StubClient([]), AgentConfig(mode="sandbox_typo"))
    with pytest.raises(ValueError, match="unknown mode"):
        agent.run("q", task_id="t", sandbox_root=str(tmp_path), backend="local")
