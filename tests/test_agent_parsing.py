"""Tool-call parsing, including the recovery paths.

Every fallback here exists because a real model did the thing. Without them the
failures land as "the model stalled", which reads as a capability result when it
is really a harness artefact - and one that only ever penalises the sandbox mode,
since the baseline never emits tool calls at all.
"""

from __future__ import annotations

import json

from sandbox_lab.agent import AgentConfig, SandboxAgent


def calls_for(content: str | None, tool_calls: list | None = None) -> list[dict]:
    agent = SandboxAgent(client=None, config=AgentConfig())
    return agent._extract_calls({"content": content, "tool_calls": tool_calls or []})


def test_native_tool_calls_are_used_directly():
    calls = calls_for(
        None,
        [{"id": "c1", "name": "bash", "arguments": json.dumps({"command": "ls"})}],
    )
    assert calls == [{"id": "c1", "name": "bash", "arguments": {"command": "ls"}}]


def test_native_calls_win_over_text():
    """If the parser produced a call, prose is not re-parsed on top of it."""
    calls = calls_for(
        "finish(999)",
        [{"id": "c1", "name": "bash", "arguments": '{"command": "ls"}'}],
    )
    assert len(calls) == 1
    assert calls[0]["name"] == "bash"


def test_recovers_tool_call_left_in_content():
    """vLLM's hermes parser sometimes misses a call and leaves the raw block."""
    content = 'Let me check.\n<tool_call>\n{"name": "bash", "arguments": {"command": "pwd"}}\n</tool_call>'
    calls = calls_for(content)
    assert calls[0]["name"] == "bash"
    assert calls[0]["arguments"] == {"command": "pwd"}


def test_recovers_prose_finish():
    """Observed on Qwen3-4B: it writes finish(849) instead of calling it."""
    content = "The remainder is 849.\n\nI verified this with Python.\n\nfinish(849)"
    calls = calls_for(content)
    assert calls == [{"id": "prosefinish_0", "name": "finish", "arguments": {"answer": "849"}}]


def test_recovers_prose_finish_variants():
    for text, expected in [
        ('finish("B")', "B"),
        ("finish(answer=42)", "42"),
        ("finish( 'x + 1' )", "x + 1"),
        ("Done. FINISH(7).", "7"),
    ]:
        calls = calls_for(text)
        assert calls and calls[0]["arguments"]["answer"] == expected, text


def test_does_not_invent_a_finish_from_mid_sentence_mention():
    """Only a trailing call counts; discussing `finish` must not end the episode."""
    assert calls_for("I will call finish(x) once I have verified the result.") == []
    assert calls_for("Now let me finish the calculation.") == []


def test_empty_finish_is_not_a_call():
    assert calls_for("finish()") == []


def test_no_calls_when_there_is_nothing_to_parse():
    assert calls_for("Just thinking out loud.") == []
    assert calls_for(None) == []


def test_bare_string_arguments_are_coerced_to_a_command():
    """Some models emit the command as a bare string rather than JSON."""
    calls = calls_for(None, [{"id": "c1", "name": "bash", "arguments": "ls -la"}])
    assert calls[0]["arguments"] == {"command": "ls -la"}


def test_malformed_json_in_text_block_is_skipped_not_fatal():
    assert calls_for("<tool_call>{not json}</tool_call>") == []
