"""Tool schemas and dispatch.

The paper gives the model exactly three tools - ``bash``, ``file_editor``,
``finish`` - and attributes its results to that minimality. We keep the same
three so the comparison holds.

Schema wording is part of the experiment, not decoration: it is the only place
the model learns that the shell is persistent, that the timeout is soft, and
that ``str_replace`` needs a unique match. A 4B model in particular reads these
descriptions far more literally than a frontier model does, so anything left
implicit will be got wrong.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .backends import Capabilities
from .sandbox import Sandbox

BASH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Execute a bash command in a persistent shell session. State carries "
            "over between calls: the working directory, environment variables, "
            "and shell functions you set will still be there next call.\n"
            "Commands are killed after 10 seconds, but the session survives - a "
            "timeout costs you the command, not your progress. For anything "
            "slow, redirect to a log and poll it:\n"
            "  nohup python train.py > /tmp/train.log 2>&1 &\n"
            "  tail -5 /tmp/train.log\n"
            "You may install packages (pip install ...) and fetch resources from "
            "the network if this environment allows it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run.",
                }
            },
            "required": ["command"],
        },
    },
}

FILE_EDITOR_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "file_editor",
        "description": (
            "View and edit files.\n"
            "- view: show a file with line numbers, or list a directory. Optional "
            "view_range [start, end]; end=-1 means end of file.\n"
            "- create: write a file, creating parent directories. Overwrites.\n"
            "- str_replace: replace old_str with new_str. old_str must match the "
            "file EXACTLY, including indentation, and must appear exactly once - "
            "include surrounding lines to disambiguate.\n"
            "- insert: insert new_str after line insert_line (0 = at the top)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert"],
                },
                "path": {"type": "string", "description": "Absolute path."},
                "file_text": {"type": "string", "description": "For create: full contents."},
                "old_str": {"type": "string", "description": "For str_replace: exact text to find."},
                "new_str": {
                    "type": "string",
                    "description": "For str_replace: replacement. For insert: text to insert.",
                },
                "insert_line": {"type": "integer", "description": "For insert: line to insert after."},
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "For view: [start, end] line range.",
                },
            },
            "required": ["command", "path"],
        },
    },
}

FINISH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": (
            "Call this when you have the final answer, or when you are certain "
            "you cannot make further progress. This ends the episode."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": (
                        "The final answer, alone - no working, no restatement of "
                        "the question. For multiple choice, the letter only."
                    ),
                }
            },
            "required": ["answer"],
        },
    },
}


def tools_for(caps: Capabilities) -> list[dict[str, Any]]:
    """The tool list for a capability configuration.

    A disabled capability means the tool is *absent*, not present-and-refusing.
    Leaving a tool visible but erroring teaches the model mid-episode that the
    environment is broken, which changes its behaviour in ways that would
    contaminate the ablation.
    """
    tools = []
    if caps.code_execution:
        tools.append(BASH_TOOL)
    # file_editor is always offered, even when file management is disabled:
    # `view` must survive so the model can still *read* the task's own context
    # documents. Removing all file access would be a different intervention than
    # removing file *management*, and the write commands are refused by the
    # sandbox anyway.
    tools.append(FILE_EDITOR_TOOL)
    tools.append(FINISH_TOOL)
    return tools


class ToolDispatcher:
    """Routes a parsed tool call to the sandbox and returns an observation."""

    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox
        self.finished = False
        self.final_answer: str | None = None
        self._handlers: dict[str, Callable[..., str]] = {
            "bash": self._bash,
            "file_editor": self._file_editor,
            "finish": self._finish,
        }

    def dispatch(self, name: str, args: dict[str, Any]) -> str:
        handler = self._handlers.get(name)
        if handler is None:
            return (
                f"ERROR: no tool named {name!r}. Available: "
                f"{', '.join(sorted(self._handlers))}."
            )
        try:
            return handler(**args)
        except TypeError as exc:
            # Wrong/extra kwargs from a shaky small model: recoverable, so report
            # the signature rather than crashing the episode.
            return f"ERROR: bad arguments for {name!r}: {exc}"

    def _bash(self, command: str = "", **extra: Any) -> str:
        if not command.strip():
            return "ERROR: 'command' is required and must be non-empty."
        return self.sandbox.bash(command)

    def _file_editor(self, command: str = "", **kwargs: Any) -> str:
        if not command:
            return "ERROR: 'command' is required (view/create/str_replace/insert)."
        return self.sandbox.file_editor(command, **kwargs)

    def _finish(self, answer: str = "", **extra: Any) -> str:
        self.finished = True
        self.final_answer = answer.strip()
        return "Episode finished."
