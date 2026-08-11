"""Sandbox primitives: a persistent shell, a file editor, and isolation."""

from .backends import (
    Backend,
    BackendUnavailable,
    Capabilities,
    DockerBackend,
    UnshareBackend,
    default_rlimits,
    select_backend,
)
from .editor import EditorError, FileEditor, PathJail
from .sandbox import Sandbox, SandboxBudgetExceeded, SandboxStats
from .shell import PersistentShell, ShellError, ShellResult
from .tools import (
    BASH_TOOL,
    FILE_EDITOR_TOOL,
    FINISH_TOOL,
    ToolDispatcher,
    tools_for,
)

__all__ = [
    "BASH_TOOL",
    "FILE_EDITOR_TOOL",
    "FINISH_TOOL",
    "Backend",
    "BackendUnavailable",
    "Capabilities",
    "DockerBackend",
    "EditorError",
    "FileEditor",
    "PathJail",
    "PersistentShell",
    "Sandbox",
    "SandboxBudgetExceeded",
    "SandboxStats",
    "ShellError",
    "ShellResult",
    "ToolDispatcher",
    "UnshareBackend",
    "default_rlimits",
    "select_backend",
    "tools_for",
]
