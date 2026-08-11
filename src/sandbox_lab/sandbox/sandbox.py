"""The sandbox an episode runs in: one shell, one editor, one root, one budget."""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .backends import Backend, Capabilities, select_backend
from .editor import EditorError, FileEditor
from .shell import PersistentShell, ShellError, ShellResult

# The paper's soft timeout. Kept as the default so runs are comparable.
DEFAULT_COMMAND_TIMEOUT_S = 10.0

# Hard ceiling on wall-clock for one episode. A 30-turn episode where every
# command burns its full timeout is ~5 minutes of pure shell; anything past this
# is a pathological loop, and on a shared T4 that starves every other episode.
DEFAULT_EPISODE_BUDGET_S = 900.0


@dataclass
class SandboxStats:
    """Per-episode telemetry. Written next to the trajectory for analysis."""

    commands: int = 0
    edits: int = 0
    timeouts: int = 0
    failed_commands: int = 0
    shell_restarts: int = 0
    shell_seconds: float = 0.0
    output_bytes: int = 0

    def as_dict(self) -> dict[str, float | int]:
        return dict(self.__dict__)


class SandboxBudgetExceeded(RuntimeError):
    """The episode ran past its wall-clock budget."""


class Sandbox:
    """A disposable computer for one episode.

    Layout mirrors the paper: the agent works in ``/testbed``, and task context
    files land in ``/testbed/documents/``.

    The sandbox owns three things the agent cannot escape: the path jail (via
    :class:`FileEditor`), the isolation backend (namespaces or a container), and
    the wall-clock budget. Everything else is the agent's problem.
    """

    def __init__(
        self,
        root: str | Path = "/testbed",
        *,
        caps: Capabilities | None = None,
        backend: str = "auto",
        command_timeout_s: float = DEFAULT_COMMAND_TIMEOUT_S,
        episode_budget_s: float = DEFAULT_EPISODE_BUDGET_S,
        max_output_bytes: int = 16_000,
        session_id: str | None = None,
    ) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.root = Path(root) / self.session_id if Path(root).name == "sessions" else Path(root)
        self.caps = caps or Capabilities()
        self.command_timeout_s = command_timeout_s
        self.episode_budget_s = episode_budget_s
        self.max_output_bytes = max_output_bytes

        self.stats = SandboxStats()
        self._backend: Backend | None = None
        self._backend_pref = backend
        self._shell: PersistentShell | None = None
        self._editor: FileEditor | None = None
        self._started_at: float | None = None

    # ----------------------------------------------------------------- setup

    def start(self) -> "Sandbox":
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "documents").mkdir(exist_ok=True)

        self._backend = select_backend(self.root, self.caps, prefer=self._backend_pref)
        self._editor = FileEditor(self.root)

        if self.caps.code_execution:
            self._shell = PersistentShell(
                self.root,
                argv_prefix=self._backend.argv_prefix(),
                rlimits=self._backend.rlimits,
                default_timeout_s=self.command_timeout_s,
                max_output_bytes=self.max_output_bytes,
            )
            self._shell.start()
            setup = self._backend.setup_script()
            if setup:
                self._shell.run(setup, timeout_s=30.0)
        self._started_at = time.monotonic()
        return self

    @property
    def backend_name(self) -> str:
        return self._backend.describe() if self._backend else "unstarted"

    @property
    def elapsed_s(self) -> float:
        return 0.0 if self._started_at is None else time.monotonic() - self._started_at

    def _check_budget(self) -> None:
        if self.elapsed_s > self.episode_budget_s:
            raise SandboxBudgetExceeded(
                f"episode exceeded its {self.episode_budget_s:.0f}s wall-clock budget"
            )

    # ------------------------------------------------------------------ tools

    def bash(self, command: str, timeout_s: float | None = None) -> str:
        """Run ``command``; return the observation string shown to the model."""
        if not self.caps.code_execution or self._shell is None:
            return "ERROR: code execution is disabled in this environment."
        self._check_budget()

        result = self._shell.run(command, timeout_s=timeout_s)
        self.stats.commands += 1
        self.stats.shell_seconds += result.duration_s
        self.stats.output_bytes += result.total_bytes
        if result.timed_out:
            self.stats.timeouts += 1
        elif not result.ok:
            self.stats.failed_commands += 1

        rendered = self._render(result)
        if result.shell_died:
            # A stray `exit` (or a crash) would otherwise brick the rest of the
            # episode with an unexplained error on every later command. Rebuild
            # the session and say so - files on disk are untouched.
            self.stats.shell_restarts += 1
            try:
                self._shell.restart()
                rendered += (
                    "\n[the shell session exited and has been restarted. Files on "
                    "disk are intact, but your working directory, environment "
                    "variables and shell functions were reset.]"
                )
            except ShellError as exc:
                rendered += f"\n[the shell session exited and could not be restarted: {exc}]"
        return rendered

    def _render(self, result: ShellResult) -> str:
        """Turn a ShellResult into the text the model sees.

        Two deliberate choices:

        * A successful command with no output returns an explicit marker rather
          than an empty string. Models reliably misread "" as a tool failure and
          retry, burning a turn.
        * The exit code is only surfaced when non-zero. Printing "exit 0" on
          every call is pure token overhead across a 30-turn episode.
        """
        body = result.output
        parts = []
        if result.timed_out:
            parts.append(
                f"[command interrupted after {result.duration_s:.1f}s - it exceeded the "
                f"{self.command_timeout_s:.0f}s limit. The shell session is still alive "
                "and your working directory is unchanged. Run long jobs in the "
                "background with nohup and poll a log file.]"
            )
        elif result.exit_code not in (0, None):
            parts.append(f"[exit code {result.exit_code}]")
        if body:
            parts.append(body)
        elif not result.timed_out:
            parts.append("[no output]")
        return "\n".join(parts)

    def file_editor(self, command: str, **kwargs: object) -> str:
        """Dispatch a ``file_editor`` call, converting errors into observations.

        Editor errors are returned as text, not raised. A malformed edit is a
        normal event in an agent episode and the model's job is to correct it;
        crashing the episode would throw away a recoverable trajectory.
        """
        if self._editor is None:
            return "ERROR: sandbox not started."
        if not self.caps.file_management and command != "view":
            return "ERROR: file management is disabled in this environment (read-only)."
        self._check_budget()
        try:
            out = self._editor(command, **kwargs)
            if command != "view":
                self.stats.edits += 1
            return out
        except EditorError as exc:
            return f"ERROR: {exc}"
        except TypeError as exc:
            return f"ERROR: bad arguments for {command!r}: {exc}"
        except OSError as exc:
            return f"ERROR: {exc}"

    # ------------------------------------------------------------- materials

    def write_document(self, name: str, content: str) -> Path:
        """Place a task context file in ``documents/``, as the paper's RL setup does."""
        path = self.root / "documents" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def snapshot(self, dest: str | Path) -> Path:
        """Copy the sandbox root out for post-hoc inspection.

        Worth doing on every episode: the files an agent leaves behind are often
        the clearest evidence of *how* it solved (or failed) a task, and they are
        gone the moment the sandbox is torn down.
        """
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(
            self.root, dest, ignore=shutil.ignore_patterns(".sandbox_lab", "__pycache__")
        )
        return dest

    def manifest(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "root": str(self.root),
            "backend": self.backend_name,
            "capabilities": {
                "code_execution": self.caps.code_execution,
                "file_management": self.caps.file_management,
                "external_resources": self.caps.external_resources,
            },
            "command_timeout_s": self.command_timeout_s,
            "stats": self.stats.as_dict(),
        }

    # ------------------------------------------------------------- teardown

    def close(self) -> None:
        if self._shell is not None:
            self._shell.close()
            self._shell = None

    def __enter__(self) -> "Sandbox":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<Sandbox {self.session_id} root={self.root} backend={self.backend_name}>"
