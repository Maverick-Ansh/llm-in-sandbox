"""A persistent, interruptible shell session backed by a real PTY.

Why a PTY and not plain pipes
-----------------------------
The paper's ``bash`` tool is specified as "a persistent shell session" with a
10-second *soft* timeout. "Soft" is the load-bearing word: when a command runs
too long we must interrupt *that command* and leave the session alive, with its
cwd, exported variables, and shell functions intact. A naive implementation
kills the shell and silently resets the agent's state mid-episode.

Doing this correctly needs the kernel's terminal line discipline:

* ``set -m`` (job control) makes bash put each command in its own process
  group and install it as the PTY's *foreground* process group.
* Writing ``\\x03`` (Ctrl-C) to the PTY master makes the tty driver deliver
  SIGINT to that foreground group only.
* bash is the session leader in a *different* process group, so it survives.

That is exactly what a human pressing Ctrl-C in a terminal does, and it is why
this class is a PTY rather than three pipes.

Command framing
---------------
Commands are written to a file and ``source``-d rather than piped into stdin as
text. Piping text breaks the moment a command contains a heredoc: the sentinel
line we append to detect completion gets swallowed as heredoc content and the
read loop hangs until the timeout. ``source``-ing a file makes heredocs,
multi-line strings, and quoting behave exactly as they would in a script, while
still running in the *current* shell so ``cd`` and ``export`` persist.
"""

from __future__ import annotations

import os
import re
import secrets
import select
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

IS_POSIX = os.name == "posix"

if IS_POSIX:  # pragma: no branch - import guard
    import fcntl
    import pty
    import termios


# Read chunk for the PTY master. 64 KiB keeps syscall count low on chatty
# commands without holding a large buffer per read.
_READ_CHUNK = 65536

# How long to wait after Ctrl-C for the command to actually die before we
# escalate to SIGKILL on the foreground process group.
_INTERRUPT_GRACE_S = 2.0

# CSI / OSC terminal control sequences.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07")


class ShellError(RuntimeError):
    """The shell session itself failed (not the command run inside it)."""


@dataclass
class ShellResult:
    """Outcome of a single ``run()``.

    ``exit_code`` is ``None`` exactly when the command was interrupted, since a
    command that never finished has no status to report.
    """

    output: str
    exit_code: int | None
    duration_s: float
    timed_out: bool = False
    truncated: bool = False
    total_bytes: int = 0
    # True when the command ended the session itself (`exit`, or a crash).
    # Commands are sourced into the live shell, so `exit` is not sandboxed away
    # from it - exactly as in a real terminal. The caller decides whether to
    # restart; silently returning exit_code=None would look like a timeout.
    shell_died: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class _HeadTailBuffer:
    """Accumulates bytes but only retains a bounded head and tail.

    A command like ``yes`` or a runaway training log can emit gigabytes before
    the timeout fires. We must keep *draining* the PTY (a full pipe buffer would
    block the child and stop it from ever reaching the sentinel) while refusing
    to hold it all in memory. Keeping the head and the tail preserves the two
    regions that actually carry signal: what the command started doing, and how
    it ended.
    """

    # Sliding window scanned for the completion sentinel. Must be comfortably
    # larger than a sentinel line so one can never straddle the window edge.
    _RECENT_CAP = 8192

    def __init__(self, head_bytes: int, tail_bytes: int) -> None:
        self._head_cap = head_bytes
        self._tail_cap = tail_bytes
        self._head = bytearray()
        self._tail = bytearray()
        # The newest bytes, regardless of which retention buffer they landed in.
        # Searching `_tail` for the sentinel would be wrong: `_tail` stays empty
        # until `_head` fills, so short commands - i.e. almost all of them -
        # would never match and every one would time out.
        self._recent = bytearray()
        self.total = 0

    def feed(self, data: bytes) -> None:
        self.total += len(data)
        self._recent += data
        if len(self._recent) > self._RECENT_CAP:
            del self._recent[: len(self._recent) - self._RECENT_CAP]

        room = self._head_cap - len(self._head)
        if room > 0:
            self._head += data[:room]
            data = data[room:]
            if not data:
                return
        self._tail += data
        if len(self._tail) > self._tail_cap:
            del self._tail[: len(self._tail) - self._tail_cap]

    @property
    def truncated(self) -> bool:
        return self.total > len(self._head) + len(self._tail)

    def recent_text(self, encoding: str) -> str:
        """Decode the newest bytes - used to hunt for the completion sentinel."""
        return self._recent.decode(encoding, errors="replace")

    def render(self, encoding: str) -> str:
        head = self._head.decode(encoding, errors="replace")
        if not self.truncated:
            return head + self._tail.decode(encoding, errors="replace")
        dropped = self.total - len(self._head) - len(self._tail)
        tail = self._tail.decode(encoding, errors="replace")
        return f"{head}\n[... {dropped} bytes of output truncated ...]\n{tail}"


class PersistentShell:
    """A long-lived ``bash`` process whose state survives across commands.

    Parameters
    ----------
    cwd:
        Working directory the shell starts in.
    env:
        Full environment for the shell. If ``None``, inherits the parent's.
    argv_prefix:
        Wrapper argv placed *before* ``bash``. This is the seam isolation
        backends hook into - e.g. ``["unshare", "--fork", "--pid", "--net"]``
        to run the shell inside fresh PID and network namespaces. Keeping this
        as plain argv means the shell class knows nothing about isolation.
    rlimits:
        ``{resource.RLIMIT_*: (soft, hard)}`` applied in the child before exec.
        Inherited by every process the shell spawns.
    default_timeout_s:
        Soft timeout per command. The paper uses 10s.
    """

    def __init__(
        self,
        cwd: str | os.PathLike[str],
        *,
        env: dict[str, str] | None = None,
        argv_prefix: list[str] | None = None,
        rlimits: dict[int, tuple[int, int]] | None = None,
        default_timeout_s: float = 10.0,
        max_output_bytes: int = 64_000,
        encoding: str = "utf-8",
    ) -> None:
        if not IS_POSIX:
            raise ShellError(
                "PersistentShell requires a POSIX PTY. On Windows, drive the "
                "sandbox on the remote Linux host instead."
            )
        self.cwd = Path(cwd)
        self.env = env
        self.argv_prefix = list(argv_prefix or [])
        self.rlimits = dict(rlimits or {})
        self.default_timeout_s = default_timeout_s
        self.max_output_bytes = max_output_bytes
        self.encoding = encoding

        self._proc: subprocess.Popen[bytes] | None = None
        self._master_fd: int | None = None
        self._token = secrets.token_hex(8)
        self._seq = 0
        # Scratch dir for the sourced command files. Lives inside the sandbox
        # root so isolation backends that swap the mount namespace still see it.
        self._scratch = self.cwd / ".sandbox_lab"

    # ------------------------------------------------------------------ setup

    def start(self) -> None:
        if self._proc is not None:
            raise ShellError("shell already started")
        self.cwd.mkdir(parents=True, exist_ok=True)
        self._scratch.mkdir(parents=True, exist_ok=True)

        master_fd, slave_fd = pty.openpty()
        self._configure_tty(slave_fd)

        # --noediting disables readline. Without it bash wraps every prompt in
        # bracketed-paste escapes (\x1b[?2004h) which land in the model's
        # observations as noise, and which models then imitate.
        argv = [*self.argv_prefix, "bash", "--noprofile", "--norc", "--noediting"]
        try:
            self._proc = subprocess.Popen(
                argv,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                cwd=str(self.cwd),
                env=self.env,
                preexec_fn=self._child_setup(slave_fd),  # noqa: PLW1509
                close_fds=True,
            )
        except FileNotFoundError as exc:
            os.close(master_fd)
            os.close(slave_fd)
            raise ShellError(f"could not start shell: {argv[0]!r} not found") from exc
        finally:
            # The child owns the slave now; holding it open in the parent would
            # keep the PTY alive forever and mask child exit (we would never see
            # EIO on the master).
            os.close(slave_fd)

        self._master_fd = master_fd
        self._init_session()

    def _configure_tty(self, slave_fd: int) -> None:
        """Turn off echo and NL->CRNL translation.

        Without ECHO off, every command we write is echoed straight back and
        ends up quoted verbatim in the model's observation. Without ONLCR off,
        every ``\\n`` becomes ``\\r\\n`` and the transcript is full of carriage
        returns.
        """
        attrs = termios.tcgetattr(slave_fd)
        iflag, oflag, cflag, lflag, ispeed, ospeed, cc = attrs
        lflag &= ~termios.ECHO
        oflag &= ~termios.ONLCR
        # ICANON and ISIG stay ON: ISIG is what turns our \x03 byte into a
        # SIGINT for the foreground process group, which is the whole point.
        termios.tcsetattr(
            slave_fd, termios.TCSANOW, [iflag, oflag, cflag, lflag, ispeed, ospeed, cc]
        )

    def _child_setup(self, slave_fd: int):
        rlimits = self.rlimits

        def _setup() -> None:  # pragma: no cover - runs in forked child
            import resource

            # New session + controlling terminal, so job control works and the
            # shell is insulated from the parent's signals.
            os.setsid()
            fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
            for key, (soft, hard) in rlimits.items():
                try:
                    resource.setrlimit(key, (soft, hard))
                except (ValueError, OSError):
                    # A limit the kernel rejects (or one already lower than the
                    # request) must not abort sandbox creation.
                    pass

        return _setup

    def _init_session(self) -> None:
        # set -m is what gives each command its own foreground process group.
        # HISTFILE off keeps the sandbox from writing outside its root.
        self._write(
            "set -m\n"
            "unset HISTFILE\n"
            "export PS1= PS2= PROMPT_COMMAND=\n"
            "export TERM=dumb\n"
            "stty -echo 2>/dev/null\n"
        )
        # Drain the startup chatter by running a no-op through the normal path.
        self.run(":", timeout_s=15.0)

    # ------------------------------------------------------------------- exec

    def run(self, command: str, *, timeout_s: float | None = None) -> ShellResult:
        """Run ``command`` and return its combined stdout+stderr and status."""
        if self._proc is None or self._master_fd is None:
            raise ShellError("shell not started")
        if self._proc.poll() is not None:
            raise ShellError(f"shell died (exit {self._proc.returncode})")

        timeout = self.default_timeout_s if timeout_s is None else timeout_s
        self._seq += 1
        sentinel = f"__SBXLAB_{self._token}_{self._seq}__"

        script = self._scratch / f"cmd_{self._seq}.sh"
        script.write_text(command, encoding=self.encoding)

        started = time.monotonic()
        # `source` runs in the current shell, so cd/export persist. The printf
        # is a separate line so $? is the command's status, not printf's.
        self._write(f"source {self._shq(str(script))}\n")
        self._write(f"printf '\\n{sentinel}%d__\\n' \"$?\"\n")

        buf = _HeadTailBuffer(self.max_output_bytes // 2, self.max_output_bytes // 2)
        pattern = re.compile(rf"{re.escape(sentinel)}(\d+)__")

        exit_code, timed_out = self._pump(buf, pattern, deadline=started + timeout)

        if timed_out:
            exit_code, timed_out = self._interrupt(buf, pattern)

        duration = time.monotonic() - started
        shell_died = self._proc.poll() is not None
        if shell_died and exit_code is None and not timed_out:
            # `exit N` takes the shell down with the reported status; recover it
            # rather than reporting a bare None the caller cannot interpret.
            exit_code = self._proc.returncode
        try:
            script.unlink()
        except OSError:
            pass

        text = self._strip_sentinel(buf.render(self.encoding), sentinel)
        return ShellResult(
            output=text,
            exit_code=exit_code,
            duration_s=duration,
            timed_out=timed_out,
            truncated=buf.truncated,
            total_bytes=buf.total,
            shell_died=shell_died,
        )

    def _pump(
        self, buf: _HeadTailBuffer, pattern: re.Pattern[str], *, deadline: float
    ) -> tuple[int | None, bool]:
        """Drain the PTY until the sentinel appears or ``deadline`` passes."""
        assert self._master_fd is not None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, True
            try:
                ready, _, _ = select.select([self._master_fd], [], [], min(remaining, 0.1))
            except (OSError, ValueError):
                return None, False
            if not ready:
                continue
            try:
                data = os.read(self._master_fd, _READ_CHUNK)
            except OSError:
                # EIO on a PTY master means every slave fd is closed: the shell
                # exited. Not a timeout - report it as such.
                return None, False
            if not data:
                return None, False
            buf.feed(data)
            # The sentinel is always among the newest output, so only the recent
            # window needs scanning. Searching the whole buffer would be O(n^2)
            # over a chatty command.
            match = pattern.search(buf.recent_text(self.encoding))
            if match:
                return int(match.group(1)), False

    def _interrupt(
        self, buf: _HeadTailBuffer, pattern: re.Pattern[str]
    ) -> tuple[int | None, bool]:
        """Ctrl-C the foreground command, escalating to SIGKILL if it ignores us.

        Returns ``(exit_code, timed_out)``. ``timed_out`` stays True even on a
        clean interrupt: the command did exceed its budget, and the agent needs
        to be told that.
        """
        assert self._master_fd is not None and self._proc is not None
        os.write(self._master_fd, b"\x03")
        code, _ = self._pump(buf, pattern, deadline=time.monotonic() + _INTERRUPT_GRACE_S)
        if code is not None:
            # bash regained control and reported the interrupted command's
            # status (typically 130). The session is healthy.
            return None, True

        # The command trapped or ignored SIGINT, so escalate - but only against
        # the *foreground* process group. os.tcgetpgrp() asks the terminal which
        # group currently owns it, which is the command. Deriving the pgid from
        # bash's own pid instead would target bash itself (it is its own group
        # leader after setsid) and SIGKILL the very session we are preserving.
        shell_pgid = os.getpgid(self._proc.pid)
        try:
            fg_pgid = os.tcgetpgrp(self._master_fd)
        except OSError:
            fg_pgid = -1

        if fg_pgid > 0 and fg_pgid != shell_pgid:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(fg_pgid, sig)
                except (ProcessLookupError, PermissionError):
                    break
                code, _ = self._pump(buf, pattern, deadline=time.monotonic() + 1.0)
                if code is not None:
                    break
        else:
            # Job control is not active, so the command shares bash's group and
            # killpg would take bash with it. Signal bash's children directly.
            self._kill_children(shell_pgid)
            self._pump(buf, pattern, deadline=time.monotonic() + 1.0)
        return None, True

    def _kill_children(self, shell_pgid: int) -> None:
        """SIGKILL bash's direct children, never bash itself."""
        assert self._proc is not None
        pid = self._proc.pid
        try:
            with open(f"/proc/{pid}/task/{pid}/children") as handle:
                raw = handle.read()
        except OSError:
            return
        for token in raw.split():
            try:
                child_pid = int(token)
            except ValueError:
                continue
            try:
                os.kill(child_pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    # ---------------------------------------------------------------- helpers

    def _write(self, text: str) -> None:
        assert self._master_fd is not None
        data = text.encode(self.encoding)
        while data:
            written = os.write(self._master_fd, data)
            data = data[written:]

    @staticmethod
    def _shq(value: str) -> str:
        """Single-quote a string for bash."""
        return "'" + value.replace("'", "'\\''") + "'"

    @staticmethod
    def _strip_sentinel(text: str, sentinel: str) -> str:
        text = re.sub(rf"\n?{re.escape(sentinel)}\d+__\n?", "", text)
        # Belt and braces alongside --noediting: any terminal control sequence
        # that still slips through is framing, not command output.
        text = _ANSI_RE.sub("", text)
        return text.replace("\r\n", "\n").strip("\n")

    # ----------------------------------------------------------------- teardown

    def close(self) -> None:
        proc, self._proc = self._proc, None
        master_fd, self._master_fd = self._master_fd, None
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                pass
        if master_fd is not None:
            try:
                os.close(master_fd)
            except OSError:
                pass

    def restart(self) -> None:
        """Rebuild the session after it exited.

        Files on disk survive (they are the sandbox root, not the process), so
        an episode loses its shell state but not its work.
        """
        self.close()
        self._seq = 0
        self.start()

    def __enter__(self) -> "PersistentShell":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
