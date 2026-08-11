"""Isolation backends.

The paper runs each episode in a Docker container. Colab has no Docker daemon,
but it does give us uid 0 and ``/usr/bin/unshare`` - which is enough to build
the isolation out of the same kernel primitives Docker itself uses, without a
daemon or an image.

The interface is deliberately just "argv to put in front of ``bash``" plus an
optional in-shell setup script. That keeps :class:`PersistentShell` completely
unaware of isolation, and makes each backend independently testable.

Why this matters for the experiments
------------------------------------
The paper attributes its gains to three meta-capabilities: external resource
access, file management, and code execution. It ablates them by changing the
*prompt*. A prompt-level ablation is weak evidence: a model told "you have no
network" can still call ``pip install`` and succeed, and a model told to avoid
files can still write them. Namespaces let us remove each capability for real:

* external resource access -> ``--net`` gives a fresh network namespace whose
  only interface is loopback. ``pip install`` genuinely fails.
* file management -> the sandbox root is bind-mounted read-only inside the
  mount namespace. ``open(..., "w")`` genuinely fails.
* code execution -> the ``bash`` tool is not exposed at all.

That turns a prompt suggestion into an enforced constraint, which is the
difference between "the model chose not to" and "the model could not".
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# POSIX-only. Imported lazily so the pure-Python parts of the package (the
# editor, the grader, the dataset loaders) stay importable on Windows for local
# development and CI, even though sandboxes themselves only run on Linux.
try:
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]


class BackendUnavailable(RuntimeError):
    """This backend cannot run on this host."""


# Default rlimits. RLIMIT_AS is the one that actually stops a runaway
# allocation from taking the whole VM down with it; the others are cheap
# insurance. NPROC caps fork bombs, FSIZE caps disk-filling, CORE=0 stops
# multi-gigabyte core dumps on a segfault.
def default_rlimits(
    address_space_mb: int = 4096,
    max_processes: int = 512,
    max_file_mb: int = 512,
    cpu_seconds: int | None = None,
) -> dict[int, tuple[int, int]]:
    if resource is None:
        return {}
    limits: dict[int, tuple[int, int]] = {
        resource.RLIMIT_AS: (address_space_mb * 1024 * 1024,) * 2,
        resource.RLIMIT_NPROC: (max_processes,) * 2,
        resource.RLIMIT_FSIZE: (max_file_mb * 1024 * 1024,) * 2,
        resource.RLIMIT_CORE: (0, 0),
    }
    if cpu_seconds is not None:
        limits[resource.RLIMIT_CPU] = (cpu_seconds, cpu_seconds)
    return limits


@dataclass
class Capabilities:
    """Which of the paper's three meta-capabilities are enabled."""

    code_execution: bool = True
    file_management: bool = True
    external_resources: bool = True

    @property
    def slug(self) -> str:
        """Short identifier used in run directories and result tables."""
        if self.code_execution and self.file_management and self.external_resources:
            return "full"
        off = []
        if not self.code_execution:
            off.append("noexec")
        if not self.file_management:
            off.append("nofile")
        if not self.external_resources:
            off.append("nonet")
        return "+".join(off) if off else "full"


@dataclass
class Backend:
    """Base backend: run the shell directly on the host, limits only."""

    root: Path
    caps: Capabilities = field(default_factory=Capabilities)
    rlimits: dict[int, tuple[int, int]] = field(default_factory=default_rlimits)

    name = "local"

    def preflight(self) -> None:
        """Raise :class:`BackendUnavailable` if this host cannot run us."""

    def argv_prefix(self) -> list[str]:
        return []

    def setup_script(self) -> str:
        """Shell run *inside* the sandbox before the agent's first command."""
        return ""

    def child_hook(self) -> Callable[[], None] | None:
        """Callable run in the forked child, after rlimits, before exec."""
        return None

    def describe(self) -> str:
        return f"{self.name} (caps={self.caps.slug})"


@dataclass
class UnshareBackend(Backend):
    """Linux namespace isolation via ``unshare`` - no daemon, no image.

    Namespaces used, and what each buys:

    ``--pid --fork --mount-proc``
        The agent sees only processes it started. ``kill -9 -1`` inside the
        sandbox cannot reach the vLLM server or the notebook kernel. ``--fork``
        is mandatory: the PID namespace only takes effect for *children*, so
        without it bash would stay in the old namespace.
    ``--mount``
        Mount changes (including the read-only bind below) stay inside.
    ``--uts --ipc``
        Hostname and SysV IPC isolation. Cheap, and stops cross-episode
        interference through shared memory segments.
    ``--net`` (only when external resource access is disabled)
        Fresh network namespace with loopback only. This is the enforced
        version of "no external resources".
    """

    name = "unshare"

    def preflight(self) -> None:
        if os.name != "posix":
            raise BackendUnavailable("unshare requires Linux")
        if shutil.which("unshare") is None:
            raise BackendUnavailable("unshare not found on PATH")
        probe = subprocess.run(
            ["unshare", "--fork", "--pid", "--mount", "true"],
            capture_output=True,
            timeout=20,
        )
        if probe.returncode != 0:
            detail = probe.stderr.decode(errors="replace").strip()
            raise BackendUnavailable(
                f"unshare rejected namespace creation: {detail or 'unknown error'}. "
                "Needs uid 0 or unprivileged user namespaces."
            )

    def argv_prefix(self) -> list[str]:
        argv = ["unshare", "--fork", "--pid", "--mount-proc", "--mount", "--uts", "--ipc"]
        if not self.caps.external_resources:
            argv.append("--net")
        return argv

    def setup_script(self) -> str:
        lines = ["hostname sandbox 2>/dev/null || true"]
        if not self.caps.external_resources:
            # A down loopback makes even 127.0.0.1 fail, which breaks tools that
            # talk to themselves. Bring it up: we are removing *external*
            # access, not local IPC.
            lines.append("ip link set lo up 2>/dev/null || true")
        if not self.caps.file_management:
            # Read-only bind of the root onto itself. Only visible in this mount
            # namespace, so concurrent episodes are unaffected.
            lines.append(f"mount --bind {self.root} {self.root} 2>/dev/null || true")
            lines.append(
                f"mount -o remount,bind,ro {self.root} 2>/dev/null || true"
            )
        return "\n".join(lines)


@dataclass
class SeccompBackend(Backend):
    """Syscall filtering where namespaces are unavailable.

    This is the backend that makes enforced ablations possible on Colab, whose
    container denies ``unshare(2)`` even to uid 0. seccomp needs no privileges
    (only ``PR_SET_NO_NEW_PRIVS`` first), and the filter is inherited by every
    descendant of the shell.

    Honest about what it does *not* do: it filters syscalls, it does not
    virtualise the machine. There is no PID or mount isolation, so the agent can
    see host processes and the wider filesystem. It is strictly weaker than
    :class:`UnshareBackend` - it is used only when that is unavailable, and only
    the network capability is genuinely enforceable this way. Filesystem
    read-only enforcement needs a mount namespace, so ``file_management=False``
    is *not* claimed here.
    """

    name = "seccomp"

    def preflight(self) -> None:
        from .seccomp import seccomp_available

        if os.name != "posix":
            raise BackendUnavailable("seccomp requires Linux")
        if not self.caps.file_management:
            raise BackendUnavailable(
                "seccomp cannot enforce a read-only filesystem - that needs a "
                "mount namespace. Use the unshare or docker backend for the "
                "file_management ablation."
            )
        if not seccomp_available():
            raise BackendUnavailable("kernel or container refused a seccomp filter")

    def child_hook(self) -> Callable[[], None] | None:
        if self.caps.external_resources:
            return None
        from .seccomp import make_network_blocker

        return make_network_blocker()


@dataclass
class DockerBackend(Backend):
    """Container isolation, matching the paper's own setup where available."""

    name = "docker"
    image: str = "python:3.12-slim"
    memory: str = "4g"

    def preflight(self) -> None:
        if shutil.which("docker") is None:
            raise BackendUnavailable("docker not found on PATH")
        probe = subprocess.run(["docker", "info"], capture_output=True, timeout=30)
        if probe.returncode != 0:
            raise BackendUnavailable("docker daemon not reachable")

    def argv_prefix(self) -> list[str]:
        argv = [
            "docker", "run", "--rm", "-i",
            "--memory", self.memory,
            "--pids-limit", "512",
            "-v", f"{self.root}:{self.root}" + ("" if self.caps.file_management else ":ro"),
            "-w", str(self.root),
        ]
        if not self.caps.external_resources:
            argv += ["--network", "none"]
        argv.append(self.image)
        return argv


def select_backend(
    root: Path,
    caps: Capabilities,
    *,
    prefer: str = "auto",
    rlimits: dict[int, tuple[int, int]] | None = None,
) -> Backend:
    """Pick the strongest isolation this host actually supports.

    Order is docker > unshare > local. ``prefer`` forces a specific backend and
    raises if it is unavailable - silently downgrading isolation would make an
    ablation run look enforced when it was not, which is worse than crashing.
    """
    limits = rlimits if rlimits is not None else default_rlimits()
    registry: dict[str, type[Backend]] = {
        "docker": DockerBackend,
        "unshare": UnshareBackend,
        "seccomp": SeccompBackend,
        "local": Backend,
    }

    if prefer != "auto":
        if prefer not in registry:
            raise ValueError(f"unknown backend {prefer!r}; expected {sorted(registry)}")
        backend = registry[prefer](root=root, caps=caps, rlimits=limits)
        backend.preflight()
        return backend

    enforced = not (caps.external_resources and caps.file_management)
    errors = []
    # Strongest first. seccomp sits below the two isolating backends because it
    # filters syscalls without virtualising anything - it can enforce the
    # network ablation, but it is not a substitute for namespaces.
    for key in ("docker", "unshare", "seccomp"):
        backend = registry[key](root=root, caps=caps, rlimits=limits)
        try:
            backend.preflight()
            return backend
        except BackendUnavailable as exc:
            errors.append(f"{key}: {exc}")

    if enforced:
        raise BackendUnavailable(
            "This run disables a capability, which requires namespace or "
            "container isolation to enforce. Falling back to the local backend "
            "would leave the capability quietly enabled and invalidate the "
            "ablation. Tried:\n  " + "\n  ".join(errors)
        )
    return Backend(root=root, caps=caps, rlimits=limits)
