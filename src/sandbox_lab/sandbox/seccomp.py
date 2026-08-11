"""Unprivileged syscall filtering via seccomp-BPF.

Why this exists
---------------
The interesting ablation in this project is removing a meta-capability *for
real* rather than asking the model not to use it. The natural tool is a network
namespace, but the Colab runtime denies ``unshare(2)`` outright - the container
drops ``CAP_SYS_ADMIN``, so even uid 0 gets ``Operation not permitted``.

seccomp is the way through. Since Linux 3.5, an **unprivileged** process may
install a syscall filter provided it first sets ``PR_SET_NO_NEW_PRIVS``, which
promises the kernel it cannot gain privileges through a later ``execve`` of a
setuid binary. That promise is what makes the operation safe to allow without
capabilities. It is the same primitive container runtimes use for syscall
filtering, and it is inherited across ``fork`` and ``execve``, so a filter
installed just before the shell starts covers every process the agent spawns.

What the filter does
--------------------
Deny ``socket(AF_INET, ...)`` and ``socket(AF_INET6, ...)`` with ``EACCES``;
allow everything else. Blocking socket *creation* is sufficient - there is
nothing to ``connect`` without a socket - and it is far less invasive than
filtering ``connect`` itself.

``AF_UNIX`` is deliberately left alone. The ablation removes *external resource*
access, not local IPC, and tools that talk to themselves over a unix socket
should keep working; breaking them would confound the ablation with unrelated
failures.

Caveats, stated plainly:

* x86-64 only. The filter checks ``arch`` and allows everything on any other
  architecture rather than failing closed, because a wrong-architecture filter
  would silently block the wrong syscall numbers.
* A process that already holds a connected socket keeps it. Nothing in an agent
  episode does that, since the filter is installed before the shell exists.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from collections.abc import Callable

# --- BPF instruction encoding (linux/bpf_common.h) ---------------------------
_BPF_LD_W_ABS = 0x20  # BPF_LD | BPF_W | BPF_ABS
_BPF_JEQ_K = 0x15  # BPF_JMP | BPF_JEQ | BPF_K
_BPF_RET_K = 0x06  # BPF_RET | BPF_K

# --- seccomp (linux/seccomp.h) -----------------------------------------------
_SECCOMP_RET_ALLOW = 0x7FFF0000
_SECCOMP_RET_ERRNO = 0x00050000
_SECCOMP_SET_MODE_FILTER = 1

_PR_SET_NO_NEW_PRIVS = 38
_NR_SECCOMP = 317  # x86-64

_AUDIT_ARCH_X86_64 = 0xC000003E
_NR_SOCKET = 41  # x86-64
_AF_INET, _AF_INET6 = 2, 10

# Offsets into struct seccomp_data: {int nr; u32 arch; u64 ip; u64 args[6];}
_OFF_NR, _OFF_ARCH, _OFF_ARG0 = 0, 4, 16


class _SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint16),
        ("jt", ctypes.c_uint8),
        ("jf", ctypes.c_uint8),
        ("k", ctypes.c_uint32),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [
        ("len", ctypes.c_ushort),
        ("filter", ctypes.POINTER(_SockFilter)),
    ]


class SeccompUnavailable(RuntimeError):
    """This kernel or container will not let us install a filter."""


def _build_network_filter(errno: int = 13) -> _SockFprog:
    """Deny AF_INET/AF_INET6 socket creation, allow everything else.

    Jump targets are offsets from the *next* instruction, so the indices below
    are load-bearing - an off-by-one silently produces a filter that blocks the
    wrong thing, which is exactly the sort of bug that would make an ablation
    look enforced when it is not. Layout:

        0 load arch
        1 arch != x86_64            -> allow (7)
        2 load nr
        3 nr != socket              -> allow (7)
        4 load args[0] (domain)
        5 domain == AF_INET         -> deny  (8)
        6 domain == AF_INET6        -> deny  (8)
        7 return ALLOW
        8 return ERRNO
    """
    deny = _SECCOMP_RET_ERRNO | (errno & 0xFFFF)
    instructions = (_SockFilter * 9)(
        _SockFilter(_BPF_LD_W_ABS, 0, 0, _OFF_ARCH),
        _SockFilter(_BPF_JEQ_K, 0, 5, _AUDIT_ARCH_X86_64),
        _SockFilter(_BPF_LD_W_ABS, 0, 0, _OFF_NR),
        _SockFilter(_BPF_JEQ_K, 0, 3, _NR_SOCKET),
        _SockFilter(_BPF_LD_W_ABS, 0, 0, _OFF_ARG0),
        _SockFilter(_BPF_JEQ_K, 2, 0, _AF_INET),
        _SockFilter(_BPF_JEQ_K, 1, 0, _AF_INET6),
        _SockFilter(_BPF_RET_K, 0, 0, _SECCOMP_RET_ALLOW),
        _SockFilter(_BPF_RET_K, 0, 0, deny),
    )
    program = _SockFprog(len(instructions), instructions)
    # The Structure does not own the instruction array, so without this the
    # array is garbage collected and `filter` dangles into freed memory.
    program._instructions = instructions  # type: ignore[attr-defined]
    return program


def make_network_blocker() -> Callable[[], None]:
    """Build a callable that installs the no-network filter on the *caller*.

    The returned callable is meant for ``subprocess`` ``preexec_fn``: it runs in
    the forked child, after ``fork`` and before ``execve``. Everything that can
    be prepared in the parent - loading libc, resolving symbols, allocating the
    BPF program - is prepared here, because allocating in a forked child of a
    threaded process can deadlock on the allocator lock.
    """
    if os.name != "posix":
        raise SeccompUnavailable("seccomp is Linux-only")

    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    program = _build_network_filter()
    program_ref = ctypes.byref(program)

    def install() -> None:  # pragma: no cover - runs in the forked child
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            os._exit(97)
        if libc.syscall(_NR_SECCOMP, _SECCOMP_SET_MODE_FILTER, 0, program_ref) != 0:
            # Exit rather than continue: a sandbox that silently kept its
            # network would make the ablation meaningless.
            os._exit(98)

    return install


def seccomp_available() -> bool:
    """Check in a throwaway child whether a filter can actually be installed.

    Done by forking, because installing a filter is irreversible for the
    process that does it - probing in-process would permanently break the
    caller's own networking.
    """
    if os.name != "posix":
        return False
    try:
        install = make_network_blocker()
    except (SeccompUnavailable, OSError):
        return False

    pid = os.fork()
    if pid == 0:  # pragma: no cover - child
        try:
            install()
            os._exit(0)
        except Exception:
            os._exit(99)
    _, status = os.waitpid(pid, 0)
    return os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
