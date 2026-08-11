"""Tests for seccomp-enforced network removal.

These carry the weight of the ablation claim on any host where ``unshare`` is
denied - which includes Colab. Each has a **positive control**: a test that only
checks "the network is down" also passes on a machine with no network at all,
and would certify an ablation that never happened.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(os.name != "posix", reason="seccomp is Linux-only")

from sandbox_lab.sandbox import (  # noqa: E402 - after the platform skip
    BackendUnavailable,
    Capabilities,
    Sandbox,
    SeccompBackend,
    seccomp_available,
)

NET_PROBE = (
    "import socket;"
    "socket.create_connection(('1.1.1.1',443),timeout=4);"
    "print('REACHED')"
)


def _has_network() -> bool:
    """Is there a network to remove in the first place?"""
    probe = subprocess.run(
        [sys.executable, "-c", NET_PROBE], capture_output=True, text=True, timeout=30
    )
    return "REACHED" in probe.stdout


requires_seccomp = pytest.mark.skipif(
    not seccomp_available(), reason="kernel/container refuses seccomp filters"
)
requires_network = pytest.mark.skipif(
    not _has_network(), reason="no network available, so removing it proves nothing"
)


@requires_seccomp
def test_seccomp_is_available_and_reports_honestly():
    assert seccomp_available() is True


@requires_seccomp
def test_backend_refuses_to_claim_the_filesystem_ablation():
    """seccomp cannot make a path read-only; it must say so rather than pretend.

    Silently accepting `file_management=False` here would produce a run labelled
    as an enforced ablation in which the model could still write files.
    """
    backend = SeccompBackend(root="/tmp", caps=Capabilities(file_management=False))
    with pytest.raises(BackendUnavailable, match="mount namespace"):
        backend.preflight()


@requires_seccomp
@requires_network
@pytest.mark.slow
def test_network_is_actually_blocked_inside_the_sandbox(tmp_path):
    caps = Capabilities(external_resources=False)
    with Sandbox(tmp_path / "off", caps=caps, backend="seccomp") as sb:
        out = sb.bash(f"python3 -c \"{NET_PROBE}\"", timeout_s=20)
    assert "REACHED" not in out


@requires_seccomp
@requires_network
@pytest.mark.slow
def test_positive_control_same_probe_succeeds_with_the_capability_on(tmp_path):
    """The control for the test above: without the ablation, the probe works."""
    with Sandbox(tmp_path / "on", caps=Capabilities(), backend="seccomp") as sb:
        out = sb.bash(f"python3 -c \"{NET_PROBE}\"", timeout_s=25)
    assert "REACHED" in out


@requires_seccomp
@pytest.mark.slow
def test_pip_install_genuinely_fails_without_external_resources(tmp_path):
    """The capability the paper names, removed for real rather than discouraged."""
    caps = Capabilities(external_resources=False)
    with Sandbox(tmp_path, caps=caps, backend="seccomp") as sb:
        out = sb.bash("pip install --no-input tabulate 2>&1 | tail -3", timeout_s=45)
    assert "Successfully installed" not in out


@requires_seccomp
@pytest.mark.slow
def test_local_ipc_and_files_still_work_without_network(tmp_path):
    """The ablation must remove *external* access only.

    Breaking unix sockets or file writes too would confound the ablation with
    unrelated failures, and the model would flounder for reasons that have
    nothing to do with the capability under test.
    """
    caps = Capabilities(external_resources=False)
    with Sandbox(tmp_path, caps=caps, backend="seccomp") as sb:
        unix = sb.bash(
            "python3 -c \"import socket;"
            "s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);s.close();print('UNIX_OK')\"",
            timeout_s=20,
        )
        assert "UNIX_OK" in unix
        assert "Created" in sb.file_editor("create", path="a.txt", file_text="x")
        assert sb.bash("echo compute > b.txt && cat b.txt") == "compute"


@requires_seccomp
@pytest.mark.slow
def test_filter_is_inherited_by_grandchildren(tmp_path):
    """seccomp survives fork and exec, so a subshell cannot escape it."""
    caps = Capabilities(external_resources=False)
    with Sandbox(tmp_path, caps=caps, backend="seccomp") as sb:
        out = sb.bash(f"bash -c 'python3 -c \\\"{NET_PROBE}\\\"'", timeout_s=25)
    assert "REACHED" not in out
