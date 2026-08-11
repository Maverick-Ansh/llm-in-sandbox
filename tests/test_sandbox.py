"""Sandbox integration tests, including the *enforced* capability ablations.

The ablation tests are the ones that give the experiments their teeth. The paper
removes a meta-capability by describing its absence in the prompt; we remove it
with kernel namespaces. These tests are what license the claim that the
capability was actually gone rather than merely discouraged.
"""

from __future__ import annotations

import os

import pytest

from sandbox_lab.sandbox import (
    BackendUnavailable,
    Capabilities,
    Sandbox,
    UnshareBackend,
)

pytestmark = pytest.mark.skipif(os.name != "posix", reason="needs Linux")


def _unshare_available(tmp_path) -> bool:
    try:
        UnshareBackend(root=tmp_path, caps=Capabilities()).preflight()
    except BackendUnavailable:
        return False
    return True


@pytest.fixture()
def sandbox(tmp_path):
    with Sandbox(tmp_path, backend="local") as sb:
        yield sb


def test_bash_and_editor_share_a_filesystem(sandbox):
    """The two tools must see the same disk, or the agent cannot chain them."""
    sandbox.file_editor("create", path="script.py", file_text="print(6 * 7)")
    assert sandbox.bash("python3 script.py") == "42"


def test_state_persists_across_bash_calls(sandbox):
    sandbox.bash("export RUN_ID=abc123")
    assert "abc123" in sandbox.bash("echo $RUN_ID")


def test_no_output_is_marked_explicitly(sandbox):
    """Empty output must not look like a tool failure to the model."""
    assert sandbox.bash("true") == "[no output]"


def test_nonzero_exit_is_surfaced(sandbox):
    assert "[exit code" in sandbox.bash("ls /nope/nope")


def test_timeout_message_tells_the_model_what_to_do(sandbox):
    out = sandbox.bash("sleep 30")
    assert "interrupted" in out
    assert "nohup" in out, "the recovery hint is what stops the model retrying blind"
    assert sandbox.stats.timeouts == 1
    assert sandbox.bash("echo alive") == "alive"


def test_exit_restarts_the_session_transparently(sandbox):
    out = sandbox.bash("exit 1")
    assert "restarted" in out
    assert sandbox.stats.shell_restarts == 1
    assert sandbox.bash("echo recovered") == "recovered"


def test_editor_errors_come_back_as_text_not_exceptions(sandbox):
    """A bad edit is a normal event; it must not end the episode."""
    out = sandbox.file_editor("str_replace", path="missing.py", old_str="a", new_str="b")
    assert out.startswith("ERROR:")


def test_path_jail_holds_through_the_sandbox(sandbox):
    out = sandbox.file_editor("create", path="/etc/pwned", file_text="x")
    assert "outside the sandbox root" in out


def test_stats_track_the_episode(sandbox):
    sandbox.bash("echo one")
    sandbox.bash("false")
    sandbox.file_editor("create", path="a.txt", file_text="x")
    assert sandbox.stats.commands == 2
    assert sandbox.stats.failed_commands == 1
    assert sandbox.stats.edits == 1


def test_documents_land_where_the_prompt_says(tmp_path):
    with Sandbox(tmp_path, backend="local") as sb:
        sb.write_document("ctx.md", "the secret is 8891")
        assert "8891" in sb.bash("cat documents/ctx.md")


# ------------------------------------------------- enforced capability ablation


def test_disabling_code_execution_removes_the_tool(tmp_path):
    caps = Capabilities(code_execution=False)
    with Sandbox(tmp_path, caps=caps, backend="local") as sb:
        assert "disabled" in sb.bash("echo hi")
        # File tools still work: this ablation removes execution, nothing else.
        assert "Created" in sb.file_editor("create", path="a.txt", file_text="x")


@pytest.mark.slow
def test_disabling_external_resources_actually_breaks_the_network(tmp_path):
    """The point of the namespace: `pip install` must genuinely fail.

    A prompt-level ablation cannot distinguish "the model obeyed" from "the
    model ignored us and it worked". This can.
    """
    if not _unshare_available(tmp_path):
        pytest.skip("unshare unavailable (needs uid 0 or unprivileged userns)")

    caps = Capabilities(external_resources=False)
    with Sandbox(tmp_path, caps=caps, backend="unshare") as sb:
        out = sb.bash(
            "python3 -c \"import socket;"
            "socket.create_connection(('1.1.1.1',443),timeout=3);print('REACHED')\"",
            timeout_s=15,
        )
        assert "REACHED" not in out

    # Control: with the capability enabled, the same probe must succeed -
    # otherwise the test above proves nothing about the namespace.
    with Sandbox(tmp_path / "net", caps=Capabilities(), backend="unshare") as sb:
        out = sb.bash(
            "python3 -c \"import socket;"
            "socket.create_connection(('1.1.1.1',443),timeout=5);print('REACHED')\"",
            timeout_s=20,
        )
        assert "REACHED" in out


@pytest.mark.slow
def test_disabling_file_management_makes_the_root_read_only(tmp_path):
    if not _unshare_available(tmp_path):
        pytest.skip("unshare unavailable")

    caps = Capabilities(file_management=False)
    with Sandbox(tmp_path, caps=caps, backend="unshare") as sb:
        out = sb.bash("echo data > should_fail.txt && echo WROTE", timeout_s=15)
        assert "WROTE" not in out
        # The editor refuses too, and reading is still allowed: this ablation
        # removes file *management*, not access to the task's own materials.
        assert "disabled" in sb.file_editor("create", path="x.txt", file_text="x")


def _disable(monkeypatch, *names: str) -> None:
    import sandbox_lab.sandbox.backends as backends

    def always_unavailable(self):
        raise BackendUnavailable("simulated")

    for name in names:
        monkeypatch.setattr(getattr(backends, name), "preflight", always_unavailable)


def test_ablation_refuses_to_silently_downgrade_isolation(tmp_path, monkeypatch):
    """If no backend can enforce it, an ablation must fail loudly.

    Falling back to the local backend would leave the capability quietly
    enabled and produce a result table that looks fine and means nothing.
    """
    _disable(monkeypatch, "UnshareBackend", "DockerBackend", "SeccompBackend")

    with pytest.raises(BackendUnavailable, match="invalidate the ablation"):
        Sandbox(tmp_path, caps=Capabilities(external_resources=False)).start()


def test_seccomp_is_used_when_namespaces_are_unavailable(tmp_path, monkeypatch):
    """The whole point of the seccomp backend.

    With namespaces denied - the Colab situation - selection must fall through
    to seccomp and still enforce the network ablation, rather than either
    raising or silently downgrading to no isolation at all.
    """
    from sandbox_lab.sandbox import seccomp_available

    if not seccomp_available():
        pytest.skip("host refuses seccomp filters")
    _disable(monkeypatch, "UnshareBackend", "DockerBackend")

    with Sandbox(tmp_path, caps=Capabilities(external_resources=False)) as sb:
        assert "seccomp" in sb.backend_name


def test_full_capabilities_may_fall_back_to_local(tmp_path, monkeypatch):
    """With nothing ablated there is nothing to enforce, so a fallback is fine."""
    import sandbox_lab.sandbox.backends as backends

    def always_unavailable(self):
        raise BackendUnavailable("simulated")

    monkeypatch.setattr(backends.UnshareBackend, "preflight", always_unavailable)
    monkeypatch.setattr(backends.DockerBackend, "preflight", always_unavailable)

    with Sandbox(tmp_path, caps=Capabilities()) as sb:
        assert sb.bash("echo ok") == "ok"
