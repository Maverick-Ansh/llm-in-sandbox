"""Tests for the persistent shell.

These are the tests that matter most in the whole repo. Every property below is
one an agent episode silently depends on, and each has a failure mode that looks
like "the model is dumb" rather than "the tool is broken":

* state not persisting  -> the model's ``cd`` is undone and it thrashes on paths
* a timeout killing the shell -> every later command fails for no visible reason
* the sentinel leaking  -> the model sees framing junk and imitates it
* output not draining   -> a chatty command deadlocks the episode
"""

from __future__ import annotations

import os
import time

import pytest

from sandbox_lab.sandbox import PersistentShell

pytestmark = pytest.mark.skipif(os.name != "posix", reason="needs a POSIX PTY")


@pytest.fixture()
def shell(tmp_path):
    sh = PersistentShell(tmp_path, default_timeout_s=10.0)
    sh.start()
    yield sh
    sh.close()


def test_basic_command_and_exit_code(shell):
    result = shell.run("echo hello")
    assert result.output == "hello"
    assert result.exit_code == 0
    assert result.ok


def test_nonzero_exit_code_is_reported(shell):
    # A subshell, not a bare `exit`: commands are sourced into the live shell,
    # so `exit` would end the session rather than report a status - see
    # test_exit_ends_the_session_and_is_flagged.
    result = shell.run("(exit 3)")
    assert result.exit_code == 3
    assert not result.ok
    assert not result.shell_died


def test_failing_command_reports_status_and_keeps_session(shell):
    assert shell.run("false").exit_code == 1
    assert shell.run("ls /definitely/not/here").exit_code != 0
    assert shell.run("echo survived").output == "survived"


def test_exit_ends_the_session_and_is_flagged(shell):
    """`exit` really does end the session, and we must say so.

    This is correct terminal behaviour and not something to paper over, but the
    caller has to be able to tell it apart from a timeout - both would otherwise
    surface as exit_code=None.
    """
    result = shell.run("exit 3")
    assert result.shell_died
    assert result.exit_code == 3
    assert not result.timed_out


def test_shell_can_be_restarted_after_exit(shell, tmp_path):
    (tmp_path / "work.txt").write_text("preserved")
    shell.run("exit 1")
    shell.restart()
    assert shell.run("echo back").output == "back"
    # Files are the sandbox root, not the process: work must survive.
    assert shell.run("cat work.txt").output == "preserved"


def test_stderr_is_interleaved_with_stdout(shell):
    result = shell.run("echo out; echo err >&2")
    assert "out" in result.output
    assert "err" in result.output


def test_state_persists_across_calls(shell, tmp_path):
    """cwd, exports and functions must survive - this is the whole point."""
    (tmp_path / "sub").mkdir()
    shell.run("cd sub")
    shell.run("export MY_VAR=persisted")
    shell.run("myfunc() { echo from_func; }")

    assert shell.run("pwd").output.endswith("sub")
    assert shell.run("echo $MY_VAR").output == "persisted"
    assert shell.run("myfunc").output == "from_func"


def test_python_state_does_not_persist_but_files_do(shell, tmp_path):
    """Sanity check on the mental model: bash is persistent, python is not."""
    shell.run("python3 -c \"open('made.txt','w').write('x')\"")
    assert (tmp_path / "made.txt").exists()


def test_sentinel_never_leaks_into_output(shell):
    result = shell.run("echo done")
    assert "__SBXLAB" not in result.output
    assert result.output == "done"


def test_command_with_heredoc_survives_framing(shell, tmp_path):
    """A heredoc must not swallow the completion sentinel.

    This is precisely why commands are sourced from a file instead of piped in
    as text; piping makes this test hang until the timeout.
    """
    result = shell.run(
        "cat <<'EOF' > poem.txt\nline one\nline two\nEOF\nwc -l < poem.txt"
    )
    assert result.exit_code == 0
    assert result.output.strip() == "2"
    assert (tmp_path / "poem.txt").read_text() == "line one\nline two\n"


def test_multiline_script_with_quotes(shell):
    result = shell.run(
        'for i in 1 2 3; do\n  echo "item $i"\ndone'
    )
    assert result.output.splitlines() == ["item 1", "item 2", "item 3"]


# ------------------------------------------------------------------ timeouts


@pytest.mark.slow
def test_timeout_interrupts_command_but_shell_survives(shell):
    result = shell.run("sleep 30", timeout_s=2.0)
    assert result.timed_out
    assert result.exit_code is None
    assert result.duration_s < 10, "should interrupt promptly, not wait out the sleep"

    # The session must still be usable, with its state intact.
    after = shell.run("echo still_alive")
    assert after.output == "still_alive"
    assert after.exit_code == 0


@pytest.mark.slow
def test_timeout_does_not_lose_working_directory(shell, tmp_path):
    (tmp_path / "deep").mkdir()
    shell.run("cd deep")
    shell.run("sleep 30", timeout_s=2.0)
    assert shell.run("pwd").output.endswith("deep")


@pytest.mark.slow
def test_output_before_timeout_is_preserved(shell):
    """A command that prints and then hangs must not lose what it printed."""
    result = shell.run("echo partial_result; sleep 30", timeout_s=2.0)
    assert "partial_result" in result.output
    assert result.timed_out


@pytest.mark.slow
def test_sigint_ignoring_command_is_escalated(shell):
    """A command that traps SIGINT must still be stopped."""
    result = shell.run("trap '' INT; sleep 30", timeout_s=2.0)
    assert result.timed_out
    assert shell.run("echo recovered").output == "recovered"


# ------------------------------------------------------------------- output


def test_large_output_is_truncated_head_and_tail(tmp_path):
    sh = PersistentShell(tmp_path, max_output_bytes=2000, default_timeout_s=20.0)
    sh.start()
    try:
        result = sh.run("seq 1 20000")
        assert result.truncated
        assert "truncated" in result.output
        assert len(result.output) < 6000
        # Head and tail both survive: we need to see how it started and ended.
        assert "\n1\n" in result.output or result.output.startswith("1\n")
        assert "20000" in result.output
        assert result.exit_code == 0
    finally:
        sh.close()


@pytest.mark.slow
def test_runaway_output_does_not_deadlock(tmp_path):
    """An infinite writer must hit the timeout, not block forever on a full pipe."""
    sh = PersistentShell(tmp_path, max_output_bytes=4000, default_timeout_s=3.0)
    sh.start()
    try:
        started = time.monotonic()
        result = sh.run("yes flood")
        assert result.timed_out
        assert time.monotonic() - started < 15
        assert sh.run("echo alive").output == "alive"
    finally:
        sh.close()


def test_no_output_command_returns_empty_not_garbage(shell):
    assert shell.run("true").output == ""


def test_context_manager_cleans_up(tmp_path):
    with PersistentShell(tmp_path) as sh:
        assert sh.run("echo ctx").output == "ctx"
