"""Tests for the file_editor tool, including the path jail."""

from __future__ import annotations

import os

import pytest

from sandbox_lab.sandbox import EditorError, FileEditor


@pytest.fixture()
def editor(tmp_path):
    return FileEditor(tmp_path)


def test_create_then_view_round_trips(editor, tmp_path):
    editor.create(path=str(tmp_path / "a.py"), file_text="x = 1\ny = 2\n")
    out = editor.view(path=str(tmp_path / "a.py"))
    assert "1\tx = 1" in out
    assert "2\ty = 2" in out


def test_view_is_one_indexed_and_matches_insert_semantics(editor, tmp_path):
    """view's line numbers must be the same numbers insert accepts.

    If these two disagree by one, every edit the model makes lands one line off
    and it takes many turns to notice.
    """
    path = str(tmp_path / "a.txt")
    editor.create(path=path, file_text="alpha\nbeta\ngamma\n")
    editor.insert(path=path, insert_line=2, new_str="INSERTED")
    assert (tmp_path / "a.txt").read_text() == "alpha\nbeta\nINSERTED\ngamma\n"


def test_insert_at_zero_goes_to_the_top(editor, tmp_path):
    path = str(tmp_path / "a.txt")
    editor.create(path=path, file_text="one\ntwo\n")
    editor.insert(path=path, insert_line=0, new_str="zero")
    assert (tmp_path / "a.txt").read_text().splitlines()[0] == "zero"


def test_str_replace_refuses_ambiguous_match(editor, tmp_path):
    path = str(tmp_path / "a.py")
    editor.create(path=path, file_text="v = 1\nv = 1\n")
    with pytest.raises(EditorError, match="appears 2 times"):
        editor.str_replace(path=path, old_str="v = 1", new_str="v = 2")
    # The file must be untouched: a refused edit that half-applied would be
    # worse than no edit at all.
    assert (tmp_path / "a.py").read_text() == "v = 1\nv = 1\n"


def test_str_replace_reports_missing_text(editor, tmp_path):
    path = str(tmp_path / "a.py")
    editor.create(path=path, file_text="hello\n")
    with pytest.raises(EditorError, match="not found verbatim"):
        editor.str_replace(path=path, old_str="goodbye", new_str="x")


def test_str_replace_unique_match_succeeds_and_echoes_context(editor, tmp_path):
    path = str(tmp_path / "a.py")
    editor.create(path=path, file_text="a\nb\nTARGET\nd\ne\n")
    out = editor.str_replace(path=path, old_str="TARGET", new_str="REPLACED")
    assert "REPLACED" in (tmp_path / "a.py").read_text()
    assert "REPLACED" in out, "the edit snippet should be echoed back to the model"


def test_view_range_and_end_of_file_sentinel(editor, tmp_path):
    path = str(tmp_path / "a.txt")
    editor.create(path=path, file_text="\n".join(str(i) for i in range(1, 21)))
    out = editor.view(path=path, view_range=[5, 7])
    assert "5\t5" in out and "7\t7" in out and "8\t8" not in out
    assert "20\t20" in editor.view(path=path, view_range=[18, -1])


def test_view_range_rejects_inverted_range(editor, tmp_path):
    path = str(tmp_path / "a.txt")
    editor.create(path=path, file_text="a\nb\nc\n")
    with pytest.raises(EditorError, match="before start"):
        editor.view(path=path, view_range=[3, 1])


# --------------------------------------------------------------- path jail


def test_jail_blocks_parent_traversal(editor, tmp_path):
    with pytest.raises(EditorError, match="outside the sandbox root"):
        editor.view(path=str(tmp_path / ".." / "escape.txt"))


def test_jail_blocks_absolute_path_outside_root(editor):
    with pytest.raises(EditorError, match="outside the sandbox root"):
        editor.create(path="/etc/passwd_clone", file_text="nope")


@pytest.mark.skipif(os.name != "posix", reason="symlink creation needs POSIX perms")
def test_jail_blocks_symlink_escape(editor, tmp_path):
    """A symlink inside the root pointing out of it must not be a way out.

    This is the case a purely lexical ".." check misses, which is why the jail
    resolves paths before comparing.
    """
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("secret")
    (tmp_path / "link.txt").symlink_to(outside)
    with pytest.raises(EditorError, match="outside the sandbox root"):
        editor.view(path=str(tmp_path / "link.txt"))


def test_relative_paths_resolve_against_root(editor, tmp_path):
    editor.create(path="nested/deep/a.txt", file_text="hi")
    assert (tmp_path / "nested" / "deep" / "a.txt").read_text() == "hi"


def test_unknown_command_is_rejected(editor):
    with pytest.raises(EditorError, match="unknown command"):
        editor(command="delete_everything", path="/tmp/x")


def test_view_directory_lists_entries(editor, tmp_path):
    editor.create(path=str(tmp_path / "documents" / "ctx.md"), file_text="body")
    out = editor.view(path=str(tmp_path))
    assert "documents/" in out


def test_binary_file_is_refused(editor, tmp_path):
    (tmp_path / "blob.bin").write_bytes(b"\x00\xff\xfe garbage")
    with pytest.raises(EditorError, match="not UTF-8"):
        editor.view(path=str(tmp_path / "blob.bin"))
