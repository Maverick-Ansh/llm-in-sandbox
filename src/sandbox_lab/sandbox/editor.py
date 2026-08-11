"""The ``file_editor`` tool: view / create / str_replace / insert.

This mirrors the four commands the paper specifies. The semantics below are the
ones that matter for agent behaviour, and each exists for a reason learned the
hard way by every text-editing tool:

* **1-indexed, line-numbered ``view``.** The model has to be able to name a line
  in a follow-up ``insert`` call. Numbering the output is what makes that
  possible, and it must match the numbering ``insert`` expects.
* **``str_replace`` demands a unique match.** If ``old_str`` appears twice, the
  edit is ambiguous and silently patching the first hit corrupts the file in a
  way the model will not notice for several turns. Refusing, and reporting
  *which* lines matched, turns a silent corruption into a recoverable error.
* **Every path is jailed.** Paths are resolved (following symlinks) and required
  to sit under the sandbox root, so ``../../etc/passwd`` and a symlink pointing
  out of the root both fail closed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Guardrail on how much file content can enter the model's context in one call.
MAX_VIEW_CHARS = 32_000
# Refuse to open anything that is obviously not source/text.
MAX_FILE_BYTES = 8 * 1024 * 1024


class EditorError(Exception):
    """A user-correctable error. The message is shown to the model verbatim."""


@dataclass
class PathJail:
    """Confines every path operation to ``root``.

    ``resolve()`` is called with ``strict=False`` so we can validate paths that
    do not exist yet (``create`` needs that), while still collapsing ``..`` and
    resolving any symlink component that *does* exist. Checking the resolved
    path means a symlink inside the sandbox that points outside it is caught,
    which a purely lexical check would miss.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()

    def check(self, raw: str) -> Path:
        if not raw or not str(raw).strip():
            raise EditorError("path must not be empty")
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise EditorError(
                f"path {raw!r} resolves outside the sandbox root {self.root}. "
                "All work must stay inside the sandbox."
            )
        return resolved


class FileEditor:
    """Stateless file operations, jailed to a root directory."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.jail = PathJail(Path(root))

    # --------------------------------------------------------------- dispatch

    def __call__(self, command: str, **kwargs: object) -> str:
        handlers = {
            "view": self.view,
            "create": self.create,
            "str_replace": self.str_replace,
            "insert": self.insert,
        }
        if command not in handlers:
            raise EditorError(
                f"unknown command {command!r}; expected one of {sorted(handlers)}"
            )
        return handlers[command](**kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ view

    def view(self, path: str, view_range: list[int] | None = None) -> str:
        target = self.jail.check(path)
        if target.is_dir():
            if view_range is not None:
                raise EditorError("view_range is not valid for a directory")
            return self._list_dir(target)
        text = self._read(target)
        lines = text.splitlines()
        start, end = 1, len(lines)
        if view_range is not None:
            start, end = self._parse_range(view_range, len(lines))
        body = self._number(lines[start - 1 : end], first_line=start)
        header = f"{target} (lines {start}-{end} of {len(lines)})"
        if len(body) > MAX_VIEW_CHARS:
            body = (
                body[:MAX_VIEW_CHARS]
                + f"\n[... truncated at {MAX_VIEW_CHARS} chars; narrow view_range ...]"
            )
        return f"{header}\n{body}"

    def _list_dir(self, target: Path) -> str:
        entries = []
        for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name)):
            if child.name.startswith(".sandbox_lab"):
                continue  # internal scratch; not the agent's business
            if child.is_dir():
                entries.append(f"{child.name}/")
            else:
                try:
                    entries.append(f"{child.name}  ({child.stat().st_size} bytes)")
                except OSError:
                    entries.append(child.name)
        listing = "\n".join(entries) if entries else "(empty directory)"
        return f"{target} (directory)\n{listing}"

    # ---------------------------------------------------------------- create

    def create(self, path: str, file_text: str) -> str:
        target = self.jail.check(path)
        if target.is_dir():
            raise EditorError(f"{target} is a directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.exists()
        target.write_text(file_text, encoding="utf-8")
        verb = "Overwrote" if existed else "Created"
        return f"{verb} {target} ({len(file_text)} chars, {file_text.count(chr(10)) + 1} lines)"

    # ----------------------------------------------------------- str_replace

    def str_replace(self, path: str, old_str: str, new_str: str = "") -> str:
        target = self.jail.check(path)
        text = self._read(target)
        count = text.count(old_str)
        if count == 0:
            raise EditorError(
                f"old_str not found verbatim in {target}. "
                "Whitespace and indentation must match exactly - view the file first."
            )
        if count > 1:
            hits = [
                i + 1
                for i, line in enumerate(text.splitlines())
                if old_str.splitlines()[0] in line
            ]
            raise EditorError(
                f"old_str appears {count} times in {target} (near lines "
                f"{hits[:10]}). Include surrounding context to make it unique."
            )
        target.write_text(text.replace(old_str, new_str, 1), encoding="utf-8")
        line_no = text[: text.index(old_str)].count("\n") + 1
        return f"Edited {target} at line {line_no}.\n{self._snippet(target, line_no)}"

    # ---------------------------------------------------------------- insert

    def insert(self, path: str, insert_line: int, new_str: str) -> str:
        target = self.jail.check(path)
        text = self._read(target)
        lines = text.splitlines()
        if not 0 <= insert_line <= len(lines):
            raise EditorError(
                f"insert_line must be between 0 and {len(lines)} "
                f"(0 inserts at the top); got {insert_line}"
            )
        addition = new_str.splitlines()
        merged = lines[:insert_line] + addition + lines[insert_line:]
        target.write_text("\n".join(merged) + "\n", encoding="utf-8")
        return (
            f"Inserted {len(addition)} line(s) into {target} after line "
            f"{insert_line}.\n{self._snippet(target, insert_line + 1)}"
        )

    # --------------------------------------------------------------- helpers

    def _read(self, target: Path) -> str:
        if not target.exists():
            raise EditorError(f"{target} does not exist")
        if target.is_dir():
            raise EditorError(f"{target} is a directory, not a file")
        size = target.stat().st_size
        if size > MAX_FILE_BYTES:
            raise EditorError(f"{target} is {size} bytes, too large to edit ({MAX_FILE_BYTES} max)")
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise EditorError(f"{target} is not UTF-8 text (binary file?)") from exc

    @staticmethod
    def _parse_range(view_range: list[int], n_lines: int) -> tuple[int, int]:
        if len(view_range) != 2:
            raise EditorError("view_range must be [start, end]")
        start, end = view_range
        if start < 1:
            raise EditorError("view_range start is 1-indexed and must be >= 1")
        # -1 is the idiomatic "to end of file".
        if end == -1:
            end = n_lines
        if end < start:
            raise EditorError(f"view_range end ({end}) is before start ({start})")
        return start, min(end, n_lines)

    @staticmethod
    def _number(lines: list[str], first_line: int) -> str:
        width = len(str(first_line + len(lines) - 1)) if lines else 1
        return "\n".join(
            f"{first_line + i:>{width}}\t{line}" for i, line in enumerate(lines)
        )

    def _snippet(self, target: Path, line_no: int, context: int = 4) -> str:
        """Echo the edited region back so the model can confirm the result.

        Without this the model routinely re-reads the whole file after every
        edit, which is the single largest avoidable source of token burn in a
        multi-turn episode.
        """
        lines = self._read(target).splitlines()
        start = max(1, line_no - context)
        end = min(len(lines), line_no + context)
        return self._number(lines[start - 1 : end], first_line=start)
