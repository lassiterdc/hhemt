"""The cross-process memo of the docs-content gate's two live derivations is keyed on every input and fails safe.

``tests/test_check_docs_content.py`` stores ``rendered_docstrings`` and ``published_source_lines`` in a per-session
directory shared by its xdist workers (``_memo_fetch``), keyed by ``_memo_key``. These tests pin what the key covers
(a byte changed in the gate script, a docs page, ``mkdocs.yml`` or a source file each changes it), that an unchanged
tree hits, that bytecode caches do not move it, that a truncated or wrong-shaped entry is a miss and never an error,
that the publication is an atomic replace, that two processes started together derive once, that the stored value of
each real derivation round-trips equal, and that the module carries no ``xdist_group``.

The probes run on a COPY of the repository subset the key reads, with a cheap stand-in derivation that counts its
calls; the real derivations (35 to 43 s each) are exercised once, by the round-trip arm.
"""

from __future__ import annotations

import os
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import tests.test_check_docs_content as T

_REPO = T._REPO


def _copy_tree(dest: Path) -> Path:
    """Copy exactly the files the memo key reads, leaving out the two large generated reports under docs."""
    for sub in ("src/hhemt", "docs"):
        shutil.copytree(
            _REPO / sub,
            dest / sub,
            ignore=shutil.ignore_patterns("__pycache__", "bundles"),
        )
    for rel in (
        "scripts/check_docs_content.py",
        *T.cdc.SHIPPED_METADATA,
        "mkdocs.yml",
        "tests/test_check_docs_content.py",
    ):
        if (_REPO / rel).is_file():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_REPO / rel, dest / rel)
    return dest


@pytest.fixture
def tree(tmp_path):
    """A copy of the key's inputs."""
    return _copy_tree(tmp_path / "tree")


def _flip_a_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))


def _first_docs_page(root: Path) -> Path:
    return sorted((root / "docs").rglob("*.md"))[0]


def _first_source_file(root: Path) -> Path:
    return sorted((root / "src" / "hhemt").glob("*.py"))[0]


class _Counted:
    """A stand-in derivation that counts its live calls."""

    def __init__(self, reader=None):
        self.calls = 0
        self._reader = reader

    def __call__(self):
        self.calls += 1
        return [("a", Path("p"), 1, self._reader() if self._reader else "x")]


def _fetch(directory: Path, root: Path, derive, name="rendered_docstrings"):
    return T._memo_fetch(directory, T._memo_key(root, name), name, derive, T._rendered_shape_ok)


@pytest.mark.parametrize(
    ("label", "target"),
    [
        ("the gate script", lambda root: root / "scripts" / "check_docs_content.py"),
        ("a docs page", _first_docs_page),
        ("mkdocs.yml", lambda root: root / "mkdocs.yml"),
        ("a source file", _first_source_file),
    ],
)
def test_one_changed_byte_in_any_input_changes_the_key_and_re_derives(tree, tmp_path, label, target):
    """A byte changed in the gate script, a docs page, ``mkdocs.yml`` or a source file moves the key and re-derives.

    Input class: one warm memo, then each of the four input kinds changed in turn. Required because a key that
    misses an input serves a stale result as a green. Killed by a key over ``src/hhemt`` alone (three of the four
    labels keep the old key). Control: the unchanged tree hits (the derivation count stays at one).
    """
    memo = tmp_path / "memo"
    memo.mkdir()
    derive = _Counted()
    _fetch(memo, tree, derive)
    _fetch(memo, tree, derive)
    assert derive.calls == 1, "the unchanged tree must hit the memo"
    before = T._memo_key(tree, "rendered_docstrings")
    _flip_a_byte(target(tree))
    assert T._memo_key(tree, "rendered_docstrings") != before, f"changing {label} did not change the key"
    _fetch(memo, tree, derive)
    assert derive.calls == 2, f"changing {label} did not re-run the derivation"


def test_the_stored_value_follows_the_content_of_a_docs_page(tree, tmp_path):
    """After a docs page changes, the memo returns what the new content yields, not the old stored value.

    Input class: a stand-in derivation that reads one docs page. Required because the point of the key is that the
    gate's result follows its inputs. Killed by a key that ignores ``docs/``: the old value comes back.
    """
    memo = tmp_path / "memo"
    memo.mkdir()
    page = _first_docs_page(tree)
    derive = _Counted(reader=lambda: page.read_text(encoding="utf-8", errors="replace")[:40])
    old = _fetch(memo, tree, derive)
    page.write_text("changed page text that is long enough to show\n", encoding="utf-8")
    new = _fetch(memo, tree, derive)
    assert old != new and new[0][3].startswith("changed page text")


def test_bytecode_caches_do_not_move_the_key(tree):
    """A ``__pycache__`` directory or a ``.pyc`` file changes nothing; a new real source file does.

    Input class: a bytecode cache added under ``src/hhemt`` and a stray ``.pyc``. Required because bytecode changes
    between runs without changing any input, and an unstable key is a silent miss. Killed by a file walk that does
    not exclude ``__pycache__``. Control in the test: a real new file DOES change the key.
    """
    before = T._memo_key(tree, "rendered_docstrings")
    cache = tree / "src" / "hhemt" / "__pycache__"
    cache.mkdir()
    (cache / "x.cpython-312.pyc").write_bytes(b"\0\1\2")
    (tree / "src" / "hhemt" / "stray.pyc").write_bytes(b"\3")
    assert T._memo_key(tree, "rendered_docstrings") == before
    (tree / "src" / "hhemt" / "new_module.py").write_text("X = 1\n", encoding="utf-8")
    assert T._memo_key(tree, "rendered_docstrings") != before


def test_the_two_derivations_have_different_keys(tree):
    """The derivation's name is part of the key, so one entry cannot answer for the other.

    Input class: the two derivation names over one tree. Killed by a key without the name.
    """
    assert T._memo_key(tree, "rendered_docstrings") != T._memo_key(tree, "published_source_lines")


@pytest.mark.parametrize("damage", ["truncate", "garbage", "wrong_shape"])
def test_a_damaged_entry_is_a_miss_that_re_derives_without_error(tree, tmp_path, damage):
    """A truncated, garbage or wrong-shaped entry is re-derived and republished; it never errors.

    Input class: the three ways an entry can be bad. Required because a half-written or foreign file must not fail
    a test or return junk. Killed by a reader that trusts the file (junk returned) or that raises on a bad pickle.
    """
    memo = tmp_path / "memo"
    memo.mkdir()
    derive = _Counted()
    good = _fetch(memo, tree, derive)
    entry = next(memo.glob("*.rendered_docstrings.pkl"))
    if damage == "truncate":
        entry.write_bytes(entry.read_bytes()[:-5])
    elif damage == "garbage":
        entry.write_bytes(b"not a pickle")
    else:
        entry.write_bytes(pickle.dumps({"not": "a list"}))
    again = _fetch(memo, tree, derive)
    assert again == good and derive.calls == 2
    assert _fetch(memo, tree, derive) == good and derive.calls == 2, "the republished entry must hit"


def test_publication_is_an_atomic_replace_with_no_temporary_left(tree, tmp_path, monkeypatch):
    """The entry is written to a temporary name and ``os.replace``d onto the final name.

    Input class: one derivation on an empty memo. Required so a reader sees no entry or a whole one. Killed by a
    write in place (the replace is never called) .
    """
    memo = tmp_path / "memo"
    memo.mkdir()
    calls = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (calls.append((Path(a), Path(b))), real(a, b))[1])
    _fetch(memo, tree, _Counted())
    assert len(calls) == 1
    src, dst = calls[0]
    assert src.parent == dst.parent == memo and dst.suffix == ".pkl" and src != dst
    assert not [p for p in memo.iterdir() if p.name.endswith(".tmp")]


_TWO_PROCESS_SCRIPT = """
import sys, time
from pathlib import Path
import tests.test_check_docs_content as T
log = Path(sys.argv[2])
def derive():
    with open(log, "a") as fh:
        fh.write("d\\n")
    time.sleep(1.5)
    return [("a", Path("p"), 1, "x")]
value = T._memo_fetch(Path(sys.argv[1]), "k", "rendered_docstrings", derive, T._rendered_shape_ok)
print(repr(value))
"""


def test_two_processes_started_together_derive_once_and_agree(tmp_path):
    """Two processes racing on an empty memo run the derivation once and return equal values.

    Input class: two interpreters started together, a derivation that sleeps 1.5 s. Required because the memo
    exists to stop each worker re-deriving. Killed by a memo without the lock (both derive: two log lines).
    """
    memo = tmp_path / "memo"
    memo.mkdir()
    log = tmp_path / "derivations.log"
    script = tmp_path / "racer.py"
    script.write_text(_TWO_PROCESS_SCRIPT, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(_REPO), str(_REPO / "src")])}
    procs = [
        subprocess.Popen(
            [sys.executable, str(script), str(memo), str(log)],
            stdout=subprocess.PIPE,
            text=True,
            env=env,
            cwd=_REPO,
        )
        for _ in range(2)
    ]
    outs = [p.communicate(timeout=120)[0] for p in procs]
    assert [p.returncode for p in procs] == [0, 0]
    assert log.read_text().splitlines() == ["d"]
    assert outs[0] == outs[1] and "x" in outs[0]


def test_the_module_carries_no_xdist_group():
    """``tests/test_check_docs_content.py`` carries no ``xdist_group`` (the withdrawn single-worker group).

    Input class: the module's source text, comments excluded. Killed by a re-introduced ``pytest.mark.xdist_group``.
    """
    code = "\n".join(
        line
        for line in (_REPO / "tests" / "test_check_docs_content.py").read_text().splitlines()
        if not line.lstrip().startswith("#")
    )
    assert re.search(r"xdist_group", code) is None


def test_the_stored_value_of_each_real_derivation_round_trips_equal():
    """Each real derivation's value pickles and unpickles to an equal value of the shape the memo accepts.

    Input class: the live ``rendered_docstrings`` and ``published_source_lines`` results (35 to 43 s each; this is the
    only test here that runs them). Required because the return types were read as annotations and never run; a value
    that does not pickle, or comes back unequal, would make the memo error or serve a different value. Killed by a
    derivation returning a non-pickleable or non-equal-after-round-trip object.
    """
    rendered = T._REAL_RENDERED()
    spans = T._REAL_SPANS()
    assert T._rendered_shape_ok(rendered) and T._spans_shape_ok(spans)
    assert rendered and spans, "an empty derivation proves nothing"
    assert pickle.loads(pickle.dumps(rendered)) == rendered
    assert pickle.loads(pickle.dumps(spans)) == spans
