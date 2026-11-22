"""Reviewer probes for change 1117-HOOK: `check-a --range` grades the pushed range.

Written by the reviewing seat (versioning-specialist), not the coder. Each arm drives the REAL
script through `_run` over a temp repo built with one `git fast-import`, so a merge graph costs no
more than a line. Nothing here reads the working tree of the repo under review except the
registration arm, which reads `.pre-commit-config.yaml` as data.

Findings these arms carry:

* F1 (the first-parent walk, repaired in the second round): the side-branch-jump-through-a-merge arm
  asserts the CRITERION ("a real jump of 2 or more must not get through") and is now a plain
  passing arm, because `_check_a_range` walks the whole range.
* F2 (the rewritten registration assertion): `test_..._admits_a_literal_base_before_range` pins
  that the coder's two-fact predicate accepts `check-a main --range`, and
  `test_the_registered_prepush_entry_is_exactly_check_a_range` is the stricter form the criterion
  ("the entry carries no ref") needs, green on the real config.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_layout_version.py"
VERS = "src/hhemt/version_migration/versions"
FIXT = "tests/fixtures/legacy_layouts"
CONST = "src/hhemt/version_migration/constants.py"


def _run(repo: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run a COPY of the checker inside `repo` (REPO_ROOT is derived from the script's location)."""
    target = repo / "scripts" / "check_layout_version.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(SCRIPT.read_text())
    return subprocess.run(
        [sys.executable, str(target), *args],
        cwd=repo,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )


def _graph(tmp_path: Path, nodes: list[tuple[str, int, list[str]]], *, no_module_at: frozenset[str] = frozenset()):
    """Build a commit graph. `nodes` is `(name, version, parent_names)` in topological order.

    Every node's tree holds the migration module and both fixture directories for each version in
    `1..version` that the graph ever uses above the first, EXCEPT a node named in `no_module_at`,
    which omits its own `V{version}` module (the probe for "read the graded commit's tree").
    Returns `{name: sha}`. Nothing is checked out.
    """
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    every = sorted({v for _, v, _ in nodes})
    stream, marks = [], {}
    for i, (name, v, parents) in enumerate(nodes):
        files = {CONST: f"LAYOUT_VERSION: int = {v}\n"}
        for w in every:
            if w > every[0] and not (name in no_module_at and w == v):
                files[f"{VERS}/V{w:04d}__m.py"] = '"""m"""\n'
            if w > every[0]:
                for n in (w - 1, w):
                    files[f"{FIXT}/v{n}/.keep"] = ""
        marks[name] = i + 1
        head = f"commit refs/heads/all\nmark :{i + 1}\ncommitter t <t@t> {1700000000 + i} +0000\n"
        stream.append(f"{head}data 2\n{name[:2]}\n")
        if parents:
            stream.append(f"from :{marks[parents[0]]}\n")
            stream.extend(f"merge :{marks[p]}\n" for p in parents[1:])
        stream.append("deleteall\n")
        for rel, text in files.items():
            stream.append(f"M 100644 inline {rel}\ndata {len(text.encode())}\n{text}\n")
    marks_file = tmp_path / ".marks"
    subprocess.run(
        ["git", "fast-import", "--quiet", f"--export-marks={marks_file}"],
        cwd=tmp_path,
        input="".join(stream).encode(),
        check=True,
        capture_output=True,
    )
    by_mark = dict(line.split()[:2] for line in marks_file.read_text().splitlines())
    marks_file.unlink()
    return {name: by_mark[f":{m}"] for name, m in marks.items()}


def _push(repo: Path, shas: dict[str, str], from_name: str, to_name: str, *, remote: str = ""):
    env = {"PRE_COMMIT_FROM_REF": shas[from_name], "PRE_COMMIT_TO_REF": shas[to_name], "PRE_COMMIT_REMOTE_NAME": remote}
    return _run(repo, "check-a", "--range", env=env)


def test_check_a_range_a_squash_holding_three_bumps_is_named_at_that_commit(tmp_path: Path) -> None:
    """A squash push is ONE commit, so its jump is the push's jump: 31 -> 34 in one commit fails."""
    shas = _graph(tmp_path, [("c0", 30, []), ("c1", 31, ["c0"]), ("sq", 34, ["c1"])])
    out = _push(tmp_path, shas, "c1", "sq")
    assert out.returncode == 1 and "jumped from 31 to 34" in out.stderr and shas["sq"][:9] in out.stderr, out.stderr


def test_check_a_range_a_jump_in_the_middle_of_a_push_is_named_and_its_neighbours_pass(tmp_path: Path) -> None:
    """31 -> 32 -> 34 -> 35: only the middle commit is a jump, and it is the one named."""
    shas = _graph(tmp_path, [("c0", 31, []), ("a", 32, ["c0"]), ("b", 34, ["a"]), ("c", 35, ["b"])])
    out = _push(tmp_path, shas, "c0", "c")
    assert out.returncode == 1, out.stdout + out.stderr
    assert "jumped from 32 to 34" in out.stderr and shas["b"][:9] in out.stderr, out.stderr
    assert f"{shas['a'][:9]}: PASS" in out.stdout and f"{shas['c'][:9]}: PASS" in out.stdout, out.stdout


def test_check_a_range_reads_the_graded_commits_tree_not_a_later_commits(tmp_path: Path) -> None:
    """The pushed commit bumps by +1 but its OWN tree lacks the migration module; a LATER commit
    outside the range adds it. A read of the tree HEAD would pass; the graded commit's tree fails."""
    shas = _graph(
        tmp_path,
        [("c0", 30, []), ("c1", 31, ["c0"]), ("bad", 32, ["c1"]), ("later", 32, ["bad"])],
        no_module_at=frozenset({"bad"}),
    )
    out = _push(tmp_path, shas, "c1", "bad")
    assert out.returncode == 1 and "no migration module" in out.stderr and shas["bad"][:9] in out.stderr, out.stderr


def test_check_a_range_a_merge_of_a_stepwise_side_branch_passes(tmp_path: Path) -> None:
    """Control for the merge arms: side branch 31 -> 32 -> 33 in +1 steps, merged into a spine that
    did not bump. Every commit is legitimate, so the push passes whatever the walk covers."""
    shas = _graph(
        tmp_path,
        [
            ("c0", 31, []),
            ("s1", 32, ["c0"]),
            ("s2", 33, ["s1"]),
            ("c1", 31, ["c0"]),
            ("m", 33, ["c1", "s2"]),
        ],
    )
    out = _push(tmp_path, shas, "c0", "m")
    assert out.returncode == 0, out.stdout + out.stderr


def test_check_a_range_a_side_branch_jump_reached_only_through_a_merge_is_named(tmp_path: Path) -> None:
    """The criterion: a real jump of 2 or more must not get through the push. The side branch commit
    `s1` takes 31 -> 33 in one commit and has never been pushed (nothing on the remote). It arrives in
    the pushed range only as the second parent of the merge `m`."""
    shas = _graph(
        tmp_path,
        [("c0", 31, []), ("s1", 33, ["c0"]), ("c1", 31, ["c0"]), ("m", 33, ["c1", "s1"])],
    )
    out = _push(tmp_path, shas, "c0", "m")
    assert out.returncode == 1 and "jumped from 31 to 33" in out.stderr and shas["s1"][:9] in out.stderr, (
        out.stdout + out.stderr
    )


def test_check_a_range_a_side_branch_jump_already_on_the_named_remote_is_not_regraded(tmp_path: Path) -> None:
    """The reason a full walk would not double-grade: a commit already reachable from the remote was
    graded by the push that put it there. Green today and green under a full walk."""
    shas = _graph(
        tmp_path,
        [("c0", 31, []), ("s1", 33, ["c0"]), ("c1", 31, ["c0"]), ("m", 33, ["c1", "s1"])],
    )
    subprocess.run(["git", "update-ref", "refs/remotes/origin/feat", shas["s1"]], cwd=tmp_path, check=True)
    out = _push(tmp_path, shas, "c0", "m", remote="origin")
    assert out.returncode == 0, out.stdout + out.stderr


def _two_fact_predicate(entry: str) -> bool:
    """The coder's rewritten assertion, verbatim as a predicate (tests/test_check_layout_version.py,
    `test_layout_version_registrations_supply_the_range_predicate`)."""
    return entry.split()[-1] == "--range" and "HEAD~1" not in entry


def _strict_predicate(entry: str) -> bool:
    """What the criterion needs: the entry's argv after the script is exactly `check-a --range`."""
    return entry.split("check_layout_version.py ")[1].split() == ["check-a", "--range"]


def test_the_two_fact_registration_predicate_admits_a_literal_base_before_range() -> None:
    """F2: `check-a main --range` carries a literal base, which `main()` resolves ahead of
    PRE_COMMIT_FROM_REF (`base_ref or os.environ.get(...)`), restoring a wide base. The old assertion
    (`split()[-1] == "HEAD~1"`) rejected that entry; the rewritten pair accepts it."""
    mutant = "scripts/hook_env.sh python scripts/check_layout_version.py check-a main --range"
    assert _two_fact_predicate(mutant) is True
    assert _strict_predicate(mutant) is False
    good = "scripts/hook_env.sh python scripts/check_layout_version.py check-a --range"
    assert _two_fact_predicate(good) is True and _strict_predicate(good) is True


def test_a_literal_base_in_the_entry_overrides_the_pushed_range(tmp_path: Path) -> None:
    """Why F2 matters. The environment names the push `c1..c2` (33 -> 34, +1, no jump). Appending a
    literal base `main` (the root, v30) makes the same invocation grade `main..c2`, which holds the
    30 -> 32 jump at `j` that this push does not carry: the entry's base wins over the environment."""
    shas = _graph(tmp_path, [("c0", 30, []), ("j", 32, ["c0"]), ("c1", 33, ["j"]), ("c2", 34, ["c1"])])
    subprocess.run(["git", "branch", "-q", "-f", "main", shas["c0"]], cwd=tmp_path, check=True)
    env = {"PRE_COMMIT_FROM_REF": shas["c1"], "PRE_COMMIT_TO_REF": shas["c2"], "PRE_COMMIT_REMOTE_NAME": ""}
    env_only = _run(tmp_path, "check-a", "--range", env=env)
    with_base = _run(tmp_path, "check-a", "main", "--range", env=env)
    assert env_only.returncode == 0, env_only.stdout + env_only.stderr
    assert with_base.returncode == 1 and "jumped from 30 to 32" in with_base.stderr, with_base.stdout + with_base.stderr


def test_the_registered_prepush_entry_is_exactly_check_a_range() -> None:
    """The strict form of the rewritten assertion, on the real config. Green today."""
    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / ".pre-commit-config.yaml").read_text())
    entries = [
        h["entry"] for repo in cfg["repos"] for h in repo["hooks"] if h.get("id") == "check-layout-version-a-range"
    ]
    assert len(entries) == 1
    assert _strict_predicate(entries[0]), entries[0]
