"""The anonymization guard's generic pattern layer, its carrier-resolution rule and its CI step.

Every email address and per-user path this file plants is BUILT AT RUN TIME FROM FRAGMENTS,
never written as a literal, so this file is not itself a hit of the guard it tests (no
exemption row covers a test file). Tokens are synthetic (``ZZ...``); no producer-specific
value appears here.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

import scripts.check_anonymization as guard  # repo root is on sys.path under pytest
from tests import carrier_states as cs

REPO_ROOT = Path(guard.__file__).resolve().parent.parent
_SCRIPTS = ("check_anonymization.py", "check_commit_message_anonymization.py")
_CARRIER_TOKENS = [f"ZZCARRIERTOKEN{i:02d}" for i in range(1, 9)]


def _planted_email(local: str = "planted") -> str:
    return local + "@" + "example" + ".invalid"


def _planted_path(user: str = "someone") -> str:
    return "/" + "home" + "/" + user + "/data/run"


@pytest.fixture(autouse=True)
def _isolated_git_and_env(tmp_path, monkeypatch):
    """No developer carrier, git config or home leaks into a test."""
    monkeypatch.delenv("HHEMT_ANONYMIZATION_BLOCKLIST", raising=False)
    empty_cfg = tmp_path / "empty_gitconfig"
    empty_cfg.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty_cfg))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("HOME", str(tmp_path / "home_dir"))


def _carrier(tmp_path: Path, tokens: list[str] | None = None) -> Path:
    p = tmp_path / "carrier_outside_repo.txt"
    p.write_text("# synthetic\n" + "\n".join(tokens or _CARRIER_TOKENS) + "\n", encoding="utf-8")
    return p


def _repo(tmp_path: Path, files: dict[str, str], *, with_scripts: bool = False) -> Path:
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    if with_scripts:
        (root / "scripts").mkdir(exist_ok=True)
        for name in _SCRIPTS:
            shutil.copy(REPO_ROOT / "scripts" / name, root / "scripts" / name)
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root


# --------------------------------------------------------------------------------------
# Layer (a): the two pattern classes, with no carrier configured.
# --------------------------------------------------------------------------------------
def test_planted_email_fails_with_no_carrier(tmp_path, capsys) -> None:
    """Property: an address in a tracked file fails the guard with no carrier set, and its text is not
    printed. Class: NEW CAPABILITY (the parent exits on a missing carrier). Killed by: dropping the
    email class from the scan. A second correct implementation: any matcher of the same class that
    reports a class label.
    """
    root = _repo(tmp_path, {"src/leak.py": f"owner = '{_planted_email()}'\n"})
    rc = guard.main(["--root", str(root)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "src/leak.py:1" in err
    assert "email-address pattern" in err
    assert _planted_email() not in err  # a value is never printed


def test_planted_per_user_path_fails_with_no_carrier(tmp_path, capsys) -> None:
    """Property: a per-user absolute path fails the guard with no carrier and the user segment is not
    printed. Class: NEW CAPABILITY. Killed by: dropping the path class. A second correct implementation:
    a stricter segment pattern that still matches the planted segment.
    """
    root = _repo(tmp_path, {"src/leak.py": f"data = '{_planted_path()}'\n"})
    rc = guard.main(["--root", str(root)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "per-user absolute path pattern" in err
    assert "someone" not in err


def test_clean_tree_with_no_carrier_prints_one_notice_and_exits_zero(tmp_path, capsys) -> None:
    """Property: with no carrier a clean tree exits 0 and the guard prints exactly one notice. Class:
    REGRESSION (the parent exits non-zero here). Killed by: restoring the exit on a missing carrier or
    printing the notice twice. A second correct implementation: the notice on stdout, if the assertion
    read both streams.
    """
    root = _repo(tmp_path, {"src/ok.py": "import hhemt\nx = 1\n"})
    rc = guard.main(["--root", str(root)])
    err = capsys.readouterr().err
    assert rc == 0
    assert err.count("NOTICE") == 1
    assert "no private blocklist carrier is configured" in err


def test_the_toolkit_author_address_passes_and_another_address_still_fails(tmp_path, capsys) -> None:
    """Property: the address declared in pyproject.toml is exempt by value and any other address in the
    same file shape still fails. Class: NEW CAPABILITY. Killed by: exempting every address, or none. A
    second correct implementation: reading the address from the authors table only.
    """
    author = _planted_email("the.author")
    pyproject = f'[project]\nname = "x"\nauthors = [{{name = "A", email = "{author}"}}]\n'
    root = _repo(tmp_path, {"pyproject.toml": pyproject, "CITATION.cff": f"mail: {author}\n"})
    assert guard.main(["--root", str(root)]) == 0
    capsys.readouterr()
    # must-fire twin: the SAME file shape with a different address is a hit.
    root2 = _repo(
        tmp_path / "second", {"pyproject.toml": pyproject, "CITATION.cff": f"mail: {_planted_email('other')}\n"}
    )
    assert guard.main(["--root", str(root2)]) == 1


def test_exemption_is_keyed_on_path_and_class(tmp_path, capsys) -> None:
    """A (glob, class) pair passes; the other class in the same file, and the same class in a
    file the glob does not cover, still fail."""
    table = 'exemptions:\n  - glob: "docs/example.md"\n    class: "path"\n    reason: "documented example segment"\n'
    both = f"{_planted_path()}\n{_planted_email()}\n"
    root = _repo(
        tmp_path,
        {
            guard.EXEMPTIONS_RELPATH: table,
            "docs/example.md": f"{_planted_path()}\n",
            "docs/other_class.md": both,
            "docs/other_file.md": f"{_planted_path()}\n",
        },
    )
    hits = guard.scan(root, None)
    by_file = {(h.path, h.label.split(" ")[0]) for h in hits}
    assert ("docs/example.md", "per-user") not in by_file  # exempt pair passes
    assert ("docs/other_file.md", "per-user") in by_file  # same class, uncovered file
    assert {h.path for h in hits} == {"docs/other_class.md", "docs/other_file.md"}
    # "docs/other_class.md" is not covered by the glob at all; narrow the control to the class:
    table2 = table.replace("docs/example.md", "docs/other_class.md")
    root2 = _repo(tmp_path / "second", {guard.EXEMPTIONS_RELPATH: table2, "docs/other_class.md": both})
    labels = [h.label for h in guard.scan(root2, None)]
    assert labels == [guard._CLASS_LABEL[guard.CLASS_EMAIL]]  # path exempt, email still fires


def test_unrecognised_exemptions_text_refuses(tmp_path) -> None:
    """Property: an exemptions file outside the restricted format makes the guard refuse rather than
    silently load a thinner table. Class: NEW CAPABILITY. Killed by: skipping unparsable lines. A second
    correct implementation: a fuller YAML parser that rejects unquoted values.
    """
    root = _repo(tmp_path, {guard.EXEMPTIONS_RELPATH: "exemptions:\n  - glob: unquoted\n", "a.txt": "x\n"})
    with pytest.raises(SystemExit) as exc:
        guard.scan(root, None)
    assert "restricted exemptions format" in str(exc.value)


def test_the_public_blocklist_file_is_absent_from_the_tree() -> None:
    """Property: the repository carries no producer-specific list file. Class: REGRESSION (the parent
    tracks it). Killed by: re-adding the file. A second correct implementation: the check run over the
    tracked set instead of the file system.
    """
    assert not (REPO_ROOT / "scripts" / "anonymization_blocklist.txt").exists()


def test_exemptions_file_parses_identically_by_the_stdlib_parser_and_yaml() -> None:
    """Property: the stdlib-only parser and a real YAML parser read the committed table identically. Class:
    NEW CAPABILITY. Killed by: a hand edit that drifts out of the restricted subset in a way only one
    parser accepts. A second correct implementation: comparing against the JSON form.
    """
    path = REPO_ROOT / guard.EXEMPTIONS_RELPATH
    ours = [(e.glob, e.cls, e.reason) for e in guard.load_exemptions(path)]
    theirs = [(e["glob"], e["class"], e["reason"]) for e in yaml.safe_load(path.read_text("utf-8"))["exemptions"]]
    assert ours == theirs
    assert len(ours) > 30
    assert {c for _, c, _ in ours} <= set(guard.PATTERN_CLASSES)


def test_the_guard_run_over_the_whole_worktree_with_no_carrier_exits_zero(capsys) -> None:
    """The exemption table is total over the tracked set. The exemptions are what make it so: the
    control below shows the same files fire without them."""
    assert guard.main(["--root", str(REPO_ROOT)]) == 0
    capsys.readouterr()


@pytest.mark.parametrize(
    "rel, cls",
    [
        ("docs/how-to/reprex-roundtrip.md", guard.CLASS_PATH),
        ("tests/test_partition_split.py", guard.CLASS_EMAIL),
        ("src/hhemt/constants.py", guard.CLASS_PATH),
    ],
)
def test_exempted_files_fire_without_the_table(rel: str, cls: str) -> None:
    """Must-fire control for the totality claim: these tracked files DO carry the class, and
    the closed table is what exempts them."""
    table = guard.load_exemptions(REPO_ROOT / guard.EXEMPTIONS_RELPATH)
    assert cls in guard.exempt_classes(rel, table)
    text = (REPO_ROOT / rel).read_text(encoding="utf-8")
    fired = [guard.pattern_class_hits(line, frozenset({cls}), frozenset()) for line in text.splitlines()]
    assert any(fired), f"{rel} no longer carries class {cls}: its exemption row is stale"


def test_the_guards_own_sources_and_these_tests_are_not_hits() -> None:
    """Property: the guard scripts and their tests carry no literal the pattern classes fire on. Class: NEW
    CAPABILITY. Killed by: writing an address or a per-user path as a literal in any of them. A second
    correct implementation: scanning through the guard's own scan after the files are tracked.
    """
    allowed = guard.toolkit_author_addresses(REPO_ROOT)
    for rel in (
        "scripts/check_anonymization.py",
        "scripts/check_commit_message_anonymization.py",
        "tests/test_check_anonymization_patterns.py",
        "tests/carrier_states.py",
        "tests/test_check_anonymization.py",
    ):
        for line in (REPO_ROOT / rel).read_text(encoding="utf-8").splitlines():
            assert not guard.pattern_class_hits(line, frozenset(guard.PATTERN_CLASSES), allowed), rel


def test_pattern_controls() -> None:
    """Property: each root and the Windows form fire the path class, and a placeholder token, a bare at-
    sign and a relative path fire neither class. Class: NEW CAPABILITY. Killed by: removing a root or
    widening a class. A second correct implementation: a different segment pattern with the same
    verdicts on these inputs.
    """
    all_classes = frozenset(guard.PATTERN_CLASSES)
    fire = lambda s: guard.pattern_class_hits(s, all_classes, frozenset())  # noqa: E731
    assert fire(_planted_email()) == [guard.CLASS_EMAIL]
    assert fire(_planted_path()) == [guard.CLASS_PATH]
    for root_name in ("Users", "scratch", "project", "gpfs"):
        assert fire("/" + root_name + "/" + "somebody" + "/x") == [guard.CLASS_PATH]
    assert fire("C:" + chr(92) + "Users" + chr(92) + "somebody" + chr(92) + "x") == [guard.CLASS_PATH]
    # over-fire controls: a placeholder token, a bare at-sign and a relative path match neither class
    # the Windows alternative: lower case, forward slash and the JSON-escaped doubled backslash all
    # fire (the must-fire controls), while a drive path outside the users directory does not
    bs = chr(92)
    for form in (
        "c:" + bs + "users" + bs + "somebody" + bs + "x",
        "/".join(("C:", "Users", "somebody", "x")),
        "/".join(("c:", "users", "somebody", "x")),
        "C:" + bs * 2 + "Users" + bs * 2 + "somebody" + bs * 2 + "x",
    ):
        assert fire(form) == [guard.CLASS_PATH], form
    assert fire("C:" + bs + "Program Files" + bs + "tool") == []
    assert fire("{hhemt_home}/data/run") == []
    assert fire("a = b @ c") == []
    assert fire("see tests/data/run for the fixture") == []
    # the toolkit's own address is exempt by value only when the caller passes it
    assert guard.pattern_class_hits(_planted_email(), all_classes, frozenset({_planted_email()})) == []


def test_email_class_is_linear_on_a_long_run() -> None:
    """The literal non-possessive form stalls on the ~100 MB bundle pages."""
    run = "a" * 200_000
    start = time.monotonic()
    assert guard.pattern_class_hits(run, frozenset({guard.CLASS_EMAIL}), frozenset()) == []
    assert time.monotonic() - start < 2.0
    # must-fire twin: the same run followed by a valid domain is a hit
    assert guard.pattern_class_hits(run + "@x.com", frozenset({guard.CLASS_EMAIL}), frozenset()) == [guard.CLASS_EMAIL]


# --------------------------------------------------------------------------------------
# Layer (c): a synthetic carrier; a producer-specific value is never printed.
# --------------------------------------------------------------------------------------
def test_planted_synthetic_token_fails_and_its_text_is_in_neither_stream(tmp_path, monkeypatch, capsys) -> None:
    """Property: a planted token of a synthetic carrier fails the guard by index and length and its text is
    in neither stream. Class: REGRESSION (the parent prints the token). Killed by: printing the token. A
    second correct implementation: a different index base, with the assertion edited to match.
    """
    monkeypatch.setenv("HHEMT_ANONYMIZATION_BLOCKLIST", str(_carrier(tmp_path)))
    root = _repo(tmp_path, {"src/leak.py": "account = 'ZZCARRIERTOKEN03'\n"})
    rc = guard.main(["--root", str(root)])
    out = capsys.readouterr()
    assert rc == 1
    assert "ZZCARRIERTOKEN03" not in out.out + out.err
    assert "token #3 (16 chars)" in out.err
    assert "src/leak.py:1" in out.err


def test_list_mode_prints_lengths_not_tokens(tmp_path, monkeypatch, capsys) -> None:
    """Property: --list prints the token count and lengths and never a token. Class: REGRESSION (guards the
    existing masking). Killed by: printing a token. A second correct implementation: omitting the
    lengths.
    """
    monkeypatch.setenv("HHEMT_ANONYMIZATION_BLOCKLIST", str(_carrier(tmp_path)))
    root = _repo(tmp_path, {"src/ok.py": "x = 1\n"})
    assert guard.main(["--root", str(root), "--list"]) == 0
    out = capsys.readouterr()
    assert "token 3: 16 chars" in out.out
    assert "ZZCARRIERTOKEN" not in out.out + out.err


def test_unset_variable_degrades_and_a_nonexistent_variable_path_refuses(tmp_path, monkeypatch, capsys) -> None:
    """Property: an unset variable degrades to a notice and exit 0, and a variable naming a missing file
    refuses, naming the path and no token. Class: REGRESSION. Killed by: treating a missing file as
    unset. A second correct implementation: raising a subclass of SystemExit.
    """
    root = _repo(tmp_path, {"src/ok.py": "x = 1\n"})
    assert guard.main(["--root", str(root)]) == 0  # the unset arm
    capsys.readouterr()
    missing = tmp_path / "no_such_carrier.txt"
    monkeypatch.setenv("HHEMT_ANONYMIZATION_BLOCKLIST", str(missing))  # the must-fire twin
    with pytest.raises(SystemExit) as exc:
        guard.main(["--root", str(root)])
    assert str(missing) in str(exc.value.code)
    assert "ZZ" not in str(exc.value.code)


def _run_script(root: Path, script: str, *args: str, env_extra: dict[str, str] | None = None):
    env = dict(os.environ)
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, str(root / "scripts" / script), *args], cwd=root, env=env, capture_output=True, text=True
    )


def test_commit_message_guard_arms(tmp_path) -> None:
    """Property: the commit-message guard degrades with one notice, fails on an address or a per-user path
    without printing it, refuses a missing carrier, and names a token only by index. Class: REGRESSION
    and NEW CAPABILITY. Killed by: exiting on a missing carrier, or printing the value. A second correct
    implementation: the same checks through main() in process.
    """
    root = _repo(tmp_path, {"README.md": "x\n"}, with_scripts=True)
    clean = tmp_path / "clean.txt"
    clean.write_text("feat: tidy the thing\n", encoding="utf-8")
    dirty_email = tmp_path / "dirty_email.txt"
    dirty_email.write_text(f"fix: thanks {_planted_email()}\n", encoding="utf-8")
    dirty_path = tmp_path / "dirty_path.txt"
    dirty_path.write_text(f"fix: ran in {_planted_path()}\n", encoding="utf-8")
    dirty_token = tmp_path / "dirty_token.txt"
    dirty_token.write_text("fix: touched ZZCARRIERTOKEN05\n", encoding="utf-8")
    carrier = _carrier(tmp_path)
    script = "check_commit_message_anonymization.py"

    # no carrier: patterns only, one notice, exit 0 on a clean message
    r = _run_script(root, script, str(clean))
    assert (r.returncode, r.stderr.count("NOTICE")) == (0, 1)
    # a planted address / path exits non-zero and the value is not printed
    for msg, value in ((dirty_email, _planted_email()), (dirty_path, "someone")):
        r = _run_script(root, script, str(msg))
        assert r.returncode == 1 and value not in r.stdout + r.stderr
    # a Co-Authored-By trailer carries an address and is refused like any other: nothing is exempt
    trailer = tmp_path / "trailer.txt"
    trailer.write_text("fix: tidy\n\nCo-Authored-By: Someone <" + _planted_email("co.author") + ">\n", encoding="utf-8")
    r = _run_script(root, script, str(trailer))
    assert r.returncode == 1
    assert "email-address pattern" in r.stderr and "co.author" not in r.stdout + r.stderr
    # under `git commit -v` everything from the scissors line down is discarded by git: an address on
    # an added diff line below it does not reject, the same text above it does
    scissors = "# " + "-" * 24 + " >8 " + "-" * 24
    template = "\n".join(f"# template comment line {i}" for i in range(13))
    diff = "diff --git a/f b/f\n+owner = '" + _planted_email("added.line") + "'\n"
    below = tmp_path / "below.txt"
    below.write_text(f"fix: tidy\n\n{template}\n{scissors}\n{diff}", encoding="utf-8")
    r = _run_script(root, script, str(below))
    assert r.returncode == 0, r.stderr
    above = tmp_path / "above.txt"
    above.write_text(f"fix: tidy {_planted_email('added.line')}\n\n{template}\n{scissors}\n", encoding="utf-8")
    assert _run_script(root, script, str(above)).returncode == 1
    # `#` lines written with -m stay scanned (git keeps them), so an address in one rejects
    hashline = tmp_path / "hashline.txt"
    hashline.write_text(f"fix: tidy\n\n# note {_planted_email('hash.line')}\n", encoding="utf-8")
    assert _run_script(root, script, str(hashline)).returncode == 1
    # a configured but nonexistent carrier refuses (the must-fire twin of the unset arm)
    missing = tmp_path / "no_such_carrier.txt"
    r = _run_script(root, script, str(clean), env_extra={"HHEMT_ANONYMIZATION_BLOCKLIST": str(missing)})
    assert r.returncode != 0 and str(missing) in r.stderr and "Traceback" not in r.stderr
    # a synthetic carrier: a planted token fails, by index and length, never by text
    r = _run_script(root, script, str(dirty_token), env_extra={"HHEMT_ANONYMIZATION_BLOCKLIST": str(carrier)})
    assert r.returncode == 1
    assert "token #5 (16 chars)" in r.stderr and "ZZCARRIERTOKEN05" not in r.stdout + r.stderr


def test_file_guard_subprocess_pins_the_exit_status_of_each_outcome_class(tmp_path) -> None:
    """Property: the real script exits 0, 1 and 1 for degrade, a usable carrier with a hit, and a refusal.
    Class: NEW CAPABILITY. Killed by: mapping a refusal to exit 0. A second correct implementation: a
    different non-zero code for the refusal, with the pin edited.
    """
    root = _repo(tmp_path, {"src/leak.py": "account = 'ZZCARRIERTOKEN02'\n"}, with_scripts=True)
    clean_root = _repo(tmp_path / "clean", {"src/ok.py": "x = 1\n"}, with_scripts=True)
    carrier = _carrier(tmp_path)
    degrade = _run_script(clean_root, "check_anonymization.py")
    usable_hit = _run_script(root, "check_anonymization.py", env_extra={"HHEMT_ANONYMIZATION_BLOCKLIST": str(carrier)})
    refuse = _run_script(
        clean_root, "check_anonymization.py", env_extra={"HHEMT_ANONYMIZATION_BLOCKLIST": str(tmp_path / "nope.txt")}
    )
    assert (degrade.returncode, usable_hit.returncode, refuse.returncode) == (0, 1, 1)
    assert "Traceback" not in refuse.stderr and "NOTICE" in degrade.stderr


# --------------------------------------------------------------------------------------
# CR: the state space is GENERATED and every cell is run through the real resolver and loader.
# --------------------------------------------------------------------------------------
def test_state_space_sizes_and_builder_coverage(tmp_path) -> None:
    """Property: the generated state space has the stated sizes, the builder covers every state and the
    evaluator is defined on every cell. Class: NEW CAPABILITY. Killed by: adding a state without a
    builder or an evaluator rule. A second correct implementation: deriving the sets from the loader's
    branches.
    """
    assert len(cs.PER_TIER_STATES) == 24 and len(set(cs.PER_TIER_STATES)) == 24
    assert len(cs.KEY_TIER_STATES) == 26
    assert len(cs.PER_TIER_STATES) ** 2 * len(cs.KEY_TIER_STATES) == 14_976
    try:
        built = {s for s in cs.PER_TIER_STATES if cs.build_value(s, tmp_path / "t", tmp_path / "h") is not None}
        assert built == set(cs.PER_TIER_STATES)  # the builder's keys equal PER_TIER_STATES
        keys = {s for s in cs.KEY_TIER_STATES if cs.build_key_config(s, tmp_path / "k", tmp_path / "h")}
        assert keys == set(cs.KEY_TIER_STATES)
    finally:
        for locked in tmp_path.rglob("locked_dir"):  # let pytest remove the unsearchable directories
            locked.chmod(0o700)
    assert set(cs.USABLE_TOKEN_COUNT) <= set(cs.PER_TIER_STATES)
    assert set(cs.NOT_CONFIGURED) <= set(cs.PER_TIER_STATES)
    for f in ("unset", "empty", "well_formed", "mistyped_path"):  # the evaluator defines every cell
        for v in cs.PER_TIER_STATES:
            for k in cs.KEY_TIER_STATES:
                assert cs.evaluate(f, v, k).outcome in {"degrade", "usable", "refuse"}


def test_every_cell_of_the_repository_guard_state_space(tmp_path, monkeypatch) -> None:
    """Property: all 14,976 cells of flag x variable x key, run through the real resolver and loader, match
    the reference evaluator, including precedence and refusals. Class: REGRESSION (the parent falls
    through an unusable tier) and NEW CAPABILITY. Killed by: skipping an unusable tier, accepting a
    relative path, ignoring a corrupt git config, or counting duplicates toward the floor. A second
    correct implementation: a different order of checks inside the loader with the same verdicts.
    """
    home = tmp_path / "home_dir"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "relative_carrier.txt").write_text("# synthetic\n" + "\n".join(_CARRIER_TOKENS) + "\n", encoding="utf-8")
    monkeypatch.chdir(cwd)
    root = tmp_path / "plain_root"  # not a git repository: only the global config is read
    root.mkdir()
    no_git = tmp_path / "no_git_on_path"
    no_git.mkdir()

    flag = {s: cs.build_value(s, tmp_path / "flag", home) for s in cs.PER_TIER_STATES}
    variable = {s: cs.build_value(s, tmp_path / "variable", home) for s in cs.PER_TIER_STATES}
    key = {s: cs.build_key_config(s, tmp_path / "key", home) for s in cs.KEY_TIER_STATES}

    real_path = os.environ["PATH"]
    monkeypatch.setenv("PATH", real_path)
    monkeypatch.setenv("HHEMT_ANONYMIZATION_BLOCKLIST", "placeholder")  # so teardown restores the unset state
    checked = skipped = 0
    try:
        for fs in cs.PER_TIER_STATES:
            for vs in cs.PER_TIER_STATES:
                for ks in cs.KEY_TIER_STATES:
                    cfg, key_ok, git_off, key_value = key[ks]
                    if not (flag[fs].buildable and variable[vs].buildable and key_ok):
                        skipped += 1
                        continue
                    if variable[vs].value is None:
                        os.environ.pop("HHEMT_ANONYMIZATION_BLOCKLIST", None)
                    else:
                        os.environ["HHEMT_ANONYMIZATION_BLOCKLIST"] = variable[vs].value
                    os.environ["GIT_CONFIG_GLOBAL"] = str(cfg)
                    os.environ["PATH"] = str(no_git) if git_off else real_path
                    exp = cs.evaluate(fs, vs, ks)
                    err = None
                    res = None
                    tokens: list[str] = []
                    try:
                        res = guard._resolve_carrier(root, flag[fs].value)
                        if res.path is not None:
                            tokens = guard.load_blocklist(res.path, res.tier or "the carrier")
                    except SystemExit as e:
                        err = str(e.code)
                    cell = (fs, vs, ks)
                    if exp.outcome == "usable":
                        assert err is None and res is not None, (cell, err)
                        assert (res.tier, len(tokens)) == (exp.tier, exp.tokens), cell
                    elif exp.outcome == "degrade":
                        assert err is None and res is not None and res.path is None, (cell, err)
                        assert res.empty_tiers == exp.empty_tiers, cell
                        notice = guard.no_carrier_notice(res.empty_tiers)
                        assert ("set but empty" in notice) == bool(exp.empty_tiers), cell
                        assert all(t in notice for t in exp.empty_tiers), cell
                    else:
                        assert err is not None, cell
                        assert exp.tier in err and "Traceback" not in err and "ZZ" not in err, (cell, err)
                        deciding = {
                            cs.TIER_FLAG: flag[fs].value,
                            cs.TIER_ENV: variable[vs].value,
                            cs.TIER_KEY: key_value,
                        }[exp.tier]
                        if deciding is not None:
                            assert str(Path(deciding)) in err, (cell, err)
                    checked += 1
    finally:
        os.environ["PATH"] = real_path
        for locked in tmp_path.rglob("locked_dir"):  # let pytest remove the unsearchable directories
            locked.chmod(0o700)
    assert checked + skipped == 14_976
    if os.geteuid() != 0:
        assert skipped == 0


# --------------------------------------------------------------------------------------
# CI: the secret step cannot leak the list.
# --------------------------------------------------------------------------------------
def _workflow_steps():
    wf = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "anonymization-guard.yml").read_text("utf-8"))
    return wf["jobs"]["anonymization-guard"]["steps"]


def test_the_ci_secret_step_is_conditioned_masked_and_never_echoes() -> None:
    """Property: over the parsed workflow the secret is passed only as env, one step sets the variable and
    only when the secret is non-empty, masks come before the export, and nothing echoes the content.
    Class: NEW CAPABILITY. Killed by: moving the secret expression into a run script or exporting before
    masking. A second correct implementation: a single step with an emptiness guard in its script.
    """
    steps = _workflow_steps()
    secret = "secrets.HHEMT_ANONYMIZATION_BLOCKLIST_CONTENT"
    # the secret expression appears in no run script
    assert all("${{ secrets" not in s.get("run", "") for s in steps)
    # it is passed only as an env: variable
    carriers = [s for s in steps if secret in str(s.get("env", {}))]
    assert len(carriers) == 2
    # exactly one step sets the guard's variable, and it runs only when the secret is non-empty
    setters = [s for s in steps if "HHEMT_ANONYMIZATION_BLOCKLIST=" in s.get("run", "")]
    assert len(setters) == 1
    setter = setters[0]
    probe = next(s for s in steps if s.get("id") == "blocklist_secret")
    assert setter["if"] == f"steps.{probe['id']}.outputs.present == 'true'"
    assert '[ -n "$BLOCKLIST_CONTENT" ]' in probe["run"]
    run = setter["run"]
    assert run.index("umask 077") < run.index("printf '%s' \"$BLOCKLIST_CONTENT\"")
    assert "chmod 600" in run and "$RUNNER_TEMP" in run
    assert run.index("::add-mask::") < run.index("GITHUB_ENV")  # masks registered before export
    # nothing echoes the content
    assert not re.search(r"echo[^\n]*BLOCKLIST_CONTENT", "\n".join(s.get("run", "") for s in steps))
    # the guard step comes after the setter and runs the script bare
    names = [s.get("name") for s in steps]
    assert names.index("Run anonymization guard") > steps.index(setter)
    assert steps[-1]["run"] == "python scripts/check_anonymization.py"
