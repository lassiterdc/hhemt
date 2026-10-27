"""The live-pin registration guard is WIRED, not merely present in `scripts/`.

THE PROBLEM THIS FILE SOLVES. `scripts/check_live_pin_registered.py` exists to refuse a
commit whose `TRITON_PIN` is not recorded as assessed in `src/hhemt/model_defects.py`. A
guard that is never invoked refuses nothing, and for most of this campaign that was its
actual state: its own declaration comment in `tests/fixtures/_triton_source_cache.py` read
"as of this move that guard is wired into NOTHING", with its only runner being
`tests/test_check_live_pin_registered.py` -- i.e. exactly the vehicle the guard's own
docstring rules out, because a pytest target can be deselected by a marker expression and a
compile-tier target can be skipped by venue.

So there are two separable properties and they fail independently. The guard's LOGIC is
exercised by `tests/test_check_live_pin_registered.py`. The guard's REGISTRATION -- that it
is attached to a venue which actually runs, at the one venue where reading the worktree is
sound -- has no other asserting surface, and that is this file's whole subject.

WHAT THIS FILE DELIBERATELY DOES NOT DO: it does not re-assert anything about the pin, the
registry, or the guard's verdicts. Those belong to the two modules named above, and
duplicating them here would create a second source for one predicate.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The three operands this file asserts an agreement between. Named as constants so a rename
#: breaks ONE line rather than being silently searched for in prose.
HOOK_ID = "check-live-pin-registered"
GUARD_REL = "scripts/check_live_pin_registered.py"
PRECOMMIT_REL = ".pre-commit-config.yaml"

#: The venue the guard is sound at, and the reason it is exactly one. The guard reads the
#: WORKTREE -- it resolves both tracked files from its own `REPO_ROOT` and `read_text()`s them.
#: `.pre-commit-config.yaml`'s own venue-split comment establishes that pre-push is not a
#: clean-tree venue (`stash = not all_files and not files` reverts unstaged edits while
#: staged-uncommitted ones survive), so a worktree read there is unsound in both directions.
INTENDED_STAGES = ["pre-commit"]


def _registration(hooks: list[dict]) -> dict:
    """The one `HOOK_ID` hook, or a legible failure naming its absence.

    A bare `next(...)` raises `StopIteration`, which reports as an internal error rather than
    as "the guard is not registered" -- measured on the pre-change config, where two arms of
    this file failed that way and named nothing a reader could act on.
    """
    matching = [h for h in hooks if h["id"] == HOOK_ID]
    assert matching, (
        f"no {HOOK_ID!r} registration in {PRECOMMIT_REL}: the guard at {GUARD_REL} is inert. "
        "test_the_guard_is_registered_at_its_intended_stages carries the full diagnosis."
    )
    return matching[0]


@pytest.fixture(scope="module")
def hooks() -> list[dict]:
    """Every hook declared in `.pre-commit-config.yaml`, flattened across repos.

    Module-scoped because the file is read-only here and parsing it per test buys nothing.
    """
    cfg = yaml.safe_load((REPO_ROOT / PRECOMMIT_REL).read_text(encoding="utf-8"))
    return [h for repo in cfg["repos"] for h in repo["hooks"]]


def test_the_config_parses_to_a_populated_hook_list(hooks: list[dict]) -> None:
    """Property: the fixture yields a non-empty hook list carrying ids.

    Class: NEW CAPABILITY. The plausible wrong implementation is a fixture that silently
    yields `[]` -- from a renamed config, a restructured `repos:` key, or a YAML load that
    returned None. Red under it: this test fails first, in file order.

    Why it is required: it is the VACUITY GUARD for every assertion below. Each of those is a
    membership test over this list, and a membership test over an empty list reports exactly
    like a clean one. Without this arm a config rename would turn the rest of this file green.

    What kills it: pointing `PRECOMMIT_REL` at a path that does not exist, or a `repos:`
    restructure that stops yielding dicts with `id`.

    A second correct implementation under which it still passes: any config with any number of
    hooks in any order, under any number of `repo:` blocks -- the assertion is on population
    and shape, never on membership or count.
    """
    assert hooks, f"{PRECOMMIT_REL} parsed to zero hooks; every assertion below would pass vacuously"
    assert all("id" in h for h in hooks), "a hook entry carries no id, so id-keyed lookup below is unsound"


def test_the_guard_is_registered_at_its_intended_stages(hooks: list[dict]) -> None:
    """Property: exactly one hook declares `HOOK_ID`, its entry invokes the guard script, and
    its `stages` is exactly `INTENDED_STAGES`.

    Class: REGRESSION. On the pre-change tree `.pre-commit-config.yaml` carried no
    registration for this guard at all -- `grep -c check_live_pin_registered` over that file
    returned 0 -- so this test red on the tree as it stood before the wiring landed.

    Why it is required: a guard in `scripts/` with no registration is inert, and its inertness
    is invisible. The two failure directions are different and both are covered here: NO
    registration leaves the invariant unenforced, and an ALL-STAGES registration (the pre-commit
    default when `stages:` is omitted) reaches pre-push, where the guard's worktree read is
    unsound in both directions for the reason `INTENDED_STAGES` records.

    What kills it: deleting the hook block; dropping its `stages:` key so it inherits the
    all-stages default; adding a second registration at another venue without giving this guard
    a range-based mode that is sound there.

    A second correct implementation under which it still passes: any hook `name:`, any position
    in the file, any `repo: local` block, and any interpreter prefix that still names the guard
    script -- the assertion turns on the id, the script the entry names, and the venue, none of
    which is wording this change introduced.
    """
    matching = [h for h in hooks if h["id"] == HOOK_ID]
    assert len(matching) == 1, f"expected exactly one {HOOK_ID!r} registration, found {len(matching)}"
    hook = matching[0]

    entry = hook.get("entry", "")
    assert GUARD_REL in entry, f"{HOOK_ID}'s entry does not name {GUARD_REL}: {entry!r}"

    stages = hook.get("stages")
    assert stages is not None, (
        f"{HOOK_ID} carries no explicit stages:, so it runs at EVERY installed venue by "
        "pre-commit's all-stages default -- including pre-push, where this guard's worktree "
        "read is unsound. Declare stages: [pre-commit]."
    )
    assert stages == INTENDED_STAGES, f"{HOOK_ID} declares {stages}; the sound venue set is {INTENDED_STAGES}"


def test_the_registration_fires_on_every_commit_rather_than_on_a_file_filter(hooks: list[dict]) -> None:
    """Property: the registration carries `always_run: true` and `pass_filenames: false`, and
    declares no `files:` filter.

    Class: NEW CAPABILITY. The plausible wrong implementation is a `files:`-filtered
    registration -- e.g. `files: ^tests/fixtures/_triton_source_cache\\.py$` -- which looks
    tighter and is the natural thing to reach for. Red under it: `files` is present and the
    `always_run` assertion fails.

    Why it is required, and it is the subtlest of the three: the invariant spans TWO files. The
    defect it exists to catch is a pin moved in one file while the registry edit is forgotten in
    the other, so the commit that needs refusing may stage only ONE of them -- or, after a
    registry-only edit, neither of the pair in a way a path filter anticipates. A filtered hook
    would skip exactly the commit whose asymmetry is the defect. `pass_filenames: false` is the
    matching half: the guard takes an optional `--root` and no file arguments, so a filename list
    would be passed as positional arguments it does not accept.

    What kills it: adding a `files:` key; dropping `always_run`; setting `pass_filenames: true`.

    A second correct implementation under which it still passes: one that additionally declares
    `exclude:`, `language:`, or a `name:` of any wording -- the assertion is on the three keys
    that decide WHETHER the hook runs and what it is handed, not on the rest of the block.
    """
    hook = _registration(hooks)
    assert hook.get("always_run") is True, f"{HOOK_ID} is not always_run; a staged-path filter can skip the defect"
    assert hook.get("pass_filenames") is False, (
        f"{HOOK_ID} passes filenames; the guard accepts no positional file arguments"
    )
    assert "files" not in hook, (
        f"{HOOK_ID} declares a files: filter. The invariant spans two files and the commit that "
        "needs refusing may stage only one of them, so a path filter skips the asymmetry that IS "
        "the defect."
    )


def test_the_registered_script_exists_and_the_entry_honours_the_interpreter_contract(hooks: list[dict]) -> None:
    """Property: the script the entry names exists on disk, and the entry runs it through
    `uv run --locked`.

    Class: NEW CAPABILITY. The plausible wrong implementation is a registration naming a script
    path that was renamed or never added -- pre-commit then fails at invocation time with an
    executable-not-found, which reads like an environment problem rather than a dangling
    registration. Red under it: the `is_file()` assertion fails and names the missing path.

    Why it is required: this file's other arms assert that a STRING appears in the config. Only
    this one asserts the string resolves to a real script, and only this one asserts the
    interpreter. `.pre-commit-config.yaml`'s INTERPRETER CONTRACT comment records why the prefix
    is load-bearing: a git hook inherits no shell state and Debian/Ubuntu ship no unversioned
    `python`, so a bare `python scripts/...` entry aborts on a stock machine, and a conda pin
    would resolve `hhemt` to the MAIN clone's `src` and grade a worktree's files against main's
    code.

    What kills it: renaming or deleting the guard script without updating the entry; replacing
    the `uv run --locked` prefix with a bare `python` or a conda invocation.

    A second correct implementation under which it still passes: any entry that adds further
    `uv` flags or script arguments after the prefix -- the assertion is on the prefix and on the
    named path resolving, not on the entry being exactly one string.
    """
    hook = _registration(hooks)
    entry = hook["entry"]
    # The path is taken FROM THE ENTRY rather than from GUARD_REL. Asserting that the constant
    # resolves would test this file's own constant and pass over an entry naming a renamed
    # script, since `GUARD_REL in entry` is a SUBSTRING test and
    # `scripts/check_live_pin_registered.py.RENAMED` satisfies it.
    named = [tok for tok in entry.split() if tok.endswith(".py")]
    assert len(named) == 1, f"{HOOK_ID}'s entry names {len(named)} .py paths, expected 1: {entry!r}"
    assert (REPO_ROOT / named[0]).is_file(), (
        f"{HOOK_ID}'s entry names {named[0]!r}, which does not exist on disk -- a dangling "
        "registration fails at invocation time and reads like an environment problem"
    )
    assert entry.startswith("uv run --locked "), (
        f"{HOOK_ID}'s entry is {entry!r}; every local hook in this config runs through "
        "`uv run --locked` for the reasons that file's INTERPRETER CONTRACT comment records"
    )
