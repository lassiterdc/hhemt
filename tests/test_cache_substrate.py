"""The cache substrate: one derivation of each root, one address per checkout, four marker arms.

These are unit tests over `hhemt.cache`, deliberately independent of the synthetic fixtures:
the substrate is what the toolkit reports its own cache from (`hhemt cache status`), so it has
to be checkable without a pytest session that shares a cache. The heavy properties -- that a
cached tree is rebuilt only when it should be -- live in the fixture tier and are not
duplicated here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hhemt.cache import paths as cache_paths

_REPO_ROOT = Path(__file__).resolve().parents[1]

# The derivation token is assembled from two pieces ON PURPOSE. Spelled contiguously it would
# appear in this file's own source and the scan below would report this test as a derivation
# site -- a matcher that finds itself is the classic unbounded instrument.
_DERIVATION_TOKEN = "user_cache_dir(" + '"hhemt")'


def _fake_checkout(base: Path) -> Path:
    """A directory tree with a `.git` entry, which is all `checkout_root` asks for.

    A real `git init` is deliberately avoided: the property under test is the ROOT
    RESOLUTION, and a real repository would additionally test git.
    """
    root = base / "checkout"
    (root / ".git").mkdir(parents=True)
    (root / "tests" / "fixtures").mkdir(parents=True)
    return root


def _tier_line(output: str) -> str:
    """The single `tier_address:` value the status verb printed, from captured output."""
    lines = [line for line in output.splitlines() if "tier_address:" in line]
    assert len(lines) == 1, output
    return lines[0].split("tier_address:", 1)[1].strip()


def test_cache_root_is_derived_in_exactly_one_place():
    """Property: no module under `src/` or `tests/` derives the platform cache root except
    `hhemt.cache.paths`.

    Class: REGRESSION -- red on the pre-change tree, where five files carried the derivation.

    Why it is required: a producer writing into one root while a sweeper deletes another is a
    data-loss bug that presents as an unexplained cache miss, and it is invisible in review
    because each copy is individually correct.

    What kills it: adding a second `platformdirs.user_cache_dir(...)` call anywhere under
    `src/` or `tests/`, including in a test's own fixture setup.

    A second correct implementation that still passes: deriving the root through
    `importlib.metadata` application dirs, or through `os.environ["XDG_CACHE_HOME"]`
    directly, yields one path -- this test measures the COUNT of derivations at the substrate,
    not which library performs it.
    """
    hits = sorted(
        path.relative_to(_REPO_ROOT).as_posix()
        for tree in (_REPO_ROOT / "src", _REPO_ROOT / "tests")
        for path in tree.rglob("*.py")
        if _DERIVATION_TOKEN in path.read_text(encoding="utf-8", errors="replace")
    )
    assert hits == ["src/hhemt/cache/paths.py"]


def test_tier_address_follows_the_checkout_root_not_the_invocation_directory(tmp_path):
    """Property: every directory inside one checkout addresses the SAME tier.

    Class: NEW CAPABILITY -- the plausible wrong implementation is "digest the path you were
    handed", which is what the pre-change code did; under it this assertion fails because the
    two paths digest differently.

    Why it is required: `pytest tests/` does not change directory, so a
    working-directory-rooted address gives one checkout two addresses, each cold for the
    other, and the status verb then reports a machine that looks cold while its trees sit on
    disk.

    What kills it: removing the root resolution from `tier_address_for`.

    A second correct implementation that still passes: resolving the root with
    `git rev-parse --show-toplevel` instead of the `.git`-ancestor walk yields the same root
    for a real checkout, and the address is then minted identically.
    """
    root = _fake_checkout(tmp_path)
    assert cache_paths.tier_address_for(root) == cache_paths.tier_address_for(root / "tests" / "fixtures")


def test_two_checkouts_get_different_addresses(tmp_path):
    """Property: two checkouts never collide onto one tier address.

    Class: NEW CAPABILITY -- a plausible wrong implementation is a constant or a
    basename-only address, under which both checkouts return the same value.

    Why it is required: the failure it prevents was measured -- three cluster checkouts fell
    onto one tier and shared one compiled-solver directory, so a rebuild in one corrupted
    another.

    What kills it: any address derivation that discards the full path.

    A second correct implementation that still passes: a longer digest, or a digest of the
    checkout's `.git` directory path, both keep the two distinct.
    """
    first = _fake_checkout(tmp_path / "one")
    second = _fake_checkout(tmp_path / "two")
    assert cache_paths.tier_address_for(first) != cache_paths.tier_address_for(second)


def test_worktree_address_is_the_worktree_name_and_the_fixture_facade_agrees(tmp_path):
    """Property: under `.claude/worktrees/{name}/` the address is `{name}`, and the fixture
    facade (`checkout_slug`) returns exactly what the substrate mints.

    Class: REGRESSION -- the worktree arm is pre-existing behaviour that the relocation must
    preserve; the equivalence half reds if the facade and the substrate drift apart.

    Why it is required: the run-dir reaper's worktree arm matches a tier name against a git
    worktree basename, so an address that stops being the worktree name silently removes a
    live tier's protection.

    What kills it: routing one of the two entry points around the substrate.

    A second correct implementation that still passes: `checkout_slug` may delegate by
    re-exporting the substrate function rather than by calling it, and the names still agree.
    """
    from tests.fixtures._triton_source_cache import checkout_slug

    root = tmp_path / ".claude" / "worktrees" / "wt-name"
    (root / ".git").mkdir(parents=True)
    (root / "tests").mkdir()
    assert cache_paths.tier_address_for(root) == "wt-name"
    assert checkout_slug(root.resolve()) == "wt-name"


def test_roots_nest_under_one_cache_root():
    """Property: all four roots are named relative to ONE cache root, at the depths the
    substrate documents.

    Class: NEW CAPABILITY -- a plausible wrong implementation is T2 placed beside T1 instead
    of inside its address directory, under which the reaper's `_NEVER_REAP` nesting
    assumption stops holding.

    Why it is required: the per-checkout/per-machine split is what keeps two checkouts from
    sharing a run tree, and the nesting is what a reaper's exclusion list is written against.

    What kills it: moving any root to a different parent.

    A second correct implementation that still passes: any naming of the intermediate
    directories is free; only the parentage asserted here is constrained.
    """
    cache = cache_paths.cache_root()
    assert cache_paths.synthetic_models_root() == cache / "synthetic_test_models"
    assert cache_paths.synthetic_runs_root() == cache / "synthetic_test_runs"
    assert cache_paths.slug_runs_root("addr") == cache / "synthetic_test_runs" / "addr"
    assert cache_paths.software_root("addr") == cache / "synthetic_test_runs" / "addr" / "_software"
    assert cache_paths.canonical_root() == cache / "_triton_canonical" / "triton"


def test_marker_state_machine_arms(tmp_path):
    """Property: the four marker arms each fire on their own input, and ARM 0 raises rather
    than marking when its own postcondition is unmet.

    Class: REGRESSION -- red on the pre-change tree, where `hhemt.cache.markers` does not
    exist (`ModuleNotFoundError`); the arms themselves are behaviour the fixtures depend on
    and must survive the relocation unchanged.

    Why it is required: an arm that silently collapses into another turns "the tree is
    incomplete" into "hand out a broken tree", which is the failure the marker exists to
    prevent (ten tests blamed for a fixture's precondition).

    What kills it: dropping the provenance comparison (`stale` collapses into `complete`), or
    dropping the payload check (`unsatisfied` collapses into `complete`).

    A second correct implementation that still passes: storing the payload as a dict, or
    comparing provenance per key rather than by dict equality, preserves all four arms.
    """
    from hhemt.cache import markers

    analysis = tmp_path / "analysis"
    analysis.mkdir()
    payload = ("wanted.txt",)
    provenance = {"builder_digest": "digest-a", "toolkit_sha": "sha-a"}

    assert markers.marker_state(analysis, payload, provenance=provenance) == "absent"
    (analysis / "wanted.txt").write_text("x", encoding="utf-8")
    markers.finish_build(analysis, payload, "the_fixture", provenance=provenance)
    assert markers.marker_state(analysis, payload, provenance=provenance) == "complete"
    other = {"builder_digest": "digest-b", "toolkit_sha": "sha-a"}
    assert markers.marker_state(analysis, payload, provenance=other) == "stale"

    (analysis / "wanted.txt").unlink()
    assert markers.marker_state(analysis, payload, provenance=provenance) == "unsatisfied"
    with pytest.raises(RuntimeError, match="postcondition is UNMET"):
        markers.finish_build(analysis, payload, "the_fixture", provenance=provenance)


def test_cache_status_reports_the_same_tier_from_a_subdirectory(tmp_path, monkeypatch):
    """Property: `hhemt cache status` names one tier address, and it is the same one whether
    the verb is invoked from a checkout root or from a subdirectory of that checkout.

    Class: NEW CAPABILITY -- the plausible wrong implementation is a
    working-directory-rooted address, under which the two invocations report different
    addresses.

    Why it is required: this verb is the evidence surface the cache's own claims are read
    against, and a verb that answers differently depending on where it was run reports a
    checkout as cold while its trees are on disk.

    What kills it: deriving the address from `Path.cwd()` in the verb, or dropping the
    `cache status` subcommand.

    A second correct implementation that still passes: any verb whose report derives the
    address from the checkout root preserves this; the assertions read only the reported
    address, not the table's shape.
    """
    from typer.testing import CliRunner

    from hhemt.cli import app

    root = _fake_checkout(tmp_path)
    runner = CliRunner()

    monkeypatch.chdir(root)
    from_root = runner.invoke(app, ["cache", "status"])
    monkeypatch.chdir(root / "tests" / "fixtures")
    from_subdirectory = runner.invoke(app, ["cache", "status"])

    assert from_root.exit_code == 0, from_root.output
    assert from_subdirectory.exit_code == 0, from_subdirectory.output
    assert _tier_line(from_root.output) == _tier_line(from_subdirectory.output)


def test_checkout_root_resolves_a_linked_worktree_whose_dot_git_is_a_file(tmp_path):
    """Property: a checkout whose `.git` is a FILE -- a linked git worktree -- resolves to its
    OWN root, so its tier address is the worktree name.

    Class: NEW CAPABILITY -- the plausible wrong implementation tests `.git` for
    `is_dir()` rather than `exists()`, which is what an author reaching for "is this a
    repository root" writes first. Measured on that form: `checkout_root(deep)` returns
    `None` and the address falls through to the digest branch.

    Why it is required: git records a linked worktree with a `.git` FILE holding a
    `gitdir:` pointer, and every `.claude/worktrees/{name}` tree in this repository --
    including the one this suite runs in -- is one. Under the `is_dir()` form a worktree
    resolves to no root, so its tier stops being named after its worktree and
    `_run_dir_reaper`'s live-worktree protection silently stops matching it, leaving a
    live tier reapable once its TTL passes.

    What kills it: `.git` detection that requires a directory.

    A second correct implementation that still passes: resolving the root with
    `git rev-parse --show-toplevel`, which answers the same root for a linked worktree.
    """
    root = tmp_path / ".claude" / "worktrees" / "wt-linked"
    deep = root / "tests" / "fixtures"
    deep.mkdir(parents=True)
    # A linked worktree's `.git` is a FILE, not a directory. Written as the pointer git
    # actually writes so the tree is representative rather than merely file-shaped.
    (root / ".git").write_text("gitdir: /elsewhere/.git/worktrees/wt-linked\n", encoding="utf-8")

    assert cache_paths.checkout_root(deep) == root.resolve()
    assert cache_paths.tier_address_for(deep) == "wt-linked"


def test_cache_status_creates_no_root(tmp_path, monkeypatch):
    """Property: `hhemt cache status` reports on the cache roots WITHOUT materializing any.

    Class: NEW CAPABILITY -- the plausible wrong implementation creates each root
    (`path.mkdir(parents=True, exist_ok=True)`) before testing its presence, which is the
    shape a status verb takes when its author reaches for "make sure it is there, then
    report on it". Measured: creating one root before the verb runs is detected by the
    same instrument this test uses.

    Why it is required: the verb is what a reader runs to decide whether a machine is
    warm, so a verb that creates the roots answers a question about a cache it just
    manufactured -- and it manufactures that cache on the disk-constrained machines the
    cache exists to relieve.

    What kills it: any `mkdir` on a root path reachable from `cache status`.

    A second correct implementation that still passes: reporting through a summary object
    that computes every path without touching the filesystem, provided nothing is created.
    """
    from typer.testing import CliRunner

    from hhemt.cli import app

    xdg = tmp_path / "xdg"
    xdg.mkdir()
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    root = cache_paths.cache_root()
    if not str(root).startswith(str(xdg)):
        pytest.skip("this platform's cache-dir resolution does not honour XDG_CACHE_HOME")
    assert not root.exists(), "precondition: the cache root is absent before the verb runs"

    result = CliRunner().invoke(app, ["cache", "status"])

    assert result.exit_code == 0, result.output
    assert "tier_address:" in result.output
    assert not root.exists(), f"`cache status` created {root}"


def test_conftest_defines_no_marker_machinery():
    """Property: the fixture tree is a CLIENT of the cache substrate -- `tests/conftest.py`
    DEFINES none of the four names `hhemt.cache.markers` owns, nor the `hhemt.cache.keys`
    composer that moved with them.

    Class: REGRESSION -- red on the pre-change tree, where conftest defined `_marker_state`,
    `_finish_build`, `_warn_unsatisfied` and `_unmet_payload` (measured on
    `git show HEAD:tests/conftest.py`: all four present); those four bodies are what moved
    into `src/hhemt/cache/markers.py`. `_builder_provenance` is the fifth: conftest defined
    it too, and its body moved into `src/hhemt/cache/keys.py`.

    Why it is required: the relocation is the whole point of the substrate -- the toolkit
    could not describe its own cache without importing the test tree -- and a local
    re-definition is how the two copies come back and drift, with a producer writing one
    marker convention and a reader comparing another.

    What kills it: re-adding any of the four as a local function in conftest.

    A second correct implementation that still passes: conftest importing them under
    underscored aliases (`from hhemt.cache import marker_state as _marker_state`). An
    IMPORT is a client; only a DEFINITION is measured here.

    FITNESS FUNCTION, named as one deliberately: this asserts a layer direction
    (`tests/` -> `src/`) rather than a behaviour. It is not an unusually strict unit test.
    """
    import ast

    tree = ast.parse((_REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8"))
    defined = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    owned = {"marker_state", "finish_build", "warn_unsatisfied", "unmet_payload", "builder_provenance"}
    shadowed = sorted(defined & (owned | {"_" + name for name in owned}))

    assert shadowed == [], shadowed
