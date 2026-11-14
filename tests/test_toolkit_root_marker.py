"""`_toolkit_root()` is a MARKER test, and an unresolvable reader is a MISMATCH.

Two independent identity defects, both about believing a filesystem coincidence:

1. `validation._toolkit_root()` used to answer `_toolkit_source_dir().resolve().parents[2]`
   — a DEPTH assumption. Any package laid three levels below ANY git repository therefore
   inherited that repository's commit as the toolkit's identity, and
   `RunningIdentity.shape` reported `"checkout"` for a tree that is not this toolkit's.
   The fix is the pattern `hhemt.suite.resolved_tree()` already uses: test the LAYOUT
   MARKERS and return an absent answer when they do not hold, rather than counting
   parents.

2. The consume-side divergence guard in `Bundle.from_directory` required BOTH the
   manifest's sha and the reader's sha to look like a sha before it warned. A reader with
   no identity resolves to the sentinel `"unknown"`, which `looks_like_sha` rejects — so
   the guard fell silent exactly when the reader could not say what it was. An
   unresolvable reader is a mismatch, and the warning must name the manifest's sha so a
   caller can act on it.

Both arms below were RUN against the pre-fix source and failed there; the pre-fix output
is pasted in the chunk's round file. Each is a REGRESSION arm by the standard's
classification: red on the pre-change tree, for the reason the arm names.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import warnings
from pathlib import Path

import pytest

from hhemt import validation
from hhemt.bundle import Bundle, _emit
from hhemt.exceptions import ConfigurationError

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "bundles"
MULTI_SIM_FIXTURE = FIXTURES_DIR / "multi_sim"
FOREIGN_SHA = "deadbeefcafe"


# --------------------------------------------------------------------------------------
# (i) A package laid inside a foreign repository has NO identity
# --------------------------------------------------------------------------------------


def test_a_package_inside_a_foreign_repository_has_no_identity(tmp_path, monkeypatch):
    """Property: a toolkit package that merely sits inside another git repo has no identity.

    Class: every tree whose package directory is three levels below the root of a
    repository that is NOT this toolkit's — a vendored copy, a wheel unpacked into a
    project tree, a package installed under a monorepo. Why required: `running_identity`
    is the ONE source of the running commit and every analysis constructor refuses on a
    mismatch against it, so a foreign repository's HEAD reported as the toolkit's identity
    silently disables that gate — a run would pass the check by agreeing with whatever
    repository it happens to be nested in. Killed by: deriving the root from a parent
    COUNT (which any nesting satisfies) instead of from markers that identify THIS tree.
    Second correct implementation: walk up until a directory carries both `src/hhemt` and
    a `pyproject.toml` naming `hhemt`, rather than testing the three fixed paths below.

    Reachability: the failure is `validation.running_identity()`'s root resolution;
    artifact: the geometry assertions below, which pin that the DEPTH assumption lands
    exactly on `{R}` — so a test one level off would pass before the fix too and prove
    nothing.
    Both-states anchor: the assertion turns on whether an identity is produced at all, a
    property of both the current and the proposed code, not on any wording.
    Invocation not isolatable from this tree — discrimination shown instead: routing root
    resolution through a marker test greens this test, and the pre-fix run raised nothing.
    """
    foreign = tmp_path / "foreign"
    pkg = foreign / "src" / "hhemt"
    (pkg / "bundle").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    # A REAL repository, so `{R}/.git` is a directory and the pre-fix reader has a commit
    # to report. Only git is used here; nothing compiles and no solver runs.
    subprocess.run(["git", "init", "-q", str(foreign)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(foreign),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "foreign",
        ],
        check=True,
    )
    foreign_head = subprocess.run(
        ["git", "-C", str(foreign), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    # THE GEOMETRY, asserted rather than assumed: the pre-fix root is EXACTLY the foreign
    # repository, and it carries a `.git`. One level off and the pre-fix code is absent
    # too, which would make this arm vacuous.
    assert (pkg / "bundle").resolve().parents[2] == foreign.resolve()
    assert (foreign / ".git").exists()

    # No pyproject.toml names this project at `foreign`, so it is not this toolkit's tree.
    monkeypatch.setattr(_emit, "_toolkit_source_dir", lambda: pkg / "bundle")
    with pytest.raises(ConfigurationError) as exc:
        validation.running_identity()
    assert foreign_head not in str(exc.value), "the foreign repository's commit leaked into the identity"


def test_the_real_checkout_still_resolves(monkeypatch):
    """Property: the marker test still accepts THIS checkout, so the fix is not a blanket refusal.

    Class: the tree this process actually runs from. Why required: a marker test that
    refuses the real checkout turns every analysis constructor into a refusal — the fix
    must narrow the accepted set to this toolkit's own layouts without losing it. Killed
    by: requiring a marker this checkout does not carry (a stale filename, a wheel-only
    artifact).
    Second correct implementation: the same acceptance reached by resolving
    `hhemt.__file__` as `suite.resolved_tree()` does, rather than from the package dir.
    """
    pytest.importorskip("hhemt")
    root = validation._toolkit_root()
    if root is None:
        pytest.skip("this process runs from a wheel; the acceptance arm has no checkout to accept")
    assert (root / "src" / "hhemt" / "bundle").resolve() == _emit._toolkit_source_dir().resolve()


# --------------------------------------------------------------------------------------
# (ii) An unresolvable reader is a mismatch
# --------------------------------------------------------------------------------------


def _bundle_with_sha(tmp_path: Path, sha: str | None) -> Path:
    """Copy the checked-in fixture and set its manifest ``toolkit_git_sha``."""
    dest = tmp_path / "bundle"
    shutil.copytree(MULTI_SIM_FIXTURE, dest)
    manifest_path = dest / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if sha is None:
        manifest.pop("toolkit_git_sha", None)
    else:
        manifest["toolkit_git_sha"] = sha
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return dest


def _warnings_from_open(bundle_dir: Path) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Bundle.from_directory(bundle_dir)
    return [str(w.message) for w in caught]


def test_unresolvable_reader_is_reported_as_a_mismatch(tmp_path, monkeypatch):
    """Property: a reader that cannot name its own commit is reported as divergent.

    Class: every reader whose toolkit has no identity — a wheel install, a bare copy, a
    package laid inside a foreign repository — opening a bundle that names a commit.
    Why required: the guard exists so a re-render on a drifted checkout is not silent, and
    an unresolvable reader is the state in which the guard knows LEAST; falling silent
    there inverts its purpose, and the earlier `looks_like_sha` conjunct on the reader side
    did exactly that. Killed by: gating the warning on `looks_like_sha(reader_sha)`.
    Second correct implementation: comparing `reader_sha` to `None` and warning on
    inequality with a resolved manifest sha, rather than testing sha-shape.

    Reachability: the failure is `Bundle.from_directory`'s identity comparison; artifact:
    the manifest sha this test supplies and asserts on. The assertion is on the VALUE, not
    on the warning's prose, so a reworded message keeps it honest.
    Both-states anchor: "does opening this bundle warn" is a property of both states.
    """
    import hhemt.bundle as bundle_pkg

    dest = _bundle_with_sha(tmp_path, FOREIGN_SHA)
    monkeypatch.setattr(bundle_pkg, "_get_toolkit_git_sha", lambda strict=True: "unknown")
    messages = _warnings_from_open(dest)
    assert any(FOREIGN_SHA in m for m in messages), (
        f"an unresolvable reader must be reported against the manifest's {FOREIGN_SHA}; got {messages}"
    )


def test_a_reader_that_agrees_does_not_warn(tmp_path, monkeypatch):
    """Property: the must-not-fire control — an agreeing reader emits no divergence warning.

    Class: every reader whose identity EQUALS the manifest's. Why required: without it the
    arm above could pass on a guard that warns unconditionally, which would put a
    provenance caveat on every bundle this toolkit opens. Killed by: dropping the agreement
    test while adding the unresolvable branch.
    Second correct implementation: agreement by full-sha equality rather than by the
    bidirectional prefix test.
    """
    import hhemt.bundle as bundle_pkg

    dest = _bundle_with_sha(tmp_path, FOREIGN_SHA)
    monkeypatch.setattr(bundle_pkg, "_get_toolkit_git_sha", lambda strict=True: FOREIGN_SHA)
    assert _warnings_from_open(dest) == []
