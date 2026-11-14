"""The tier-1 identity: the terms that name a cached tree, bottomed from both sides.

``hhemt.cache.keys`` composes the identity a fixture completion marker records and compares.
This module constrains EVERY term of that composition, and it grew in two halves: arms 1-3
and their guard constrain ``toolkit_src_digest`` -- the toolkit's own SOURCE BYTES, read from
the working tree -- and arms 4-14 constrain the other eight terms, the tier-0 composition
they reach through, and the payload arm the marker state machine separates from them.

The arms are named together because they bound the composition from OPPOSITE sides, and a
reader who saw only one of them could not tell which bound had been chosen. Arm 1 is an
under-sensitivity floor: a change outside ``src/hhemt/`` must not move the identity. Arm 2 is
the over-sensitivity the key ACCEPTS by decision: a comment-only edit inside ``src/`` does
move it. Arm 3 is the one that separates a working-tree read from a ``git show HEAD:`` read,
and it is the arm that matters most -- the wrong reading satisfies every other arm in this
file, so a green on arms 1 and 2 is no evidence about it at all.

From arm 4 on, each arm STATES WHETHER IT COMPARES A KEY OR OBSERVES A MARKER, because the
two are different instruments: a key comparison says two identities differ, and a marker
observation says the state machine acts on that difference.

One guard bounds the source term's DOMAIN rather than either sensitivity:
``test_the_digest_ranges_into_SUBDIRECTORIES_of_src_hhemt``. It exists because all three
graded arms touch a TOP-LEVEL module, so a digest that walked only ``src/hhemt/*.py`` would
pass every one of them while covering a small fraction of the package.

Every arm builds its own throwaway checkout, fake tier-0 root or payload under ``tmp_path``.
None of them touches the real repository or the real caches, so a test run cannot invalidate
a live fixture marker as a side effect.
"""

from __future__ import annotations

import hashlib
import inspect
import re
import subprocess
from pathlib import Path

from hhemt.cache import keys as cache_keys
from hhemt.cache import markers as cache_markers

#: The composition's key set, named once because several arms compare against it and the rest
#: would otherwise restate it. It is a SET for assertion purposes: the shape fixes WHICH terms
#: compose the identity, never the order a dict literal happens to be built in.
_IDENTITY_KEYS = tuple(cache_keys.TIER1_TERMS)

#: This repository's root, for the arms that compose a provenance block without a checkout.
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(repo: Path, *args: str) -> str:
    """Run ``git`` in ``repo`` with a pinned identity and return stdout.

    The identity flags are per-invocation rather than configured into the fixture, so the
    throwaway repository never depends on the host's git configuration and never writes a
    ``~/.gitconfig``.
    """
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _checkout(base: Path) -> Path:
    """A throwaway checkout root with both identity inputs and one committed seed.

    ``src/hhemt/mod.py`` is the toolkit source the second term ranges over;
    ``tests/conftest_like.py`` stands in for the builder sources the FIRST term ranges over,
    which the caller supplies explicitly because they live in the test tree.
    """
    repo = base / "checkout"
    (repo / "src" / "hhemt").mkdir(parents=True)
    (repo / "src" / "hhemt" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "conftest_like.py").write_text("BUILDER = 1\n", encoding="utf-8")
    (repo / "docs").mkdir()
    (repo / "docs" / "notes.md").write_text("first\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "seed")
    return repo


def _builder_sources(repo: Path) -> tuple[Path, ...]:
    """The first term's input, held constant across every arm so it cannot explain a delta."""
    return (repo / "tests" / "conftest_like.py",)


def test_arm_1_a_change_outside_src_hhemt_leaves_the_identity_unchanged(tmp_path):
    """Property: a COMMIT that touches nothing under ``src/hhemt/`` -- a docs page, a
    ``library/`` note, a ``scripts/`` module -- leaves the identity block byte-identical.

    Class: REGRESSION -- red on the pre-change tree. There the second term was
    ``git rev-parse HEAD``, so ANY commit moved it, and the block's key set was
    ``{"builder_digest", "toolkit_sha"}`` rather than ``{"builder_digest",
    "toolkit_src_digest"}``. Both differences redden this arm.

    Reachability: the named input class is "a file written and committed outside
    ``src/hhemt/``", reached by ``_git(repo, "commit", ...)`` after the write at this test's
    own call site; the failure it reaches is a moved digest.

    Both-states anchor: the assertion turns on the identity BLOCK's equality, which is a
    property of the current and proposed implementations alike; it does not assert any
    wording this change introduces -- the key set is compared to a tuple this module names
    rather than to a message.

    Invocation isolated: the red is not invocation-dependent. The fixture writes its own
    repository under ``tmp_path`` and the code reads only the ``repo_root`` argument, so no
    path, flag or environment variable of the pytest invocation can green it.

    Why it is required: this is the whole point of replacing a commit identity with a source
    identity. The pre-change term refused a rebuilt tree for a documentation edit, which is
    a full HPC round-trip bought for nothing, and it did so during the very build waves in
    which ``src/`` is being edited for real.

    What kills it: ranging the digest over the repository rather than over ``src/hhemt/``, or
    returning to ``git rev-parse HEAD``.

    A second correct implementation that still passes: hashing every tracked file whose
    path starts with ``src/hhemt/`` and ends in ``.py`` yields the same invariances; this arm
    reads only whether the block moved, never how the digest was framed.
    """
    from hhemt.cache.keys import builder_provenance

    repo = _checkout(tmp_path)
    sources = _builder_sources(repo)
    before = builder_provenance(sources, repo_root=repo)

    # A SET, not a positional tuple. The property is WHICH terms compose the identity; a
    # correct implementation that builds the same keys in the other insertion order must
    # pass, and under a tuple comparison it fails. Measured both ways in the round file.
    assert set(before) == set(_IDENTITY_KEYS), before

    (repo / "docs" / "notes.md").write_text("second\n", encoding="utf-8")
    (repo / "library").mkdir()
    (repo / "library" / "note.md").write_text("x\n", encoding="utf-8")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "runner.py").write_text("print('hi')\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "docs, library and scripts only")

    after = builder_provenance(sources, repo_root=repo)

    assert after == before, (
        "a commit touching nothing under src/hhemt/ moved the cache identity; the tier-1 "
        "marker will read stale after every documentation edit"
    )
    # The scripts/ module is a `.py` file on purpose: it distinguishes "the glob is wrong"
    # from "the glob's ROOT is wrong", and only the latter is this arm's property.
    assert after["toolkit_src_digest"] != "", "the digest collapsed to its absent-source value"


def test_arm_2_a_comment_only_src_edit_changes_the_identity(tmp_path):
    """Property: adding a COMMENT to a file under ``src/hhemt/`` changes the identity, and
    it does so even when the executable body is untouched.

    Class: NEW CAPABILITY -- the plausible wrong implementation is a digest over parsed and
    re-unparsed syntax trees, or over ``ast.dump`` output, which normalizes comments away.
    Under it the two trees here are identical and this arm is RED.

    Reachability: the input class is "an edit whose only difference is comment text",
    reached by writing the same statement under an added ``#`` line; the failure it reaches
    is an unchanged digest.

    Both-states anchor: the assertion turns on comment text being visible to the digest,
    which is true of the proposed implementation and false of the AST-normalizing
    alternative -- it does not turn on any name or message this change introduces.

    Invocation isolated: the fixture owns its repository and the code reads only
    ``repo_root``, so no aspect of how pytest was invoked can produce or green this red.

    Why it is required: this over-invalidation is a DECISION, not a limitation, and an
    undocumented decision is the thing a later "cleanup" removes. Splitting a body from a
    comment needs an AST normalization that is itself a drift surface, and the failure it
    would introduce -- a real body edit the key cannot see -- costs a wrong result that reads
    as a correct one.

    What kills it: normalizing comments out of the digest.

    A second correct implementation that still passes: stripping whitespace-normalized blank
    lines while still hashing comment text yields the same result; the arm asserts only that
    a comment-only delta moves the identity.
    """
    from hhemt.cache.keys import builder_provenance

    repo = _checkout(tmp_path)
    sources = _builder_sources(repo)
    before = builder_provenance(sources, repo_root=repo)

    module = repo / "src" / "hhemt" / "mod.py"
    module.write_text("# a later commit documents this line\nx = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "comment only")

    after = builder_provenance(sources, repo_root=repo)

    assert after["toolkit_src_digest"] != before["toolkit_src_digest"], (
        "a comment-only edit under src/hhemt/ left the identity unchanged; the digest is "
        "normalizing comments away and a later refactor will normalize a body with them"
    )


def test_arm_3_an_uncommitted_src_edit_changes_the_identity(tmp_path):
    """Property: an edit to ``src/hhemt/`` that is neither committed nor staged changes the
    identity, while ``HEAD`` still holds the pre-edit bytes.

    Class: NEW CAPABILITY -- the plausible wrong implementation is
    ``git show HEAD:{path}`` (or ``git cat-file`` on the ``HEAD`` tree) per file, which
    satisfies every other arm in this file and returns the COMMITTED bytes here. Under it
    ``after`` equals ``before`` and this arm is RED.

    Reachability: the input class is "an uncommitted working-tree edit under
    ``src/hhemt/``"; the test reaches it by writing the file and deliberately not staging or
    committing, and the failure it reaches is an unchanged digest.

    Both-states anchor: the final assertion reads ``HEAD``'s own bytes and requires them to
    be the PRE-edit content, so the test states the fact that makes it discriminating -- if
    the edit were somehow committed, that assertion fails rather than the arm passing
    vacuously.

    Invocation isolated: the fixture owns its repository; ``repo_root`` is the only input the
    code reads, so no pytest path, flag or environment variable can green this red.

    Why it is required: this is the defect the term exists to remove. A builder editing
    ``src/`` mid-run left every marker reading current, so a tree built by different code was
    handed out under a current marker -- a wrong result that reads as a real one. A
    ``git show HEAD:`` implementation restores exactly that behaviour one level up while
    passing every description of "a digest of the toolkit sources".

    What kills it: reading the digest from a git object instead of the working tree, or
    caching the digest keyed on ``HEAD``.

    A second correct implementation that still passes: walking the tree and hashing each
    file's ``stat().st_mtime_ns`` alongside its size and path is a different framing of the
    same working-tree read; this arm asserts only that the identity moved while ``HEAD`` did
    not.
    """
    from hhemt.cache.keys import builder_provenance

    repo = _checkout(tmp_path)
    sources = _builder_sources(repo)
    before = builder_provenance(sources, repo_root=repo)

    module = repo / "src" / "hhemt" / "mod.py"
    module.write_text("x = 2\n", encoding="utf-8")

    after = builder_provenance(sources, repo_root=repo)

    assert after["toolkit_src_digest"] != before["toolkit_src_digest"], (
        "an UNCOMMITTED edit under src/hhemt/ left the identity unchanged; the digest is "
        "being read from HEAD, so a tree built by edited code is handed out as current"
    )
    assert _git(repo, "show", "HEAD:src/hhemt/mod.py") == "x = 1\n", (
        "the fixture no longer discriminates: the HEAD blob is not the pre-edit content, so "
        "a HEAD-reading implementation would have been green for the wrong reason"
    )


def test_the_digest_ranges_into_SUBDIRECTORIES_of_src_hhemt(tmp_path):
    """Property: a ``.py`` file NESTED in a subdirectory of ``src/hhemt/`` contributes to the
    identity -- a COMMITTED edit to it, with every top-level module byte-identical, moves the
    digest.

    NO ORDINAL, deliberately. The shape enumerates arms 1-14 in this module across two work
    packages, and P4's arm 4 is a T0-generator edit. This guard is an ADDITION defending the
    domain clause of the same term rather than a renumbering, so naming it ``arm_4`` would
    collide with P4's arm by title and make two different properties unaddressable by the
    node id the DoD cites.

    Class: NEW CAPABILITY -- the plausible wrong implementation is a top-level-only walk,
    ``src_root.glob("*.py")`` in place of ``src_root.rglob("*.py")``, which is a one-word slip
    and which passes every other arm in this file. Red under it, output pasted in the round
    file. It is deliberately a COMMITTED edit rather than an uncommitted one: committed keeps
    this arm blind to the ``git show HEAD:`` implementation arm 3 already constrains, so each
    arm isolates one property instead of two.

    Reachability: the input class is "a committed edit to a ``.py`` inside a subdirectory of
    ``src/hhemt/``"; this test reaches it by creating ``src/hhemt/pkg/deep.py``, committing
    it, then rewriting and committing it again. The failure it reaches is an unchanged
    digest.

    Both-states anchor: the assertion turns on whether a NESTED module's bytes enter the
    digest, which is true of every implementation satisfying the shape's "every ``*.py``
    under ``src/hhemt/``" and false of a top-level-only walk. It does not turn on any name or
    message this change introduces.

    Invocation isolated: the fixture owns its repository under ``tmp_path`` and the code reads
    only its ``repo_root`` argument, so no pytest path, flag or environment variable can
    green this.

    Why it is required: the shape's term is "every ``*.py`` under ``src/hhemt/``", and the
    package's own tree is dominated by subpackages (``cache``, ``config``, ``suite``, ``eda``,
    ``report_renderers``, ``version_migration``). A digest that walked only the top level would
    cover a small fraction of the sources while every arm 1-3 stayed green, so the term's
    protection would be silently narrower than the sentence that justifies it.

    What kills it: replacing ``rglob`` with ``glob``, or listing top-level modules by name.

    A second correct implementation that still passes: hashing every git-tracked path that
    starts with ``src/hhemt/`` and ends in ``.py``, recursively, yields the same
    discrimination; this arm reads only whether the identity moved.
    """
    from hhemt.cache.keys import builder_provenance

    repo = _checkout(tmp_path)
    sources = _builder_sources(repo)
    top_level = repo / "src" / "hhemt" / "mod.py"

    nested = repo / "src" / "hhemt" / "pkg" / "deep.py"
    nested.parent.mkdir()
    nested.write_text("n = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "nested module")

    before = builder_provenance(sources, repo_root=repo)
    top_level_before = top_level.read_bytes()

    nested.write_text("n = 2\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "edit the nested module only")

    after = builder_provenance(sources, repo_root=repo)

    assert top_level.read_bytes() == top_level_before, (
        "the fixture no longer discriminates: the top-level module changed too, so a moved "
        "digest would not be attributable to the nested file"
    )
    assert after["toolkit_src_digest"] != before["toolkit_src_digest"], (
        "an edit to a .py NESTED under src/hhemt/ left the identity unchanged; the digest "
        "walks only the top level, so most of the toolkit's sources are outside the term the "
        "shape defines as 'every *.py under src/hhemt/'"
    )


# ======================================================================================
# Arms 4-14: the other eight terms, the tier-0 composition they reach, and the payload arm
# ======================================================================================


def _sha256_file(path: Path) -> str:
    """SHA-256 of one file's bytes, for the arms that compare raster artifacts."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _marker(analysis_dir: Path, *, tier0_key: str = "", payload=()) -> dict:
    """Write a completion marker whose provenance carries ``tier0_key``, and return it."""
    provenance = cache_keys.builder_provenance((), repo_root=_REPO_ROOT, tier0_key=tier0_key)
    cache_markers.finish_build(analysis_dir, payload, "cache_key_identity_arm", provenance=provenance)
    return provenance


def test_arm_4_a_tier0_generator_edit_flips_the_marker_to_stale(tmp_path, monkeypatch):
    """Property: a change to the SYNTHETIC GENERATOR's source changes tier 0's key, and the
    tier-1 marker written against the old key reads ``stale``.

    Class: NEW CAPABILITY -- the plausible wrong implementation is a tier-1 composition that
    re-derives tier 0's terms instead of composing tier 0's key, or one that omits
    ``tier0_key`` because the artifacts are "synthetic anyway". Under either the marker stays
    ``complete`` across a generator edit and the tree is handed out against artifacts it was
    not built from. Red under them; output pasted in the round file.

    Reachability: the input class is "a generator-source edit", reached by rebinding the
    module's ``_GENERATOR_SOURCE_HASH`` -- the digest of the generator's own bytes -- and
    recomputing the tier-0 key through the real ``_cache_key``.

    Both-states anchor: the assertion turns on the marker ARM the state machine returns, a
    property of the current implementation and of any implementation that composes tier 0's
    key; it asserts no wording this change introduces.

    Invocation isolated: the arm builds its own analysis directory and its own tier-0 key
    string under ``tmp_path``; nothing about how pytest was invoked can green it.

    Why it is required: this is the one term that connects the two tiers. Without it a
    generator edit invalidates tier 0 and leaves every analysis tree reading current.

    What kills it: dropping ``tier0_key`` from the composition, or deriving it here from
    something other than tier 0's own key.

    A second correct implementation that still passes: stamping the tier-0 key's PREIMAGE
    (the params, version and generator hash) rather than its digest moves on the same input;
    this arm reads only the state the marker reports.

    COMPARES A KEY and OBSERVES A MARKER: the key inequality is asserted, then the marker arm.
    """
    from tests.fixtures.synthetic_model import cache as synth_cache

    params = synth_cache.SyntheticModelParams()
    key_before = synth_cache._cache_key(params)
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()

    provenance_before = _marker(analysis_dir, tier0_key=key_before)
    assert cache_markers.marker_state(analysis_dir, (), provenance=provenance_before) == "complete"

    monkeypatch.setattr(synth_cache, "_GENERATOR_SOURCE_HASH", "an edited generator's own bytes")
    key_after = synth_cache._cache_key(params)
    assert key_after != key_before, "the tier-0 key did not move on a generator-source edit"

    provenance_after = cache_keys.builder_provenance((), repo_root=_REPO_ROOT, tier0_key=key_after)
    assert cache_markers.marker_state(analysis_dir, (), provenance=provenance_after) == "stale"


def test_arm_5_an_edit_to_the_solver_declaration_flips_the_identity(tmp_path):
    """Property: an edit to the module that DECLARES the TRITON pin and remote moves the
    ``solver_identity`` term, and therefore the composed identity.

    Class: NEW CAPABILITY -- the plausible wrong implementation captures only the pin VALUE,
    which is a constant and cannot move on an edit that changes the module's other contents;
    under it two different solver declarations share one identity. Red under it.

    Reachability: the input class is "an edit to ``tests/fixtures/_triton_source_cache.py``",
    reached by copying the real module and appending a line to the copy.

    Both-states anchor: the assertion turns on whether the TERM's digest moves with the file's
    bytes -- true of the current implementation and of any byte-level reading, false of a
    constant capture -- and not on a name or message this change introduces.

    Invocation isolated: the copy lives under ``tmp_path`` and the helper reads only its path.

    Why it is required: the pin/remote declaration decides which solver a tree was built
    against, so a tree built under a different declaration is a tree built by different code.

    What kills it: hashing a constant, or hashing the CONFIG value the caller resolved (which
    is ``effective_pin``'s subject and cannot see a default-only edit).

    A second correct implementation that still passes: hashing the module's AST rather than
    its bytes moves on any semantic edit; this arm reads only that the term moved.

    COMPARES A KEY.
    """
    import tests.fixtures.test_case_builder as tcb

    source = Path(tcb.__file__).resolve().parent / "_triton_source_cache.py"
    copy = tmp_path / "_triton_source_cache.py"
    copy.write_bytes(source.read_bytes())

    before = cache_keys.file_identity(copy)
    copy.write_text(copy.read_text(encoding="utf-8") + "\n# an edited declaration\n", encoding="utf-8")
    after = cache_keys.file_identity(copy)

    assert after != before, "the solver-declaration term did not move on an edit to the module"
    assert cache_keys.builder_provenance(
        (), repo_root=_REPO_ROOT, solver_identity=after
    ) != cache_keys.builder_provenance((), repo_root=_REPO_ROOT, solver_identity=before), (
        "the composed identity did not move with the solver-declaration term"
    )


def test_arm_6_a_report_config_edit_flips_the_identity(tmp_path):
    """Property: an edit to a report config a completed-tree fixture hands to ``run()`` moves
    the ``configs_digest`` term, and therefore the composed identity.

    Class: NEW CAPABILITY -- the plausible wrong implementation digests the config's PARSED
    content, which normalizes key order and comments away; under it two configs that render
    differently share one identity. Red under it.

    Reachability: the input class is "a report-config edit", reached by copying the real
    ``synth_multisim_report_config.yaml`` and rewriting it.

    Both-states anchor: the assertion turns on whether the TERM's digest moves with the file's
    bytes -- true of the current implementation and of any byte-level reading -- and not on a
    name this change introduces.

    Invocation isolated: the copy lives under ``tmp_path`` and the helper reads only its path.

    Why it is required: the report config decides which figures a cached tree produced, so a
    tree built under one config must not be handed out under another.

    What kills it: digesting a parsed representation, or listing the configs by NAME.

    A second correct implementation that still passes: hashing each config's bytes in a
    different order-stable framing yields the same discrimination; this arm reads only that
    the term moved.

    COMPARES A KEY.

    Live population note: the multisim fixture's ``report_config`` path resolves to a
    nonexistent directory today, so it passes ``None`` to ``run()`` and the assembled
    ``configs_digest`` is ``""`` on that fixture. This arm therefore exercises the TERM's
    mechanism against the catalog's real report config rather than asserting on the fixture's
    current value; the path defect is recorded in the round file rather than repaired here,
    because repairing it would change what ``run()`` does on that fixture.
    """
    source = _REPO_ROOT / "configs" / "reports" / "synth_multisim_report_config.yaml"
    assert source.is_file(), f"the arm's input is missing: {source}"
    copy = tmp_path / "report_config.yaml"
    copy.write_bytes(source.read_bytes())

    before = cache_keys.builder_digest((copy,))
    copy.write_text(copy.read_text(encoding="utf-8") + "\n# an edited report config\n", encoding="utf-8")
    after = cache_keys.builder_digest((copy,))

    assert after != before, "the report-config term did not move on an edit to the config"
    assert cache_keys.builder_provenance(
        (), repo_root=_REPO_ROOT, configs_digest=after
    ) != cache_keys.builder_provenance((), repo_root=_REPO_ROOT, configs_digest=before), (
        "the composed identity did not move with the report-config term"
    )


def test_arm_7_every_case_parameter_flips_the_identity():
    """Property: varying ANY member of the case-parameter surface -- ``n_events``,
    ``n_reporting_tsteps_per_sim``, ``TRITON_reporting_timestep_s``, ``params`` or
    ``additional_system_configs`` -- moves ``case_params_digest``.

    Class: NEW CAPABILITY -- the plausible wrong implementation digests a subset (typically
    only ``params``) and leaves the event count or the system overlay out; under it a tree
    built for five events is handed out for six. Red under it for each omitted member.

    Reachability: the input class is "one case parameter changed", reached by calling the real
    digest function once per member with that member alone varied.

    Both-states anchor: the assertion turns on the DIGEST moving with a value it ranges over,
    which is true of any complete composition and false of any incomplete one; it does not
    turn on a name this change introduces.

    Invocation isolated: the function is pure and takes no path, flag or environment input.

    Why it is required: these are the parameters a tree is BUILT FOR. A cache hit on a
    different build request is a wrong result that reads as a correct one.

    What kills it: leaving any of the five members out of the digest.

    A second correct implementation that still passes: a tuple-keyed digest over the same five
    values moves identically; this arm reads only that the digest moved.

    COMPARES A KEY.
    """
    base = {
        "n_events": 5,
        "n_reporting_tsteps_per_sim": 12,
        "triton_reporting_timestep_s": 10.0,
        "params": {"cell_size_m": 10.0, "n_rows": 30},
        "additional_system_configs": {"TRITONSWMM_branch_key": "aaa"},
    }
    variants = {
        "n_events": 6,
        "n_reporting_tsteps_per_sim": 13,
        "triton_reporting_timestep_s": 20.0,
        "params": {"cell_size_m": 10.0, "n_rows": 31},
        "additional_system_configs": {"TRITONSWMM_branch_key": "bbb"},
    }
    baseline = cache_keys.case_params_digest(**base)
    for member, value in variants.items():
        moved = cache_keys.case_params_digest(**{**base, member: value})
        assert moved != baseline, f"case_params_digest did not move on a change to {member}"


def test_arm_8_an_effective_pin_change_flips_the_identity():
    """Property: with an otherwise identical tree, a change to the pin or software directory
    a CALLER provisioned moves ``effective_pin``, and therefore the composed identity.

    Class: NEW CAPABILITY -- the plausible wrong implementation stamps only the module DEFAULT
    pin, which is identical for two callers provisioning different solvers; under it a tree
    built against one solver is handed out for another. Red under it.

    Reachability: the input class is "a caller-level pin override", reached by varying the two
    values the override consists of and leaving every other term alone.

    Both-states anchor: the assertion turns on the DIGEST of the resolved pair moving, true of
    any implementation that reads the caller's values and false of one that reads the module
    constant; it does not turn on a name this change introduces.

    Invocation isolated: the function is pure and takes no path, flag or environment input.

    Why it is required: two callers at one HEAD can provision DIFFERENT solvers, and a cache
    that cannot tell them apart serves a tree built by code that is not the code asked for.

    What kills it: reading the module default rather than the caller's overlay.

    A second correct implementation that still passes: hashing the two values as a tuple rather
    than as a JSON list moves identically; this arm reads only that the digest moved.

    COMPARES A KEY.
    """
    baseline = cache_keys.effective_pin_digest("aaaaaaaa", None)
    assert cache_keys.effective_pin_digest("bbbbbbbb", None) != baseline, "a pin override did not move it"
    assert cache_keys.effective_pin_digest("aaaaaaaa", "/some/software/dir") != baseline, (
        "a software-directory override did not move it"
    )
    # `None` and "" are DIFFERENT identities: the second is a caller declaring an empty
    # override, which is a caller bug that must not collapse into "no override".
    assert cache_keys.effective_pin_digest("aaaaaaaa", "") != baseline


def test_arm_9_a_truncated_payload_entry_reads_unsatisfied(tmp_path):
    """Property: a payload entry that still EXISTS but no longer matches the ``(size, sha256)``
    its marker recorded reads ``unsatisfied`` -- not ``complete``, and not ``stale``.

    Class: NEW CAPABILITY -- the plausible wrong implementation checks only that the entry
    exists, so a truncated store reads as intact; and the other plausible one folds the
    payload digest into the identity comparison, which reports ``stale`` and sends a reader
    looking for a builder change. Red under either.

    Reachability: the input class is "a payload entry damaged after the marker was written",
    reached by writing a marker over a real file and then truncating that file.

    Both-states anchor: the assertion turns on the ARM the state machine returns, and the arm
    names are the module's own pre-existing vocabulary; it does not turn on wording this
    change introduces.

    Invocation isolated: the payload lives under ``tmp_path`` and the state machine reads only
    the paths it is given.

    Why it is required: a tree damaged after its marker was written is exactly the case the
    marker exists to catch, and an existence-only check cannot see it.

    What kills it: dropping the digest comparison, or merging it into the identity comparison.

    A second correct implementation that still passes: content-hashing every file under the
    payload directory, rather than keying on a shape digest of its (path, size) pairs,
    detects the same truncation for a Zarr store; this arm reads only the arm reported.

    OBSERVES A MARKER.
    """
    analysis_dir = tmp_path / "analysis"
    payload_entry = analysis_dir / "analysis_datatree.zarr" / "chunk0"
    payload_entry.parent.mkdir(parents=True)
    payload_entry.write_text("0123456789", encoding="utf-8")
    payload = ("analysis_datatree.zarr",)

    provenance = _marker(analysis_dir, payload=payload)
    assert (
        cache_markers.marker_state(
            analysis_dir,
            payload,
            provenance=provenance,
            payload_digest=cache_keys.payload_digest(analysis_dir, payload),
        )
        == "complete"
    )

    payload_entry.write_text("012", encoding="utf-8")

    assert (
        cache_markers.marker_state(
            analysis_dir,
            payload,
            provenance=provenance,
            payload_digest=cache_keys.payload_digest(analysis_dir, payload),
        )
        == "unsatisfied"
    ), "a truncated payload entry read as intact; the marker's payload arm is existence-only"


def test_arm_10_a_tier0_change_regenerates_the_derived_raster(tmp_path, monkeypatch):
    """Property: after a tier-0 generator edit, the REBUILT tree's ``elevation_{res}m.dem``
    digest differs from its pre-edit digest AND equals the digest a from-scratch control build
    of the same case produces against the SAME tier-0 key; and it does not equal the SOURCE
    raster's digest.

    Class: NEW CAPABILITY -- the plausible wrong implementation keeps the idempotent skip on
    the cached construction branch, so the derived raster survives the tier-0 change; under it
    the rebuilt digest EQUALS the pre-edit digest and this arm is red. The shape's own words:
    this arm is what separates a repair from a detector.

    Reachability: the input class is "a generator edit whose produced raster changes", reached
    by rebinding the generator-source digest (the edit's identity effect, which re-mints tier
    0) and then rewriting the newly built raster (the edit's byte effect, which a real source
    edit would produce). The arm CONSTRUCTS the case rather than reusing a catalog entry,
    because no catalog entry supplies a raster.

    Both-states anchor: the assertions turn on RASTER DIGESTS and on equality with an
    independent from-scratch build, not on any name or message this change introduces.

    Invocation isolated: this arm constructs its own run roots under ``tmp_path`` and stubs
    exactly two things with reasons stated in the body -- the network clone (no solver is
    needed to coarsen a DEM) and the analysis object (its constructor refuses a dirty
    checkout, which a Build-stage tree always is). It reproduces under any invocation.

    Why it is required: the processed raster is a COARSENED DERIVATIVE of the source raster, so
    its digest cannot equal the source's; only the from-scratch control is a valid referent,
    and only a real regeneration reproduces it.

    What kills it: keeping ``overwrite_outputs_if_already_created=False`` on the cached branch,
    or triggering regeneration on ``stale`` rather than on a tier-0 mismatch.

    A second correct implementation that still passes: unlinking the derived rasters by name on
    a tier-0 mismatch produces the same digest; this arm reads only the raster's bytes.

    COMPARES A KEY AND OBSERVES A MARKER, then compares two artifacts.
    """
    import rasterio

    import tests.fixtures.test_case_builder as tcb
    from tests.fixtures.synthetic_model import cache as synth_cache

    class _StubAnalysis:
        """The analysis object, reduced to the two attributes this builder consumes.

        A synthetic builder touches its analysis for ``cfg_analysis.toggle_sensitivity_analysis``
        and ``analysis_paths.analysis_dir`` only. The real constructor refuses a checkout with
        modified tracked files, and every Build-stage tree has them by construction, so using
        the real object would make this arm unrunnable exactly where it is needed.
        """

        def __init__(self, *_args, **_kwargs):
            self.cfg_analysis = type("_Cfg", (), {"toggle_sensitivity_analysis": False})()
            self.analysis_paths = type("_Paths", (), {"analysis_dir": Path(".")})()

    monkeypatch.setattr(tcb, "provision_borrower", lambda *_a, **_k: None)
    monkeypatch.setattr(tcb, "TRITONSWMM_analysis", _StubAnalysis)

    params = synth_cache.SyntheticModelParams()
    artifacts_before = synth_cache.get_or_build_synthetic_case(params)
    tier0_before = artifacts_before.cache_dir.name

    run_root = tmp_path / "runs"

    def _build(name: str, root: Path, *, start_from_scratch: bool):
        return tcb.retrieve_synth_TRITON_SWMM_test_case(
            analysis_name=name,
            params=params,
            runs_root_override=root,
            start_from_scratch=start_from_scratch,
        )

    case_before = _build("arm10", run_root, start_from_scratch=True)
    analysis_dir = case_before.system_directory / "arm10"
    dem_before = case_before.system.sys_paths.dem_processed
    digest_before = _sha256_file(dem_before)

    # The generator edit's IDENTITY effect: tier 0 re-mints under a new key.
    monkeypatch.setattr(synth_cache, "_GENERATOR_SOURCE_HASH", "an edited generator's own bytes")
    artifacts_after = synth_cache.get_or_build_synthetic_case(params)
    tier0_after = artifacts_after.cache_dir.name
    assert tier0_after != tier0_before, "the arm no longer discriminates: tier 0 did not re-mint"

    # The generator edit's BYTE effect: the same params now produce a different raster.
    with rasterio.open(artifacts_after.dem) as src:
        profile = src.profile
        shifted = src.read() + 5.0
    with rasterio.open(artifacts_after.dem, "w", **profile) as dst:
        dst.write(shifted)

    # A marker attesting the tree was built from tier 0 BEFORE the edit.
    cache_markers.finish_build(
        analysis_dir,
        (),
        "arm_10",
        provenance=cache_keys.builder_provenance((), repo_root=_REPO_ROOT, tier0_key=tier0_before),
    )
    assert cache_markers.recorded_tier0_key(analysis_dir) == tier0_before

    case_after = _build("arm10", run_root, start_from_scratch=False)
    digest_after = _sha256_file(case_after.system.sys_paths.dem_processed)

    case_control = _build("arm10_control", tmp_path / "control_runs", start_from_scratch=True)
    digest_control = _sha256_file(case_control.system.sys_paths.dem_processed)

    assert digest_after != digest_before, (
        "the derived raster survived a tier-0 change; the cached construction branch is still "
        "taking its idempotent skip and the tree now holds rasters from artifacts it no longer "
        "matches"
    )
    assert digest_after == digest_control, (
        "the rebuilt raster differs from a from-scratch build against the same tier-0 key, so "
        "the regeneration is not reproducing the from-scratch result"
    )
    assert digest_after != _sha256_file(artifacts_after.dem), (
        "the processed raster equals the SOURCE raster's digest; the control compared against "
        "the wrong referent -- the processed raster is a COARSENED DERIVATIVE of the source"
    )


def test_arm_11_a_supplied_raster_moves_tier0_and_a_synthetic_one_does_not(tmp_path, monkeypatch):
    """Property, in BOTH directions: a change to a SUPPLIED EXTERNAL raster moves tier 0's key
    -- and, through ``tier0_key``, the tier-1 identity -- while a change to a raster INSIDE the
    tier-0 root does not move it at all.

    Class: NEW CAPABILITY -- the plausible wrong implementation infers the raster's domain from
    a path test, in which case an in-cache path is silently treated as external (the key
    depends on an artifact it produces, which can never converge) or an external one is
    silently treated as an output (the key ignores a real input). Red under either, in opposite
    directions, which is why both are asserted.

    Reachability: the input class is "a supplied external raster whose bytes change", reached by
    writing a file outside any cache root and re-keying. The SYNTHETIC direction is reached by
    pointing the tier-0 root at ``tmp_path``, building a real synthetic case and rewriting its
    ``dem.tif``.

    Both-states anchor: the assertions turn on key EQUALITY and INEQUALITY, which are properties
    of any implementation of the rule and not of any wording this change introduces.

    Invocation isolated: every root this arm reads is either under ``tmp_path`` or the
    monkeypatched fake tier-0 root; no pytest path, flag or environment variable can green it.

    Why it is required: the no-op is INERT on this tree today and stays inert only while the
    domain predicate holds. Asserting only the external side would let an omission -- a case
    that supplies a raster but passes ``None`` -- hide until it served a stale tree.

    What kills it: keying ``DEM_fullres`` by reading it back out of the cache's own output, or
    dropping the guard so an in-cache path is accepted.

    A second correct implementation that still passes: hashing the raster's metadata plus its
    bytes moves on the same input; this arm reads only whether the key moved.

    COMPARES A KEY, in both directions, plus the guard's refusal.
    """
    from tests.fixtures.synthetic_model import cache as synth_cache

    params = synth_cache.SyntheticModelParams()

    # DIRECTION 1 -- an EXTERNAL raster is a real input.
    synthetic_root = tmp_path / "tier0"
    monkeypatch.setattr(synth_cache, "_cache_root", lambda: synthetic_root)

    baseline = synth_cache._cache_key(params)
    external = tmp_path / "external_source_dem.tif"
    external.write_bytes(b"raster version one\n")
    keyed = synth_cache._cache_key(params, source_raster=external)
    assert keyed != baseline, "a supplied external raster did not move tier 0's key"

    external.write_bytes(b"raster version two\n")
    moved = synth_cache._cache_key(params, source_raster=external)
    assert moved != keyed, "a change to a supplied external raster did not move tier 0's key"
    assert cache_keys.builder_provenance((), repo_root=_REPO_ROOT, tier0_key=moved) != cache_keys.builder_provenance(
        (), repo_root=_REPO_ROOT, tier0_key=keyed
    ), "the tier-1 identity did not move with tier 0's key"

    # DIRECTION 2 -- a raster INSIDE the tier-0 root is an OUTPUT and is not a term.
    artifacts = synth_cache.get_or_build_synthetic_case(params)
    assert synthetic_root in artifacts.dem.resolve().parents, "the fake tier-0 root was not used"
    before = synth_cache._cache_key(params)
    artifacts.dem.write_bytes(artifacts.dem.read_bytes() + b"tampered")
    after = synth_cache._cache_key(params)
    assert after == before, (
        "a change to a raster INSIDE the tier-0 root moved the key; the key is ranging over an artifact it produces"
    )
    try:
        synth_cache._cache_key(params, source_raster=artifacts.dem)
    except ValueError:
        pass
    else:  # pragma: no cover - the guard is the assertion
        raise AssertionError(
            "an in-cache path was accepted as a supplied raster; the domain guard did not refuse a caller error"
        )


def _component_sample(component) -> str:
    """A plausible RAW value for ``component`` at (or finer than) its own granularity."""
    return {
        "major.minor": "3.4.5",
        "major": "4.2.1",
        "compiler": "cc (Some Distro 1.2.3-9ubuntu1~1.4) 11.4.0",
        "enum": "cuda",
    }[component.granularity]


def _component_bump(component) -> str:
    """A raw value differing AT the row's granularity -- i.e. a change the policy must see."""
    return {
        "major.minor": "3.9.5",
        "major": "7.2.1",
        "compiler": "cc (Some Distro 1.2.3-9ubuntu1~1.4) 12.4.0",
        "enum": "hip",
    }[component.granularity]


def _component_patch_variant(component) -> str:
    """A raw value differing only in the PATCH of its version -- a change it must NOT see."""
    return {
        "major.minor": "3.4.99",
        "major": "4.2.99",
        "compiler": "cc (Some Distro 1.2.3-9ubuntu1~1.4) 11.4.99",
        "enum": "cuda",
    }[component.granularity]


def test_arm_12_every_real_component_flips_the_key_its_row_names():
    """Property: changing each component the policy admits -- at or above its own granularity --
    moves the digest of the TIER ITS ROW NAMES, and leaves the other tier where it was.

    Class: NEW CAPABILITY -- the plausible wrong implementation puts every component in one
    digest, so an artifact-stack change reds tier 1 directly (skipping the ``tier0_key``
    composition) and a solver change reds tier 0 (which cannot depend on it). Red under it in
    the tier-isolation half.

    Reachability: the input class is "one admitted component changed", reached by passing an
    override for that component alone into the real ``environment_digest``.

    Both-states anchor: the assertion turns on DIGEST movement across a policy table the module
    declares, and on the other tier's digest being unmoved; neither reads wording this change
    introduces.

    Invocation isolated: the override makes the arm independent of the machine it runs on -- no
    installed version, path or flag can green it.

    Why it is required: the tier attachment is the whole rule. One digest would either
    invalidate tier 0 for a compiler change (which cannot affect a raster) or fail to
    invalidate tier 1 for one (which does affect a simulation).

    What kills it: collapsing the two tiers, or dropping a row from the table.

    A second correct implementation that still passes: hashing per-tier sub-tables rather than
    filtering one table moves identically; this arm reads only the digests.

    COMPARES A KEY.
    """
    real_rows = [c for c in cache_keys.ENVIRONMENT_COMPONENTS if c.tier != "excluded"]
    assert real_rows, "the policy table admits no components; the arm would pass vacuously"
    other_tier = {"solver": "artifact", "artifact": "solver"}

    for component in real_rows:
        moved = cache_keys.environment_digest(component.tier, values={component.name: _component_sample(component)})
        different = cache_keys.environment_digest(component.tier, values={component.name: _component_bump(component)})
        assert different != moved, f"{component.name} did not move the {component.tier} digest"

        neighbour = other_tier[component.tier]
        assert cache_keys.environment_digest(
            neighbour, values={component.name: _component_bump(component)}
        ) == cache_keys.environment_digest(neighbour), (
            f"{component.name} leaked into the {neighbour} digest; its row names one tier"
        )


def test_arm_13_declared_noise_and_patch_digits_do_not_flip_it():
    """Property: changing a PATCH digit of an admitted component, or any declared-noise
    component, moves NO tier's digest.

    Class: NEW CAPABILITY -- the plausible wrong implementation digests raw version strings, so
    a ``pyswmm`` patch bump invalidates every analysis tree; under it this arm is red. This is
    the OVER-SENSITIVITY arm, and it is the pair of arm 12 rather than a restatement: one asserts
    what must move, the other what must not, and either alone leaves one direction unmeasured.

    Reachability: the input class is "a patch-level change to a real component" and "a change to
    a declared-noise component", each reached by an override on one component.

    Both-states anchor: the assertion turns on DIGEST EQUALITY, which is a property of the
    policy table and not of wording this change introduces; the must-fire control is arm 12.

    Invocation isolated: the overrides make the arm independent of the machine.

    Why it is required: an unfiltered environment digest IS the over-sensitivity this key
    exists to remove. The exclusion is a decision on the record, and a decision with no arm is
    a decision a later cleanup deletes.

    What kills it: hashing raw version strings, or admitting a declared-noise row.

    A second correct implementation that still passes: normalizing to major.minor at probe time
    rather than at digest time yields the same equality; this arm reads only the digests.

    COMPARES A KEY.
    """
    for component in cache_keys.ENVIRONMENT_COMPONENTS:
        if component.tier == "excluded":
            for tier in ("solver", "artifact"):
                assert cache_keys.environment_digest(
                    tier, values={component.name: "some very different value"}
                ) == cache_keys.environment_digest(tier), (
                    f"the excluded component {component.name} reached the {tier} digest"
                )
            continue
        if component.granularity == "enum":
            # An enum has no patch axis: its granularity IS its whole value, so the
            # patch-exclusion arm has no instance on this row. Recorded rather than skipped
            # silently, because a missing iteration and an absent obligation look alike.
            continue
        patched = cache_keys.environment_digest(
            component.tier, values={component.name: _component_patch_variant(component)}
        )
        assert patched == cache_keys.environment_digest(
            component.tier, values={component.name: _component_sample(component)}
        ), f"a PATCH-level change to {component.name} moved the {component.tier} digest"


def test_arm_14_an_analysis_overlay_cannot_reach_the_identity():
    """Property: ``additional_analysis_configs`` is not a term of the tier-1 identity, and there
    is no parameter through which it could become one.

    Class: NEW CAPABILITY -- the plausible wrong implementation adds the overlay to the case
    parameters "for completeness", which would invalide every tree whenever a per-call overlay
    is used; that is the ADDRESSING concern the scope-out separates out. Red under it.

    Reachability: the input class is "an overlay declared on the composition surface", reached
    by INSPECTING the two composition functions' parameter names -- the surface an implementer
    would actually extend -- rather than by calling them with a value they do not accept.

    Both-states anchor: the assertion turns on the PRESENCE of a name in a signature, which is
    true today and independent of any wording this change introduces. The must-fire control is
    the positive arm below: the same matcher MUST find ``additional_system_configs``, proving it
    ranges over real parameter names rather than over an empty set.

    Invocation isolated: signatures are static; nothing about the invocation can green it.

    Why it is required: the scope-out is only falsifiable if a violation is detectable. A test
    that merely observed today's behaviour would pass equally on a surface that has no
    mechanism to carry the overlay and on one that carries it under a name nobody checked.

    What kills it: adding an analysis-overlay parameter to either function.

    A second correct implementation that still passes: an allow-list of exact parameter names
    rather than a deny-list regex is stricter and passes; this arm reads only what is present.

    COMPARES A KEY -- by construction: it asserts the absence of a term, which is the only
    instrument available for a scope-out.
    """
    forbidden = re.compile(r"analysis|overlay")
    for function in (cache_keys.builder_provenance, cache_keys.case_params_digest):
        names = [name for name in inspect.signature(function).parameters if forbidden.search(name)]
        assert not names, (
            f"{function.__name__} gained parameter(s) {names}; an analysis overlay is an "
            "ADDRESSING concern and must not enter the tier-1 identity"
        )

    # MUST-FIRE CONTROL: the same matcher finds the SYSTEM overlay, which IS a term. Without
    # this the arm above would pass just as well on a matcher that matched nothing.
    system_overlay_names = [
        name for name in inspect.signature(cache_keys.case_params_digest).parameters if re.search(r"system", name)
    ]
    assert "additional_system_configs" in system_overlay_names, (
        "the matcher no longer finds the SYSTEM overlay, which is a live term; the arm above is measuring an empty set"
    )


def test_a_MINOR_change_to_a_major_granularity_row_does_not_flip_the_digest():
    """Property: a MINOR-level version bump of a component whose policy granularity is
    ``major`` moves NO tier's digest, while a MAJOR-level bump of the same component moves it.

    NO ORDINAL, deliberately, for the reason the subdirectory guard above states: the shape
    enumerates fourteen arms across two work packages, and this one defends the GRANULARITY
    column of the policy table rather than adding a fifteenth behaviour the shape names.
    Calling it ``arm_15`` would let a later reader cite it against a DoD that never asked for
    it.

    Class: NEW CAPABILITY -- the plausible wrong implementation honours a row's TIER but not
    its GRANULARITY, normalizing every row at major.minor. Under it this arm is RED while
    BOTH graded arms stay GREEN: the admitted-component arm varies the MAJOR digit and the
    patch arm varies only the PATCH digit, so a major.minor normalizer satisfies each of them
    and still invalidates every analysis tree on a ``pyswmm``, ``numpy`` or ``pandas`` MINOR
    bump -- exactly the over-sensitivity the ``major`` granularity exists to exclude. Red
    under it; output pasted in the round file.

    Reachability: the input class is "a minor-level bump of a row the policy declares
    ``major``", reached by passing an override for that row alone into the real
    ``environment_digest``.

    Both-states anchor: the assertion turns on DIGEST EQUALITY across a minor bump, which is
    true of any implementation that truncates at the row's own granularity and false of one
    that truncates at major.minor. It asserts no wording this change introduces.

    Invocation isolated: the override makes the arm independent of the machine it runs on, and
    the discrimination is shown rather than assumed -- normalizing the ``major`` rows at
    major.minor REDS this test while leaving the admitted-component and patch arms green.

    Why it is required: ``major`` is a decision on the record for three rows, and the patch arm
    guards only the axis those rows share with EVERY other row. Without this arm the decision
    has no instrument that can separate it from a major.minor table.

    What kills it: normalizing a ``major`` row at major.minor, whether at probe time or at
    digest time.

    A second correct implementation that still passes: truncating at PROBE time to the major
    component and digesting the truncated string yields the same equality; this arm reads only
    the digests.
    """
    major_rows = [c for c in cache_keys.ENVIRONMENT_COMPONENTS if c.granularity == "major"]
    assert major_rows, "no row declares major granularity; this arm would pass vacuously"

    for component in major_rows:
        sample = "4.2.1"
        minor_bump = "4.9.1"
        major_bump = "7.2.1"

        before = cache_keys.environment_digest(component.tier, values={component.name: sample})
        after_minor = cache_keys.environment_digest(component.tier, values={component.name: minor_bump})
        assert after_minor == before, (
            f"a MINOR-level bump of {component.name} moved the {component.tier} digest; its "
            f"policy row declares granularity 'major', so keying it at major.minor "
            f"invalidates every analysis tree on a minor release"
        )
        # MUST-FIRE CONTROL: the same instrument, on the same component, must detect a MAJOR
        # bump. Without it the equality above is also consistent with a digest that reads
        # nothing at all.
        after_major = cache_keys.environment_digest(component.tier, values={component.name: major_bump})
        assert after_major != before, (
            f"a MAJOR-level bump of {component.name} did not move the {component.tier} "
            f"digest; the instrument is not reading this component at all"
        )
