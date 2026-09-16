"""Coverage for the BLOCKING/ADVISORY verdict partition in aggregate.py.

No cluster, no compile, no solver: every test builds a synthetic run directory on
tmp_path and calls `aggregate()` directly. Runs in milliseconds.

WHY THIS FILE EXISTS. The instrument that would catch a wrong verdict partition IS
the thing being edited, so the failure is self-concealing: a partition one notch too
generous moves a REAL finding into the advisory bucket, `verdict` reads GREEN, and
every consumer downstream reports a green nobody earned. Before this split the
opposite held -- measured 2026-09-04, a run with 4 passing tests and failed=0
unevaluated=0 absent=0 reported NOT-GREEN, on three independent appends that fire on
healthy runs. Both directions are covered below.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hhemt.suite import aggregate as A

SHA = "a" * 40
SOFTWARE = "/cache/hhemt/synthetic_test_runs/main/_software"
TREE = "/cache/hhemt/synthetic_test_runs/main/synth_multi_sim"


#: The write samples MEASURED on run 20260904T074157Z_cd4d8f6d9a91. Every `_software`
#: writer wrote only these; four chunks wrote the identical provision lock. Fixtures are
#: built from this rather than from an author's model of the steady state -- the first
#: version of this file used a single writer and a made-up sample, and it agreed with the
#: spec because both came from the same wrong belief.
LOCK_SAMPLES = [".triton.provision.lock", "triton/.build_tritonswmm_cpu.compile.lock"]


def _entry(label, *, is_dir=True, stamp=False, written=False, samples=None):
    return {
        "label": label,
        "root_shape": "shared",
        "is_dir": is_dir,
        "has_version_stamp": stamp,
        "existed_at_first_open": True,
        "exists_at_end": True,
        "read": True,
        "written": written,
        "write_samples": list(samples if samples is not None else LOCK_SAMPLES) if written else [],
    }


def _build(root: Path, *, chunk_entries, exits=None, junit_failures=()):
    """Write a minimal run directory: N chunks, 2 collected node ids each."""
    n = len(chunk_entries)
    exits = exits or [0] * n
    nodes = [f"tests/test_x.py::test_{i}" for i in range(2 * n)]
    chunks = [
        {
            "chunk_id": c,
            "kind": "heavy",
            "files": ["tests/test_x.py"],
            "node_ids": nodes[2 * c : 2 * c + 2],
            "expected_fixtures": [],
        }
        for c in range(n)
    ]
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "FAKE",
                "source_sha": SHA,
                "collected": nodes,
                "chunk_count": n,
                "cheap_bins": 1,
                "chunks": chunks,
                "shared_tree_exposure": [],
            }
        ),
        encoding="utf-8",
    )
    for c, entries in enumerate(chunk_entries):
        (root / f"chunk-{c:02d}.status.json").write_text(
            json.dumps(
                {
                    "chunk_id": c,
                    "run_id": "FAKE",
                    "pytest_exit": exits[c],
                    "node_count": 2,
                    "source_sha_start": SHA,
                    "source_sha_end": SHA,
                    "dirty_start": False,
                    "dirty_end": False,
                    "resolved_hhemt": {
                        "pytest": "/x/hhemt/__init__.py",
                        "sibling": "/x/hhemt/__init__.py",
                    },
                    "session_fixtures_ok": [],
                    "observed_trees": [],
                    "written_trees": [],
                    "reach_entries": entries,
                    "slurm_state": "COMPLETED",
                }
            ),
            encoding="utf-8",
        )
        cases = []
        for node in nodes[2 * c : 2 * c + 2]:
            name = node.split("::")[1]
            body = "<failure message='boom'/>" if node in junit_failures else ""
            cases.append(
                f'<testcase classname="tests.test_x" file="tests/test_x.py" name="{name}" time="0.1">{body}</testcase>'
            )
        (root / f"chunk-{c:02d}.junit.xml").write_text(
            f'<testsuites><testsuite name="pytest" tests="2">{"".join(cases)}</testsuite></testsuites>',
            encoding="utf-8",
        )
    return nodes


def test_measured_steady_state_is_green(tmp_path):
    """The defect this split fixes, at the shape the cluster actually produces.

    FOUR chunks write `_software` and all four write the identical provision lock. That
    is not a constructed edge case: it is what chunks 2, 3, 4 and 6 did on run
    20260904T074157Z_cd4d8f6d9a91, and chunks 2, 3, 4 and 6 of 20260904T145255Z did the
    same. A writer-COUNT threshold marks this BLOCKING and reproduces the constant
    NOT-GREEN the split exists to remove; the content discriminator does not.
    """
    _build(
        tmp_path,
        chunk_entries=[{SOFTWARE: _entry("_software", written=True)} for _ in range(4)],
    )
    r = A.aggregate(tmp_path, scope="array")
    assert r["verdict"] == "GREEN", r["problems"]
    assert r["counts"]["failed"] == 0
    assert r["advisories"], "the findings must be REPORTED, not deleted"


def test_disjoint_payload_writes_are_advisory(tmp_path):
    """The `_sensitivity_configs` shape, also measured: two writers, DIFFERENT files.

    Chunk 2 wrote three `synth_sensitivity_*.csv`; chunk 4 wrote a fourth. Payload rather
    than locks, so a not-a-lock test alone would call this a race -- and it is not one,
    because no two chunks wrote the same path.
    """
    CFG = "/cache/hhemt/synthetic_test_runs/main/_sensitivity_configs"
    _build(
        tmp_path,
        chunk_entries=[
            {CFG: _entry("_sensitivity_configs", written=True, samples=["a.csv", "b.csv"])},
            {CFG: _entry("_sensitivity_configs", written=True, samples=["c.csv"])},
        ],
    )
    r = A.aggregate(tmp_path, scope="array")
    assert r["verdict"] == "GREEN", r["problems"]
    assert any("shared-by-design-write" in a for a in r["advisories"])


@pytest.mark.parametrize(
    "shape",
    ["zero-advisories", "some-advisories"],
    ids=["at_zero", "at_nonzero"],
)
def test_advisory_count_is_on_the_verdict_line_including_at_zero(tmp_path, shape):
    """Adjacency: the verdict word is unreadable without the advisory count beside it.

    The `at_zero` arm is the one that matters and it is easy to omit by accident: a count
    printed only when non-zero teaches a reader to skim for its absence, and absence is
    indistinguishable from a reporting layer too old to emit it.
    """
    if shape == "zero-advisories":
        # ONE chunk reading a STAMPED tree: nothing is cross-chunk, so neither the
        # under-connection nor the shared-by-design finding can fire, and `dirs` carries a
        # stamped entry so the classifier self-check does not fire either.
        _build(tmp_path, chunk_entries=[{TREE: _entry("synth_multi_sim", stamp=True)}])
    else:
        _build(
            tmp_path,
            chunk_entries=[
                {TREE: _entry("synth_multi_sim", stamp=True)},
                {TREE: _entry("synth_multi_sim", stamp=True)},
            ],
        )
    r = A.aggregate(tmp_path, scope="array")
    if shape == "zero-advisories":
        assert r["advisories"] == []
        assert "advisories=0" in r["verdict_line"]
    else:
        assert r["advisories"]
    assert f"advisories={len(r['advisories'])}" in r["verdict_line"]


def test_same_payload_path_written_twice_is_blocking(tmp_path):
    """The case properties 1 and 2 forbid: two chunks building the same artifact.

    Both chunks write `triton/build_tritonswmm_cpu/compilation.log` -- a payload path, not
    a lock, and the SAME one. That is a concurrent build, it re-opens the measured 403
    ceiling, and it must stop a run. If the collision test is ever weakened to a writer
    count or to a bare not-a-lock test, this fails.
    """
    BUILD_LOG = "triton/build_tritonswmm_cpu/compilation.log"
    _build(
        tmp_path,
        chunk_entries=[
            {SOFTWARE: _entry("_software", written=True, samples=[BUILD_LOG])},
            {SOFTWARE: _entry("_software", written=True, samples=[BUILD_LOG])},
        ],
    )
    r = A.aggregate(tmp_path, scope="array")
    assert r["verdict"] == "NOT-GREEN"
    assert any("concurrent-build" in p for p in r["problems"])


@pytest.mark.parametrize(
    "samples, verdict, kind",
    [
        ([".triton.provision.lock"], "GREEN", "shared-by-design-write"),
        (["triton/build/libtriton.a"], "NOT-GREEN", "concurrent-build"),
        ([], "NOT-GREEN", "concurrent-build"),
    ],
    ids=["lock_is_benign", "artifact_collides", "no_samples_fail_closed"],
)
def test_discriminator_over_the_three_evidence_shapes(tmp_path, samples, verdict, kind):
    """Four chunks, same samples each: benign lock / colliding artifact / NO evidence.

    The third case is a CONTRACT test, not a regression test, and the distinction is worth
    stating so nobody spends an afternoon reproducing it. It is UNREACHABLE from the live
    instrument today, measured two ways: `run_suite.py`'s audit hook appends a sample on
    the first write, so `written=True` implies at least one sample; and the LEGACY reach
    shape keys `written_by` on the bare NAME while the modern shape keys on the ABSOLUTE
    PATH, so a legacy writer never attaches to a SHARED_BY_DESIGN entity (verified: the
    classifier reports the two as separate entries and `shared_by_design_writes` is empty).

    It is kept because the invariant it depends on -- "a recorded write always carries a
    sample" -- is undocumented and lives in a DIFFERENT MODULE from the consumer that
    relies on it. A future hook that records a write without a path would silently turn a
    concurrent build into an advisory, and this is the assertion that would fail instead.
    """
    _build(
        tmp_path,
        chunk_entries=[{SOFTWARE: _entry("_software", written=True, samples=samples)} for _ in range(4)],
    )
    r = A.aggregate(tmp_path, scope="array")
    assert r["verdict"] == verdict, r["problems"]
    haystack = r["problems"] if verdict == "NOT-GREEN" else r["advisories"]
    assert any(kind in x for x in haystack), haystack


def test_a_lock_written_by_every_chunk_is_never_blocking(tmp_path):
    """Same PATH, every chunk -- and still benign, because the path is coordination.

    This is the pair to the test above and the reason both halves of the predicate are
    needed. All four measured `_software` writers wrote the identical provision lock; a
    same-path test alone would call concurrent lock acquisition a concurrent build.
    """
    _build(
        tmp_path,
        chunk_entries=[
            {SOFTWARE: _entry("_software", written=True, samples=[".triton.provision.lock"])} for _ in range(4)
        ],
    )
    r = A.aggregate(tmp_path, scope="array")
    assert r["verdict"] == "GREEN", r["problems"]


def test_a_real_failure_still_blocks(tmp_path):
    """The partition must not have swallowed the thing the verdict is FOR.

    This is the self-concealing direction: a split one notch too generous reports GREEN
    over a failing test, and every consumer downstream inherits it.
    """
    nodes = _build(
        tmp_path,
        chunk_entries=[
            {SOFTWARE: _entry("_software", written=True)},
            {SOFTWARE: _entry("_software", written=False)},
        ],
        junit_failures=("tests/test_x.py::test_0",),
    )
    assert nodes
    r = A.aggregate(tmp_path, scope="array")
    assert r["verdict"] == "NOT-GREEN"
    assert r["counts"]["failed"] == 1


def test_under_connection_is_advisory_and_says_what_is_not_detected(tmp_path):
    """The demotion must be VISIBLE, per the round-5 commitment not to demote silently."""
    _build(
        tmp_path,
        chunk_entries=[
            {TREE: _entry("synth_multi_sim", stamp=True)},
            {TREE: _entry("synth_multi_sim", stamp=True)},
        ],
    )
    r = A.aggregate(tmp_path, scope="array")
    assert r["derivation_under_connected"], "fixture must produce an unpredicted tree"
    assert r["verdict"] == "GREEN"
    text = " ".join(r["advisories"])
    assert "derivation-under-connected" in text
    # Case-insensitive deliberately: the emitted text uppercases CANNOT for emphasis, and
    # pinning the casing would make a later wording tweak fail this test for a reason
    # unrelated to the property it exists to protect.
    assert "cannot fail a run" in text.lower()
    md = A.render_summary_md(r)
    assert "## Advisories" in md
    assert "only detector" in md


@pytest.mark.parametrize("scope", ["array", "union"])
def test_scope_problems_are_never_demoted(tmp_path, scope):
    """A caller misusing the instrument is not the instrument reporting on itself."""
    _build(tmp_path, chunk_entries=[{SOFTWARE: _entry("_software", written=True)}])
    m = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    m["scope_intent"] = "triage"
    (tmp_path / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    r = A.aggregate(tmp_path, scope=scope)
    assert r["scope"] == "triage"
    if scope == "union":
        assert any("triage" in p for p in r["problems"])


def test_complement_reports_declared_and_not_attempted(tmp_path):
    """The two derivations BRACKET the complement; the section must confuse neither.

    EXPECTATIONS ARE AUTHORED LITERALS. A guard that re-derived the declared set from
    STRUCTURAL_SKIP_REASONS would read the same tuple both derivations read and would go
    green on a rename exactly as they do. The fixture is synthetic for the same reason:
    built from the live tests/ corpus the numbers would drift with the repo and the author
    would be pushed straight back to computing them.

    THE DISAGREEING CASE IS THE POINT. `shadowed` declares a structural gate and skips
    under a NON-structural message, so it is declared-but-not-observed-structural and the
    difference set is non-empty by construction. A fixture where the two derivations agree
    is green before this change and after it, which is the both-states failure.
    """
    nodes = _build(tmp_path, chunk_entries=[{SOFTWARE: _entry("_software")}])
    observed, shadowed = nodes[0], nodes[1]

    m = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    m["declared_complement"] = [observed, shadowed]
    (tmp_path / "manifest.json").write_text(json.dumps(m), encoding="utf-8")

    structural_msg = "Local coupled run-proof; do not launch on an HPC scheduler node."
    cases = (
        f'<testcase classname="tests.test_x" file="tests/test_x.py" name="{observed.split("::")[1]}">'
        f'<skipped message="{structural_msg}"/></testcase>'
        f'<testcase classname="tests.test_x" file="tests/test_x.py" name="{shadowed.split("::")[1]}">'
        f'<skipped message="operator-gated live-deposit e2e."/></testcase>'
    )
    (tmp_path / "chunk-00.junit.xml").write_text(
        f'<testsuites><testsuite name="pytest" tests="2">{cases}</testsuite></testsuites>',
        encoding="utf-8",
    )

    r = A.aggregate(tmp_path, scope="array")

    assert r["structurally_excluded"] == [observed], "the OBSERVED partition cell is unchanged"
    # `.get(... ) or []` rather than `[...]`: pre-fix the keys are ABSENT, and a KeyError
    # would make this guard discriminate on the SCHEMA. Anchored on a value comparison, it
    # discriminates on the PROPERTY and reports what the set actually was.
    assert sorted(r.get("declared_complement") or []) == sorted([observed, shadowed])
    assert sorted(r.get("declared_not_attempted") or []) == sorted([observed, shadowed])

    assert any(a.startswith("[complement-derivation]") and shadowed in a for a in r["advisories"])
    assert not any("complement-derivation" in p for p in r["problems"]), "advisory, never blocking"

    md = A.render_summary_md(r)
    # SEPARATE THE SURFACES BEFORE ASSERTING. Every advisory is rendered into `md`, so a bare
    # `shadowed in md` is ENTAILED by the advisory assertion above: it would pass unchanged if
    # the runnable block were reverted to `structurally_excluded`, which is the exact
    # regression it would claim to pin. Extract the fenced block under the copy-verbatim line
    # and assert over THAT, then pin that the two surfaces really are separate.
    block = md.split("Copy the block verbatim:")[1].split("```")[1]
    assert "[complement-derivation]" not in block, "the advisory must not bleed into the block"
    assert observed in block and shadowed in block, block
    assert "this scope cannot evaluate" not in md, "regression pin on one known-false phrase"


def _one_chunk_junit(tmp_path, cases: str) -> None:
    (tmp_path / "chunk-00.junit.xml").write_text(
        f'<testsuites><testsuite name="pytest" tests="2">{cases}</testsuite></testsuites>',
        encoding="utf-8",
    )


def test_failed_survives_a_skip_in_the_same_junit(tmp_path):
    """A junit carrying BOTH a failure and a skip must still report the failure.

    GUARD, NOT DIFFERENTIAL -- and that is the point. The skip-reason carrier was chosen
    as a SEPARATE mapping precisely because widening `outcomes`' value type to carry the
    reason defeats four exact-match consumers without raising: `_of("FAILED")` compares
    `o == kind`, a tuple equals no literal, and the run reports zero failures. This test
    is green before the carrier lands and green after it; it REDDENS under the rejected
    shape (measured: a `(class, reason)` value makes `counts['failed']` read 0 here when
    every `parse_junit` assignment carries the tuple).
    It pins that the reason-carrying change did not touch the list the verdict is FOR.
    """
    nodes = _build(tmp_path, chunk_entries=[{SOFTWARE: _entry("_software")}])
    failed_node, skipped_node = nodes[0], nodes[1]
    _one_chunk_junit(
        tmp_path,
        f'<testcase classname="tests.test_x" file="tests/test_x.py" name="{failed_node.split("::")[1]}">'
        f"<failure message='boom'/></testcase>"
        f'<testcase classname="tests.test_x" file="tests/test_x.py" name="{skipped_node.split("::")[1]}">'
        f'<skipped message="node is required to parse JS"/></testcase>',
    )
    r = A.aggregate(tmp_path, scope="array")
    assert r["counts"]["failed"] == 1
    assert r["failed"] == [failed_node]
    assert r["incidental_skips"] == [skipped_node]
    assert r["verdict"] == "NOT-GREEN"


def test_incidental_skip_reason_reaches_the_summary(tmp_path):
    """The junit skip message is carried per node id and rendered beside it.

    Pre-change the message survives `parse_junit`'s parse and is discarded one line
    later, so the summary lists a bare node id under a header asserting `benign`. The
    fixture-shape reason used here is one a reader can classify ON SIGHT, which is the
    whole reason to surface it. Assertions are on the VALUE and the rendered ROW, never
    on the schema: `.get()` on the new key makes a pre-change run fail on the property
    (None != msg) rather than on a KeyError.
    """
    nodes = _build(tmp_path, chunk_entries=[{SOFTWARE: _entry("_software")}])
    node = nodes[1]
    msg = "sensitivity_datatree.zarr not present in fixture"
    _one_chunk_junit(
        tmp_path,
        f'<testcase classname="tests.test_x" file="tests/test_x.py" name="{nodes[0].split("::")[1]}"></testcase>'
        f'<testcase classname="tests.test_x" file="tests/test_x.py" name="{node.split("::")[1]}">'
        f'<skipped message="{msg}"/></testcase>',
    )
    r = A.aggregate(tmp_path, scope="array")
    assert r["incidental_skips"] == [node], "the incidental list itself is unchanged"
    assert (r.get("incidental_skip_reasons") or {}).get(node) == msg

    md = A.render_summary_md(r)
    header = next(line for line in md.splitlines() if line.startswith("## Incidental skips"))
    assert "benign" not in header, header
    assert "READING RULE" in md
    block = md.split("## Incidental skips")[1].split("```text")[1].split("```")[0]
    assert f"{node} — {msg}" in block, block
