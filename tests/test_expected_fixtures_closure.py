"""Pins `expected_fixtures` to the fixture CLOSURE rather than a substring scan.

The defect this closes is not "test_partition_split.py is special". It is that a
whole-file substring scan cannot distinguish a fixture NAME appearing in a string
literal from a fixture REQUEST, so any file whose subject matter is fixture names
declares fixtures no node asks for and VOIDs its own chunk.
"""

from __future__ import annotations

from pathlib import Path

from hhemt.suite import partition as P


def _manifest(tmp_path: Path, *, node_ids, closures, files):
    src = tmp_path / "tests"
    src.mkdir(parents=True, exist_ok=True)
    for f in files:
        (tmp_path / f).write_text(files[f], encoding="utf-8")
    return P.build_manifest(
        repo_root=tmp_path,
        node_ids=node_ids,
        source_sha="0" * 40,
        run_id="test",
        closures=closures,
    )


def test_a_literal_mention_is_not_a_request():
    """The discriminating case: the name appears in the file and no node requests it."""
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    body = 'CORPUS = """def test_x(tritonswmm_cpu_compiled): pass"""\ndef test_a(): pass\n'
    m = _manifest(
        tmp,
        node_ids=["tests/t.py::test_a"],
        closures={"tests/t.py::test_a": ["tmp_path"]},
        files={"tests/t.py": body},
    )
    # The scan WOULD find it; the closure does not.
    assert "tritonswmm_cpu_compiled" in P._fixtures_used(tmp / "tests/t.py", P.RECORDED_FIXTURES)
    for c in m["chunks"]:
        assert "tritonswmm_cpu_compiled" not in c["expected_fixtures"], c


def test_a_real_request_is_still_declared():
    """The satisfying arm — a closure-carried fixture must still reach expected_fixtures."""
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    m = _manifest(
        tmp,
        node_ids=["tests/t.py::test_a"],
        closures={"tests/t.py::test_a": ["tritonswmm_cpu_compiled"]},
        files={"tests/t.py": "def test_a(): pass\n"},
    )
    assert any("tritonswmm_cpu_compiled" in c["expected_fixtures"] for c in m["chunks"])


# A SUPPORT LAYER IS MANDATORY for any arm carrying a getfixturevalue file. `file_trees`
# resolves the lookup against `support_symbol_trees(repo_root)`, which records a symbol ONLY
# if its body matches `analysis_name = "..."`. Without a conftest defining the requested name
# `universe` is empty, the literal resolves to nothing, and :211 refuses the run BEFORE the
# enrichment loop -- which is why the first version of the arm below failed against correct
# code. The requirement is identical under a coarse or a precise resolver.
#
# AND IT IS WHAT MAKES THE GATE ARM FALSIFIABLE AT ALL -- it is not scaffolding for the
# dynamic arm alone. Measured: remove this conftest and run the gate arm with the token gate
# DELETED, and it returns [] and PASSES. The heavy name never resolves, so the arm reports
# success from an empty population rather than from a working gate. Deleting this constant
# does not fail a test; it silently makes one of them vacuous.
_SUPPORT = 'def rendered_synth_multi_sim():\n    analysis_name = "synth_multi_sim"\n'


def test_a_dynamic_request_is_still_declared():
    """The closure's blind spot, exercised through build_manifest.

    FAILS ON DELETION: remove the re-admission and expected_fixtures falls back to the
    supplied closure `["request"]` -- which is what pytest actually records for a dynamic
    request -- so the heavy name is absent and this assertion breaks.
    """
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    body = "def test_a(request):\n    analysis = request.getfixturevalue('rendered_synth_multi_sim')\n"
    m = _manifest(
        tmp,
        node_ids=["tests/t.py::test_a"],
        closures={"tests/t.py::test_a": ["request"]},
        files={"tests/t.py": body, "tests/conftest.py": _SUPPORT},
    )
    assert any("rendered_synth_multi_sim" in c["expected_fixtures"] for c in m["chunks"]), m


def test_the_token_gate_is_what_stops_a_corpus_file_declaring():
    """Pins the TOKEN GATE, which nothing else pins.

    The gate and the resolver are separate stages. This file carries a heavy-fixture name in
    a string literal and NO `getfixturevalue`, i.e. the shape of test_partition_split.py --
    whose literals DO resolve, measured. FAILS ON DELETION: remove the
    `_DYNAMIC_LOOKUP_MARKER not in txt` gate and the resolver declares the name, because
    resolution alone cannot tell a corpus literal from a request.
    """
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    body = 'CORPUS = ["rendered_synth_multi_sim"]\ndef test_a(): pass\n'  # no getfixturevalue
    m = _manifest(
        tmp,
        node_ids=["tests/t.py::test_a"],
        closures={"tests/t.py::test_a": ["tmp_path"]},
        files={"tests/t.py": body, "tests/conftest.py": _SUPPORT},
    )
    for c in m["chunks"]:
        assert "rendered_synth_multi_sim" not in c["expected_fixtures"], c


# NO ARM FOR THE PER-NODE GUARD, DELIBERATELY. It is redundant defence-in-depth with no
# reachable input: `partition.py:462`'s wholesale guard fires first on the same condition,
# and its message contains "no fixture closure", so any `pytest.raises(..., match=...)` arm
# would go GREEN against unchanged code and would stay green if the guard were deleted. A
# stated redundancy is a better artifact than a test reporting a property it cannot observe.
