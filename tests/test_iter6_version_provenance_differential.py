"""Differential tests for the iteration6-version-provenance control-flow sites.

Every test here is written to FAIL against the pre-fix tree and pass after the
deliverable's specs land. A provenance test that is green on both sides is not a
differential -- that is the exact failure this file exists to avoid, and it is the
reason `test_resolve_producing_stamp_is_process_stable` (which passes identically
whether the function derives a value or returns a constant) did not catch the
static-pin defect.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_PEP440_LOCAL = re.compile(r"^\d+\.\d+\.\d+\+\d+\.g[0-9a-f]{7,40}$")


# ---- S1 provenance._describe_version -------------------------------------- #
def test_s1_describe_version_is_pep440_local_not_the_static_pin():
    from hhemt.provenance import _describe_version

    v = _describe_version()
    assert _PEP440_LOCAL.match(v), f"not a PEP-440 local version: {v!r}"
    assert v != "0.1.0", "returned the static pyproject pin"


# ---- S2 provenance._is_dirty ---------------------------------------------- #
def test_s2_is_dirty_agrees_with_git_status():
    from hhemt.bundle._emit import _toolkit_source_dir
    from hhemt.provenance import _is_dirty

    actual = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_toolkit_source_dir(),
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    assert _is_dirty() is actual


# ---- S3 provenance.producing_stamp ---------------------------------------- #
def test_s3_producing_stamp_has_exactly_three_fields():
    from hhemt.provenance import producing_stamp

    s = producing_stamp()
    assert set(s) == {"hhemt_sha", "hhemt_version", "hhemt_dirty"}
    assert s["hhemt_dirty"] in ("true", "false")


# ---- S4 process_simulation._resolve_producing_stamp ----------------------- #
def test_s4_resolve_producing_stamp_version_is_derived_not_static():
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing as P

    _sha, semver = P._resolve_producing_stamp()
    assert "+" in semver, f"still the static pyproject pin: {semver!r}"


# ---- S5 reprocess_snakefile_generator emitted preamble -------------------- #
def test_s5_reprocess_preamble_routes_through_the_single_minter():
    src = Path(__import__("hhemt.reprocess_snakefile_generator", fromlist=["x"]).__file__).read_text()
    assert "_describe_version" in src
    assert '_pkg_version("hhemt")' not in src


# ---- S6 _derive_version_from_sha ------------------------------------------ #
def test_s6_derive_version_from_sha_derives_and_refuses():
    from hhemt.report_renderers.cross_experiment_compatibility import _derive_version_from_sha

    got = _derive_version_from_sha("01655abb60c2")
    assert got == "0.1.0+241.g01655abb60c2", got
    assert _derive_version_from_sha("deadbeefdead") is None


# ---- S7 the "+"-absent fallback branch in _combine_provenance_rows -------- #
def test_s7_combine_rows_backfills_a_static_pin_from_the_tree_sha(tmp_path):
    xr = pytest.importorskip("xarray")
    from hhemt.report_renderers.cross_experiment_compatibility import _combine_provenance_rows

    crate = tmp_path / "child_crates" / "exp_a"
    crate.mkdir(parents=True)
    tree = xr.DataTree()
    tree.attrs["hhemt_producing_version"] = "0.1.0"
    tree.attrs["hhemt_producing_sha"] = "01655abb60c2"
    tree.attrs["triton_producing_sha"] = "9db367ddc79f86c7f708686d1dd805dc992fb0a4"
    tree.to_zarr(crate / "sensitivity_datatree.zarr", consolidated=False)

    rows = _combine_provenance_rows(tmp_path)
    assert rows and rows[0]["toolkit_version"] == "0.1.0+241.g01655abb60c2"


# ---- S8 _bv build-split marker -------------------------------------------- #
def test_s8_build_split_is_marked_in_the_build_column():
    from hhemt.report_renderers.cross_experiment_compatibility import _provenance_table_html

    rows = [
        {
            "experiment_id": "a",
            "role": "clean",
            "model": "TRITON",
            "n_subs": 1,
            "toolkit_sha": "aaa",
            "toolkit_version": "0.1.0+240.gaaa",
            "solver_sha": "s1",
        },
        {
            "experiment_id": "b",
            "role": "clean",
            "model": "TRITON",
            "n_subs": 1,
            "toolkit_sha": "aaa",
            "toolkit_version": "0.1.0+240.gaaa",
            "solver_sha": "s1",
        },
        {
            "experiment_id": "c",
            "role": "resume",
            "model": "TRITON",
            "n_subs": 1,
            "toolkit_sha": "bbb",
            "toolkit_version": "0.1.0+241.gbbb",
            "solver_sha": "s1",
        },
    ]
    html = _provenance_table_html(rows)
    assert "0.1.0+241.gbbb *" in html, "minority build not marked"


# ---- S9 caption states constancy on the axis that did NOT split ----------- #
def test_s9_caption_states_build_constancy_when_builds_agree():
    from hhemt.report_renderers.cross_experiment_compatibility import _provenance_table_html

    rows = [
        {
            "experiment_id": "a",
            "role": "clean",
            "model": "TRITON",
            "n_subs": 1,
            "toolkit_sha": "aaa",
            "toolkit_version": "0.1.0+241.gaaa",
            "solver_sha": "s1",
        },
        {
            "experiment_id": "b",
            "role": "resume",
            "model": "TRITON",
            "n_subs": 1,
            "toolkit_sha": "aaa",
            "toolkit_version": "0.1.0+241.gaaa",
            "solver_sha": "s2",
        },
    ]
    html = _provenance_table_html(rows)
    assert "hhemt build is identical across every row" in html


# ---- S10 header relabel ---------------------------------------------------- #
def test_s10_headers_name_the_product_not_the_word_toolkit():
    from hhemt.report_renderers.cross_experiment_compatibility import _provenance_table_html

    rows = [
        {
            "experiment_id": "a",
            "role": "clean",
            "model": "TRITON",
            "n_subs": 1,
            "toolkit_sha": "aaa",
            "toolkit_version": "0.1.0+241.gaaa",
            "solver_sha": "s1",
        }
    ]
    html = _provenance_table_html(rows)
    # Assert the INTENT, not two exact literals. The rule S10 encodes is that provenance
    # headers name the PRODUCT (hhemt) and never the word Toolkit. Pinning the exact
    # strings `<th>hhemt sha</th>` / `<th>hhemt build</th>` pinned one PROPOSED wording,
    # so the renderer refining them to `hhemt bundle sha` and `hhemt build (data-producing)`
    # -- strictly more specific, and satisfying the rule -- reddened this test while the
    # relabel it guards had in fact landed. An assertion must test the invariant rather
    # than one position consistent with it.
    headers = re.findall(r"<th>([^<]*)</th>", html)
    sha_hdrs = [h for h in headers if "sha" in h.lower() and "solver" not in h.lower()]
    build_hdrs = [h for h in headers if "build" in h.lower() or "version" in h.lower()]
    assert sha_hdrs and all("hhemt" in h for h in sha_hdrs), sha_hdrs
    assert build_hdrs and all("hhemt" in h for h in build_hdrs), build_hdrs
    assert not any("toolkit" in h.lower() for h in headers), headers


# ---- S11 plots-stage capture at the sidecar choke point ------------------- #
def test_s11_manifest_sidecar_carries_the_producing_stamp(tmp_path):
    import json

    from hhemt.report_renderers._figure_emission import _emit_manifest_sidecar

    p = _emit_manifest_sidecar(tmp_path / "fig.html", {"output_format": "html"})
    payload = json.loads(Path(p).read_text())
    assert {"hhemt_sha", "hhemt_version", "hhemt_dirty"} <= set(payload)


# ---- S12 / S13 completeness check + its collector -------------------------- #
#
# The test this replaces asserted two hasattr calls and was named
# `..._is_registered_and_graceful`. Both halves of that name were false and the test
# could not see either: the function was absent from validate_analysis, and every one of
# its four returns raised TypeError (`detail=` for a field named `details`, and no
# `level=` at all). hasattr returns True on a function that cannot execute, so the
# assertion passed over exactly the defect its name claimed to cover. The lesson the
# replacements encode: an assertion about a callable must CALL it.


def _stage_stamps(**overrides):
    """A full stage map with every stage stamped at build A, minus the overrides.

    Built from `_PROVENANCE_STAGES` rather than a literal dict so a new stage cannot
    silently shrink the denominator these arms are comparing.
    """
    from hhemt.analysis_validation import _PROVENANCE_STAGES

    base = {"hhemt_sha": "a" * 40, "hhemt_version": "v0.1.0+A", "hhemt_dirty": "false"}
    stamps = {s: dict(base) for s in _PROVENANCE_STAGES}
    stamps.update(overrides)
    return stamps


def test_s12_check_provenance_completeness_constructs_and_is_registered():
    """The check RUNS, and validate_analysis names it.

    Ordering guard as much as a feature test: registering this check while it still
    raised would leave the read-model unwritten -- which fails the rule loudly on a
    FRESH tree (the declared output at workflow.py:3467 is missing) but ships the
    PREVIOUS generation's validation data silently on a RE-RUN, because the stale file
    satisfies that declaration. This test fails on either half alone.
    """
    import inspect

    from hhemt import analysis_validation as av

    src = inspect.getsource(av.validate_analysis)
    assert "check_provenance_completeness(analysis)" in src, (
        "check_provenance_completeness is not registered in validate_analysis"
    )


def test_s13_controlled_pair_mixed_build_fails_uniform_build_passes(monkeypatch):
    """THE controlled pair the contract asks for, at the seam the check reads.

    A deliberately mixed-version analysis must fail; an otherwise-identical
    single-version one must not. Both arms are dictionaries, so the pair costs
    microseconds and is deterministic -- a real two-version campaign can neither be
    produced reproducibly nor be a precondition for the gate that blocks campaigns.
    """
    from hhemt import analysis_validation as av

    mixed = _stage_stamps(report={"hhemt_sha": "b" * 40, "hhemt_version": "v0.1.0+B", "hhemt_dirty": "false"})
    monkeypatch.setattr(av, "_collect_stage_stamps", lambda _a: mixed)
    bad = av.check_provenance_completeness(object())
    assert bad.passed is False, "a mixed-build analysis must not pass"
    assert "disagree" in bad.summary

    monkeypatch.setattr(av, "_collect_stage_stamps", lambda _a: _stage_stamps())
    good = av.check_provenance_completeness(object())
    assert good.passed is True, "a single-build analysis must not warn"


def test_s14_summary_discloses_its_denominator(monkeypatch):
    """Gotcha 71: "no discrepancies" over zero examined stages must not read as a pass."""
    from hhemt import analysis_validation as av
    from hhemt.analysis_validation import _PROVENANCE_STAGES

    monkeypatch.setattr(av, "_collect_stage_stamps", lambda _a: dict.fromkeys(_PROVENANCE_STAGES))
    res = av.check_provenance_completeness(object())
    assert f"/{len(_PROVENANCE_STAGES)}" in res.summary, (
        "the summary must name how many stages were examined, not only how many were clean"
    )
    assert res.level == "aggregate"


def test_s15_dirty_checkout_is_a_failure(monkeypatch):
    from hhemt import analysis_validation as av

    dirty = _stage_stamps(plots={"hhemt_sha": "a" * 40, "hhemt_version": "v0.1.0+A", "hhemt_dirty": "true"})
    monkeypatch.setattr(av, "_collect_stage_stamps", lambda _a: dirty)
    res = av.check_provenance_completeness(object())
    assert res.passed is False
    assert "DIRTY" in res.summary


def test_s16_stage_carriers_are_pairwise_distinct_paths():
    """Contract property 3, made falsifiable rather than left structural.

    "A re-render must not replace the version that produced the science with the one
    that drew the figure" holds today because each stage writes a DIFFERENT file --
    the per-event coordinate, the tree root attr, the figure sidecars, and three
    separate manifests. Nothing asserted that. A refactor that made one writer stamp
    two stages' carriers would break the property silently, so the separation is
    pinned here by reading the collector's own source for the distinct carriers it
    consults.
    """
    import inspect

    from hhemt import analysis_validation as av

    src = inspect.getsource(av._collect_stage_stamps)
    for carrier in ("bundle_manifest.json", "combined_bundle_manifest.json", "report_manifest.json"):
        assert carrier == carrier and src.count(f'"{carrier}"') >= 1, f"{carrier} no longer read"
    assert '"plots"' in src and "hhemt_producing_sha" in src, (
        "the plots and consolidate carriers must remain separate reads"
    )
