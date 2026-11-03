"""Reviewer tests for the merge of origin/develop: the four follow-up items and the merge's residue.

Derived from the requirements, not from the code:
(a) item 1, no banner when no check applies; (b) item 3, the check's new name everywhere and the name it replaced
nowhere it was listed; (c) item 12, the two dead collectors gone and `format_sources_rst` alive; (d) one
definition sentence in both captions and the `MH` name; (e) the 13 conflicted files, with no dropped run-branch hunk
and none of the names the migration renames in the lines `origin/develop` adds.

The replaced check name is assembled from parts so this file does not itself hold it. No solver runs; the git reads are
read-only (`git diff`, `git grep`, `git show`).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
OLD_NAME = "Cross-sim " + "byte-identity"
NEW_NAME = "Byte identity across compute configurations"
BASE = "38656ee7"  # the run's base commit
DEVELOP = "fa98781e"  # origin/develop at the merge
RUN_TIP_BEFORE_1A8_MERGE = "0fdcf056"  # the first parent of the merge 736b21a5 of origin/develop

CONFLICTED = (
    "_caption_pairing_allowlist.yaml",
    "src/hhemt/analysis_validation.py",
    "src/hhemt/eda/_config_diff.py",
    "src/hhemt/report_renderers/cross_experiment_errors_and_warnings.py",
    "src/hhemt/report_renderers/cross_experiment_intercomparison_maps.py",
    "src/hhemt/report_renderers/errors_and_warnings.py",
    "src/hhemt/report_templates/captions/cross_experiment_intercomparison.rst",
    "src/hhemt/report_templates/captions/cross_experiment_intercomparison_maps.rst",
    "src/hhemt/report_templates/captions/eda_compute_sensitivity.rst",
    "src/hhemt/report_templates/captions/per_sim_event_page.rst",
    "src/hhemt/report_templates/captions/per_sim_peak_flood_depth.rst",
    "tests/test_coupled_resume_validity.py",
    "tests/test_cross_experiment_diff_maps_render.py",
)


def _git(*args: str) -> str:
    out = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    return out.stdout


def _have(commit: str) -> bool:
    return subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", commit], capture_output=True).returncode == 0


needs_history = pytest.mark.skipif(
    not (_have(BASE) and _have(DEVELOP) and _have(RUN_TIP_BEFORE_1A8_MERGE)), reason="run history not in this clone"
)


# ---------------------------------------------------------------------------------------------
# (a) item 1: no banner when no check applies
# ---------------------------------------------------------------------------------------------


def _check(name: str, passed: bool, applicable: bool):
    from hhemt.analysis_validation import CheckResult

    return CheckResult(name=name, level="aggregate", passed=passed, summary="s", applicable=applicable)


@pytest.mark.parametrize("passed", [True, False], ids=["all_pass_branch", "all_fail_branch"])
def test_the_banner_is_empty_when_no_check_applies_in_both_old_branches(passed):
    """Property: with zero applicable checks the banner returns nothing, whichever of the two old
    branches (`✓ 0 of 0 ... passed`, `✗ 0 of 0 ... failed`) the record would have taken.

    Class: reports of two not-applicable checks, all marked passed or all marked failed. A not-applicable check
    that is marked failed makes `overall_passed` False, which is the old fail branch.
    Why required: both strings state nothing; the omission note is the page's one statement.
    Kills it: restoring either branch for the empty case; a banner for a record with only N/A checks.
    Passes under: any implementation that returns an empty string or a falsy value.
    """
    from hhemt.analysis_validation import ValidationReport
    from hhemt.report_renderers.errors_and_warnings import _render_overall_banner

    report = ValidationReport(checks=[_check("a", passed, False), _check("b", passed, False)])
    assert report.overall_passed is passed  # control: the two reports really take the two old branches
    assert _render_overall_banner(report) == ""


def test_the_banner_is_kept_when_a_check_applies():
    """Control for the test above: one applicable check, one N/A check: the banner prints its tally."""
    from hhemt.analysis_validation import ValidationReport
    from hhemt.report_renderers.errors_and_warnings import _render_overall_banner

    ok = _render_overall_banner(ValidationReport(checks=[_check("a", True, True), _check("b", True, False)]))
    bad = _render_overall_banner(ValidationReport(checks=[_check("a", False, True), _check("b", True, False)]))
    assert "1 of 1 applicable checks passed" in ok
    assert "applicable checks failed" in bad


@pytest.mark.parametrize("passed", [True, False], ids=["all_pass", "all_fail"])
def test_the_page_with_no_applicable_check_prints_the_omission_note_and_no_banner(passed, tmp_path, monkeypatch):
    """Property: through the real renderer, the page for an all-N/A record prints the omission note and
    no `applicable checks` tally; a record with no checks at all keeps its own notice.

    Class: the `P07` stub analysis with the loaded record replaced by (a) two N/A checks passed, (b) two N/A checks
    failed, (c) one applicable and one N/A check (control), (d) no checks.
    Why required: the existing test renders only the stub kind; the failing branch is not on any stub.
    Kills it: a banner on either N/A page; an omission note lost with the banner; the missing-record notice lost.
    Passes under: any rendering that keeps the three statements.
    """
    import hhemt.analysis_validation as av
    from hhemt.analysis_validation import ValidationReport
    from tests.test_caption_page_kinds import _by_id, render_page

    def render_with(report) -> str:
        monkeypatch.setattr(av, "load_validation_report", lambda analysis: report)
        return render_page(_by_id("P07-failures"), tmp_path / f"r{len(report.checks)}{passed}{id(report)}")

    na = render_with(ValidationReport(checks=[_check("a", passed, False), _check("b", passed, False)]))
    assert "applicable checks" not in na
    assert "2 checks are not shown: they do not apply to this analysis." in na
    mixed = render_with(ValidationReport(checks=[_check("a", True, True), _check("b", True, False)]))
    assert "1 of 1 applicable checks passed" in mixed and "1 check is not shown" in mixed
    empty = render_with(ValidationReport(checks=[]))
    assert "applicable checks" not in empty and "Check results are not shown" in empty


# ---------------------------------------------------------------------------------------------
# (b) item 3: the rename
# ---------------------------------------------------------------------------------------------


def test_the_check_is_named_the_same_in_the_producer_the_vocabulary_and_the_notebook():
    """Property: the producer's `name`, the vocabulary table's key and display name, and the notebook
    heading all read `Byte identity across compute configurations`.

    Class: the three places that name the check (`cross_sim_identity.py` name, `errors_and_warnings.py` table,
    `_notebook.py` heading), the first and third by source text because importing them runs nothing.
    Why required: `tests/test_iter7_check_vocabulary.py` holds the producer's names equal to the table's keys;
    a name changed in one place gives a check with no description on the page.
    Kills it: a half rename; a display name differing from its key; the heading left old.
    Passes under: a name held in one constant that the three read.
    """
    from hhemt.report_renderers.errors_and_warnings import _CHECK_VOCABULARY

    assert NEW_NAME in _CHECK_VOCABULARY and OLD_NAME not in _CHECK_VOCABULARY
    assert _CHECK_VOCABULARY[NEW_NAME][0] == NEW_NAME
    producer = (REPO / "src/hhemt/eda/cross_sim_identity.py").read_text()
    assert re.search(r'^\s*name = "' + re.escape(NEW_NAME) + '"', producer, re.M)
    notebook = (REPO / "src/hhemt/eda/_notebook.py").read_text()
    assert f"## {NEW_NAME} (source-analysis only)" in notebook


def test_the_old_name_occurs_nowhere_in_src_tests_or_docs_and_the_new_name_is_pinned():
    """Property (census): the exact old string returns no line outside the bundle docs,
    including the three style allowlists and every pin; the new name is pinned in the EDA tests.

    Class: every text file of `src`, `tests`, `docs` (minus `docs/reference/bundles`), whole-tree walk including
    untracked files, matching the exact capitalised string; lowercase prose mentions are outside the rename set.
    Why required: a `git grep` skips untracked files and the style test fails on a stale allowlist entry.
    Kills it: a surviving pin, comment or allowlist baseline entry.
    Passes under: a tree where every mention is reworded.
    """
    hits = []
    for sub in ("src", "tests", "docs"):
        for p in sorted((REPO / sub).rglob("*")):
            rel = p.relative_to(REPO).as_posix()
            if not p.is_file() or "docs/reference/bundles" in rel or "__pycache__" in rel:
                continue
            if p.suffix not in {".py", ".md", ".rst", ".yaml", ".yml", ".json", ".txt", ".toml", ".cfg", ".j2"}:
                continue
            if OLD_NAME in p.read_text(errors="replace"):
                hits.append(rel)
    assert hits == [], hits
    for rel in ("tests/test_report_text_eda.py", "tests/test_eda_cross_sim_identity.py"):
        assert NEW_NAME in (REPO / rel).read_text(), rel
    # control: the matcher fires on the property (the capitalised exact string), not on lowercase prose
    assert OLD_NAME in f"x {OLD_NAME} y" and OLD_NAME not in "a cross-sim byte-identity plot"


@pytest.mark.parametrize("name", ["_notebook.py", "cross_sim_identity.py", "errors_and_warnings.py"])
def test_the_style_allowlists_hold_no_baseline_entry_for_the_old_name(name):
    """Property: the three style allowlists carry no entry for the old string.

    Class: the three YAML files that list the check; each must still parse and keep its `source` line.
    Why required: the style test fails on an entry that matches no hit, so a leftover entry is a red test.
    Kills it: an entry kept; a file emptied of its `source`.
    Passes under: `entries: []`.
    """
    import yaml

    doc = yaml.safe_load((REPO / "tests/report_style/allowlist" / f"{name}.yaml").read_text())
    assert doc["source"].endswith(name)
    assert all(OLD_NAME not in str(e) for e in (doc.get("entries") or []))


# ---------------------------------------------------------------------------------------------
# (c) item 12: the dead collectors
# ---------------------------------------------------------------------------------------------


def test_the_two_dead_collectors_are_gone_and_format_sources_rst_still_works():
    """Property: neither collector exists in the module or is called anywhere in the tree;
    `format_sources_rst`, the function the other renderers use, still returns a reStructuredText list.

    Class: the module attributes; a walk of `src` and `tests` for the two names (this file builds them from parts).
    Why required: both collectors and the one test that called one; a deletion that left a caller is an
    ImportError at render time.
    Kills it: a collector kept; the shared helper removed with them.
    Passes under: removal by either deletion or move out of the module.
    """
    from hhemt.report_renderers import _figure_emission as fe

    for name in ("collect_per_sim_" + "source_paths", "collect_sensitivity_" + "source_paths"):
        assert not hasattr(fe, name), name
        called = [
            p.relative_to(REPO).as_posix()
            for sub in ("src", "tests")
            for p in (REPO / sub).rglob("*.py")
            if name in p.read_text(errors="replace")
        ]
        assert called == [], (name, called)
    assert callable(fe.format_sources_rst)  # control: the helper remains and is the live function
    out = fe.format_sources_rst([{"path": "a/b.csv", "variables": ["col"]}, "a/c.zarr"])
    assert "b.csv" in out and "c.zarr" in out and "col" in out


# ---------------------------------------------------------------------------------------------
# (d) the MH name
# ---------------------------------------------------------------------------------------------

_CAPTIONS = REPO / "src" / "hhemt" / "report_templates" / "captions"


def _definition_sentences(text: str) -> list[str]:
    flat = " ".join(text.split())
    return [s for s in re.split(r"(?<=[.])\s+(?=[A-Z*])", flat) if "running maximum" in s]


def test_both_captions_carry_one_identical_definition_naming_the_solver_maximum():
    """Property: the two captions hold the same sentence defining the peak flood depth as the solver's running
    maximum, and neither says `post-processing`; the sentence names the solver's own variable.

    Class: `per_sim_event_page.rst` and `per_sim_peak_flood_depth.rst`; sentences containing `running maximum`.
    Why required: the two pages once defined the quantity two ways (a maximum over prints; the solver's running
    maximum); only the second is true of the stored `MH`.
    Kills it: two wordings; the old `maximum water depth across all simulation timesteps` sentence; `post-processing`.
    Passes under: any single shared wording that keeps the phrase.
    """
    a = _definition_sentences((_CAPTIONS / "per_sim_event_page.rst").read_text())
    b = _definition_sentences((_CAPTIONS / "per_sim_peak_flood_depth.rst").read_text())
    assert a and a == b, (a, b)
    assert "MH" in a[0] and "solver" in a[0].lower()
    for rel in ("per_sim_event_page.rst", "per_sim_peak_flood_depth.rst"):
        text = (_CAPTIONS / rel).read_text()
        assert "post-processing" not in text and "across all simulation timesteps" not in text, rel


def test_the_solver_maximum_prefix_maps_to_the_layout_27_depth_name():
    """Property: `TRITON_RAW_VAR_BY_PREFIX["MH"]` is `max_flood_depth`, and the peak-flood-depth
    renderer reads that name.

    Class: the one dictionary entry and the renderer's source text.
    Kills it: the entry mapped to any other name.
    """
    from hhemt.process_simulation import TRITON_RAW_VAR_BY_PREFIX

    assert TRITON_RAW_VAR_BY_PREFIX["MH"] == "max_flood_depth"
    src = (REPO / "src/hhemt/report_renderers/per_sim_peak_flood_depth.py").read_text()
    assert 'ds["max_flood_depth"]' in src or "max_flood_depth" in src


# ---------------------------------------------------------------------------------------------
# (e) the 13 conflicted files
# ---------------------------------------------------------------------------------------------


@needs_history
@pytest.mark.parametrize("rel", [f for f in CONFLICTED if not f.endswith(".yaml")])
def test_no_hunk_the_merge_resolution_dropped_survives_in_a_conflicted_file(rel):
    """Property: every line the run branch removed from a conflicted file (its non-trivial lines) is
    absent from the file now, so the resolution to `origin/develop`'s side did not leave a dropped hunk behind.

    Class: the `-` lines (longer than 25 characters, stripped) of `git diff 38656ee7 {run-branch tip}`, i.e. what
    the run took out, compared as whole stripped lines with the working file.
    Why required: `git checkout --theirs` keeps develop's text, which still contains the lines the run branch had
    rewritten; each hunk must be re-applied.
    Kills it: a conflicted file whose rewritten line is still in its pre-run form.
    Passes under: a file where develop's rewrite replaced the line with different text.
    """
    # The run branch's tip when the merge was made, pinned: HEAD moves with every later merge and then also
    # holds the later chunks' own edits (a re-indented block reads as a removed line that is still present).
    parent = RUN_TIP_BEFORE_1A8_MERGE
    removed = [
        ln[1:].strip()
        for ln in _git("diff", "-U0", BASE, parent, "--", rel).splitlines()
        if ln.startswith("-") and not ln.startswith("---") and len(ln[1:].strip()) > 25
    ]
    present = {ln.strip() for ln in (REPO / rel).read_text().splitlines()}
    assert [r for r in removed if r in present] == [], rel


@needs_history
def test_the_lines_origin_develop_added_name_no_layout_26_name_and_the_census_has_a_control():
    """Property: the names the migration renames that
    `origin/develop` introduced after the run's base (17 lines in 10 files) are absent from the working tree.

    Class: the `+` lines of `git diff 38656ee7 fa98781e` that hold a name the migration renames (50 SWMM, hydrology
    and EDA names, the TRITON depth, discharge and time names); each file that carries one is searched for the same
    names now.
    Must-fire control: the same matcher over those `+` lines finds at least 15 lines in at least 8 files.
    Why required: the merge brings these names in with develop's side; renaming only the conflicted ones would
    leave the three test files only develop changed.
    Kills it: a surviving name the migration renames, in a file develop changed.
    Passes under: any rewrite of those lines.
    """
    from tests.test_review_v0028_migration import (
        ATTR_KEYS,
        EDA_COMPUTE,
        HYDROLOGY,
        LINK,
        NODE,
        RUNOFF,
        TRITON,
        WATER_QUALITY,
    )

    names = set(TRITON) | set(NODE) | set(LINK) | set(HYDROLOGY) | set(RUNOFF) | set(WATER_QUALITY) | set(EDA_COMPUTE)
    names |= set(ATTR_KEYS) | {"timestep_min", "time_hr"}
    pat = re.compile(
        r"(?<![A-Za-z0-9_])(" + "|".join(sorted(map(re.escape, names), key=len, reverse=True)) + r")(?![A-Za-z0-9_])"
    )
    added: dict[str, int] = {}
    current = None
    for ln in _git("diff", "-U0", BASE, DEVELOP).splitlines():
        if ln.startswith("+++ b/"):
            current = ln[6:]
        elif ln.startswith("+") and not ln.startswith("+++") and current and pat.search(ln):
            added[current] = added.get(current, 0) + 1
    assert sum(added.values()) >= 15 and len(added) >= 8, added  # control: the matcher finds develop's lines
    still = {}
    quoting = {
        "scripts/vocabulary_freeze.yaml",
        "tests/test_cf_conventions.py",
        "tests/test_review_unit_discharge_relabel.py",
        "tests/test_review_flood_depth_relabel.py",
        "tests/test_review_swmm_tier_names.py",
        "tests/test_review_v0028_migration.py",
        "tests/test_version_migration_v0028.py",
    }
    for rel in added:
        p = REPO / rel
        if rel in quoting or not p.is_file() or "V0028" in rel:
            continue
        n = sum(1 for line in p.read_text(errors="replace").splitlines() if pat.search(line))
        if n:
            still[rel] = n
    assert still == {}, still
