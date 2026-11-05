"""The provenance summary says WHY each stage is unstamped (C1(c), DR-2 option 1).

Four properties, each a differential against the parent commit, where one blanket sentence
named no category and the read-model recorded no build:

1. With every counted stage absent, the summary names both absence categories.
2. Every summary arm carries the note that plots and report are not counted, and a `plots`
   or `report` stamp -- at any build, dirty or clean -- changes no verdict (D-46a option 3;
   the one-render lag clause and the plots and report table rows are retired with it).
3. No capture site exists for `setup` or `sim` -- the falsifier of the table's two
   "no site in this toolkit" rows, so the table cannot go stale silently.
4. `persist_validation_report` writes a `producing_stamp` key beside `checks`.

The counted-stage verdicts are pinned by `tests/test_iter6_version_provenance_differential.py`,
which passes unchanged; this file pins only that the two uncounted stages cannot move them.

No solver, no compile and no `synth_*` fixture: every test writes only `tmp_path` files.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import hhemt
from hhemt import analysis_validation as av
from hhemt.analysis_validation import _PROVENANCE_NOT_COUNTED_NOTE, _PROVENANCE_STAGES, _VALIDATED_STAGES

_CATEGORIES = (
    "no site in this toolkit",
    "not decidable from the tree",
)
_RETIRED_CATEGORY = "captured on the next render, by construction"
_RETIRED_LAG = "Report stage lag"
_OTHER_BUILD = {"hhemt_sha": "b" * 40, "hhemt_version": "v0.1.0+B", "hhemt_dirty": "false"}
_DIRTY = {"hhemt_sha": "a" * 40, "hhemt_version": "v0.1.0+A", "hhemt_dirty": "true"}


def _stamped(**overrides):
    base = {"hhemt_sha": "a" * 40, "hhemt_version": "v0.1.0+A", "hhemt_dirty": "false"}
    stamps = {s: dict(base) for s in _PROVENANCE_STAGES}
    stamps.update(overrides)
    return stamps


def _run(monkeypatch, stamps):
    monkeypatch.setattr(av, "_collect_stage_stamps", lambda _a: stamps)
    return av.check_provenance_completeness(object())


def test_table_is_closed_over_the_counted_stages():
    """The domain is closed: one table row per COUNTED stage, no more, no fewer.

    Property: the table's keys equal `_VALIDATED_STAGES`, so plots and report (stamped after
    this check runs, never counted) have no row. Kills: restoring either row, or adding a
    counted stage without one. A second correct implementation: the same four rows held in
    any other container or order.
    """
    assert set(av._PROVENANCE_ABSENCE_TABLE) == set(_VALIDATED_STAGES)
    assert len(av._PROVENANCE_ABSENCE_TABLE) == len(_VALIDATED_STAGES)
    assert {cat for cat, _ in av._PROVENANCE_ABSENCE_TABLE.values()} == set(_CATEGORIES)
    # Control varying the PROPERTY: the uncounted stages are still stampable stages.
    assert {"plots", "report"} <= set(_PROVENANCE_STAGES) - set(av._PROVENANCE_ABSENCE_TABLE)


def test_every_counted_stage_absent_names_both_categories(monkeypatch):
    res = _run(monkeypatch, dict.fromkeys(_PROVENANCE_STAGES))
    assert res.passed is True
    for category in _CATEGORIES:
        assert category in res.summary, f"the missing arm must name the category {category!r}"
    assert _RETIRED_CATEGORY not in res.summary
    # The blanket sentence that called every absence benign is gone.
    assert "legitimately has no stamp" not in res.summary
    # Each absent stage is named under its own category, not merely somewhere.
    for stage, (category, _sentence) in av._PROVENANCE_ABSENCE_TABLE.items():
        assert re.search(rf"{re.escape(category)} \([^)]*\b{stage}\b", res.summary), (
            f"{stage} must be grouped under {category!r}"
        )


def test_absence_clause_groups_only_the_categories_present():
    """Must-fire control for the grouping: it varies WHICH categories are present."""
    only_processing = av._absence_clause(["processing"])
    assert "not decidable from the tree (processing)" in only_processing
    assert "no site in this toolkit" not in only_processing
    assert _RETIRED_CATEGORY not in only_processing
    assert av._absence_clause([]) == "none"
    # A stage with no row is stated as uncategorised, never dropped or given a counted category.
    assert av._absence_clause(["plots"]) == "no absence category recorded (plots)"


def test_all_three_missing_listing_arms_use_the_grouping(monkeypatch):
    """Dirty, build-disagreement and missing arms each state the categories present."""
    # missing arm: setup + sim + processing absent, the rest agree
    miss = _run(monkeypatch, _stamped(setup=None, sim=None, processing=None))
    assert miss.passed is True
    assert "no site in this toolkit (setup, sim)" in miss.summary
    assert "not decidable from the tree (processing)" in miss.summary

    # dirty arm: one dirty COUNTED stage, setup + sim absent
    dirty = _run(monkeypatch, _stamped(setup=None, sim=None, consolidate=dict(_DIRTY)))
    assert dirty.passed is False
    assert "no site in this toolkit (setup, sim)" in dirty.summary

    # disagreement arm: consolidate at another sha, setup + sim absent
    disagree = _run(
        monkeypatch,
        _stamped(
            setup=None,
            sim=None,
            consolidate={"hhemt_sha": "b" * 40, "hhemt_version": "v0.1.0+B", "hhemt_dirty": "false"},
        ),
    )
    assert disagree.passed is False
    assert "disagree" in disagree.summary
    assert "no site in this toolkit (setup, sim)" in disagree.summary


_ARM_STAMPS = {
    # all stamped, one build: the all-agree arm
    "all_agree": (lambda: _stamped(), True),
    # a counted stage absent: the missing arm
    "missing": (lambda: _stamped(sim=None), True),
    # a counted stage at another build: the disagreement arm
    "disagree": (lambda: _stamped(processing=dict(_OTHER_BUILD)), False),
    # a counted stage dirty: the dirty arm
    "dirty": (lambda: _stamped(consolidate=dict(_DIRTY)), False),
}


@pytest.mark.parametrize("arm", sorted(_ARM_STAMPS))
def test_every_summary_arm_carries_the_not_counted_note(monkeypatch, arm):
    """Property: each of the four summary arms states that plots and report are not counted.

    Kills: any arm that omits `_PROVENANCE_NOT_COUNTED_NOTE` (the disagreement arm omitted it
    on the merge commit). A second correct implementation: the note built in one place and
    appended after the arm is chosen, instead of per arm.
    """
    factory, expect_passed = _ARM_STAMPS[arm]
    res = _run(monkeypatch, factory())
    assert res.passed is expect_passed
    assert _PROVENANCE_NOT_COUNTED_NOTE.strip() in res.summary
    assert _RETIRED_LAG not in res.summary


@pytest.mark.parametrize("arm", sorted(_ARM_STAMPS))
@pytest.mark.parametrize(
    "uncounted",
    [
        {"report": dict(_OTHER_BUILD)},
        {"plots": dict(_OTHER_BUILD)},
        {"report": dict(_DIRTY)},
        {"plots": dict(_DIRTY)},
        {"plots": None, "report": None},
    ],
    ids=["report_other_build", "plots_other_build", "report_dirty", "plots_dirty", "both_absent"],
)
def test_plots_and_report_stamps_change_no_verdict(monkeypatch, arm, uncounted):
    """A `plots` or `report` stamp -- other build, dirty, or absent -- moves neither the
    verdict nor the stage count, in any of the four arms.

    Class: every stamp of the two uncounted stages. Kills: counting either stage again.
    Control varying the property: the same stamps on a COUNTED stage DO move the verdict
    (`test_counted_stage_at_another_build_still_fails`).
    """
    factory, expect_passed = _ARM_STAMPS[arm]
    baseline = _run(monkeypatch, factory())
    res = _run(monkeypatch, {**factory(), **uncounted})
    assert res.passed is baseline.passed is expect_passed
    assert res.summary == baseline.summary


def test_counted_stage_at_another_build_still_fails(monkeypatch):
    """Must-fire control: the same other-build stamp on a COUNTED stage fails the check."""
    res = _run(monkeypatch, _stamped(processing=dict(_OTHER_BUILD)))
    assert res.passed is False
    assert "disagree" in res.summary
    assert f"/{len(_VALIDATED_STAGES)}" in res.summary


def test_no_capture_site_for_setup_or_sim():
    """The falsifier of the table's two "no site in this toolkit" rows.

    The matcher is the one DR-2 names. The control proves the matcher can fire at all: the
    same pattern, run over `provenance.py`, MUST match the `report` capture site. A zero on
    the two runner files is only evidence of absence if that control is non-zero.
    """
    root = Path(hhemt.__file__).parent
    pattern = re.compile(r"producing_stamp|append_stage_provenance")

    for name in ("run_simulation_runner.py", "setup_workflow.py"):
        path = root / name
        assert path.exists(), f"{name} is the file whose capture sites are being counted; it must exist"
        hits = [ln for ln in path.read_text().splitlines() if pattern.search(ln)]
        assert hits == [], (
            f"a capture site appeared in {name} ({hits[:3]}): the `setup`/`sim` rows of "
            f"_PROVENANCE_ABSENCE_TABLE say no site exists and must be revisited"
        )

    report_site = re.compile(r'append_stage_provenance\(.*"report"\)')
    control = [ln for ln in (root / "provenance.py").read_text().splitlines() if report_site.search(ln)]
    assert control, "control: the matcher must find the report capture site in provenance.py"


def test_persist_validation_report_writes_producing_stamp_beside_checks(tmp_path, monkeypatch):
    import json

    minted = {"hhemt_sha": "c" * 40, "hhemt_version": "v0.1.0+C", "hhemt_dirty": "false"}
    monkeypatch.setattr("hhemt.provenance.producing_stamp", lambda: dict(minted))
    monkeypatch.setattr(av, "validate_analysis", lambda _a, df_status=None: av.ValidationReport(checks=[]))
    analysis = SimpleNamespace(analysis_paths=SimpleNamespace(analysis_dir=tmp_path))

    out = av.persist_validation_report(analysis)

    payload = json.loads(out.read_text())
    assert set(payload) == {"checks", "producing_stamp"}
    assert payload["producing_stamp"] == minted, "the stamp must be the single minter's value at write time"
    # The persisted artifact still loads: the new key is ignored by the reader.
    assert av.load_validation_report(analysis).checks == []
