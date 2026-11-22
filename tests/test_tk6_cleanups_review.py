"""Reviewer tests for change 1118-tk6: the YAML-row setup-shared comparison and the two deleted sensitivity helpers.

What this file decides, beyond the coder's own file (``test_yaml_row_setup_shared_divergence.py``):

* reverting the YAML-row comparison turns the measured case, and every setup-shared field, red (mutation arm);
* a matrix with no divergence gets a byte-identical verdict before and after the change (behaviour held);
* a YAML-versus-YAML divergence is still refused, with its old refusal text unchanged;
* a YAML row's refusal carries the overlay path's wording, field and hint, byte for byte;
* a YAML row that cannot be read neither raises nor hides a divergence between two rows that can;
* no module under ``src/`` or ``scripts/`` still names one of the two deleted helpers.

VENUE: laptop-safe. No solver is compiled or run and no ``TRITONSWMM_system`` is constructed; the preflight function
runs over a matrix frame and YAML files under ``tmp_path``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import hhemt.validation as validation_mod
from hhemt.sensitivity_analysis import _SYSTEM_SETUP_SHARED_FIELDS, TRITONSWMM_sensitivity_analysis
from tests.test_sensitivity_system_overlay import _frame, _master, _overlay_cells, _preflight
from tests.test_yaml_row_setup_shared_divergence import (
    _divergence_errors,
    _matrix,
    _row_yaml,
)
from tests.test_yaml_row_setup_shared_divergence import (
    test_a_master_row_and_a_yaml_row_that_differ_only_in_the_dem_are_refused_naming_the_field as _measured_case,
)
from tests.test_yaml_row_setup_shared_divergence import (
    test_no_setup_shared_field_can_differ_between_a_master_row_and_a_yaml_row_of_one_compile_target as _every_field,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DELETED_HELPERS = ("compile_and_preprocess_all_targets", "compile_TRITON_SWMM_for_sensitivity_analysis")


def _revert_yaml_row_comparison(monkeypatch) -> None:
    """The pre-change behaviour: no YAML row's config reaches the setup-shared comparison."""
    monkeypatch.setattr(validation_mod, "_load_yaml_row_system_configs", lambda _df: {})


def _verdict(result) -> list[tuple[str, str, str | None, str | None]]:
    """Every error of a preflight result as ``(field, message, current_value, fix_hint)``, in order."""
    return [(e.field, e.message, e.current_value, e.fix_hint) for e in result.errors]


@pytest.mark.parametrize("yaml_first", [False, True], ids=["yaml-row-later", "yaml-row-first"])
def test_reverting_the_yaml_row_comparison_turns_the_measured_case_red(tmp_path, monkeypatch, yaml_first):
    """Property: the coder's measured-case test discriminates, in both row orders.

    Input class: the measured case (master row plus a YAML row differing only in ``DEM_fullres``), with the YAML row
    first and later. Class: REGRESSION over the mutation arm. Reachability: ``_load_yaml_row_system_configs`` is the
    one new source of YAML configs in the comparison; replacing it with an empty map is the pre-change input.
    Both-states anchor: the same call without the mutation is green (the coder's own run), and the next test holds
    the mutation green where nothing diverges. Why it is required: a test that stays green with the change reverted
    would certify nothing. Killed by: dropping the YAML rows from the comparison map. A second correct implementation
    that compares inside the YAML-loading loop is green under the unmutated call and is not touched by the mutation.
    """
    _revert_yaml_row_comparison(monkeypatch)

    with pytest.raises(AssertionError):
        _measured_case(tmp_path, monkeypatch, yaml_first)


@pytest.mark.parametrize("name", sorted(_SYSTEM_SETUP_SHARED_FIELDS))
def test_reverting_the_yaml_row_comparison_turns_every_setup_shared_field_red(tmp_path, monkeypatch, name):
    """Property: for every member of ``_SYSTEM_SETUP_SHARED_FIELDS`` the every-field test fails under the mutation.

    Input class: each field of the set, derived from the constant. Class: REGRESSION over the mutation arm.
    Reachability: as the test above. Both-states anchor: the unmutated every-field test is the coder's green run.
    Why it is required: an every-field test red on one field only would leave the other fields unprotected.
    Killed by: restricting the comparison to one field. A second correct implementation that derives the set from the
    setup readers keeps its own green under the unmutated call.
    """
    _revert_yaml_row_comparison(monkeypatch)

    with pytest.raises(AssertionError):
        _every_field(tmp_path, monkeypatch, name)


def _agreeing_master_overlay_and_yaml(tmp_path, master):
    """Master, a member-reaching overlay row, and two YAML rows that agree with the master on every shared field."""
    same = _row_yaml(tmp_path, master, {}, "same")
    swmm = _row_yaml(tmp_path, master, _overlay_cells(tmp_path, master, "SWMM_model"), "swmm")
    return _matrix({"m0": {}, "m1": _overlay_cells(tmp_path, master, "SWMM_model"), "m2": same, "m3": swmm})


def _yaml_rows_that_build_apart(tmp_path, master):
    """A master row and a YAML row that differs in a compile-key field and in ``DEM_fullres``: two targets."""
    cells = {
        **_overlay_cells(tmp_path, master, "DEM_fullres"),
        **_overlay_cells(tmp_path, master, "target_dem_resolution"),
    }
    return _matrix({"m0": {}, "m1": _row_yaml(tmp_path, master, cells, "apart")})


def _yaml_only_agreeing(tmp_path, master):
    """Two YAML rows with equal configs and no master or overlay row."""
    return _matrix({"m0": _row_yaml(tmp_path, master, {}, "a"), "m1": _row_yaml(tmp_path, master, {}, "b")})


def _overlay_only_divergent(tmp_path, master):
    """No YAML column values at all: an overlay-only divergence the change must leave exactly as it was."""
    df = _frame({"m0": {}, "m1": _overlay_cells(tmp_path, master, "DEM_fullres")})
    df["system_config_yaml"] = np.nan
    return df


@pytest.mark.parametrize(
    "build_matrix",
    [_agreeing_master_overlay_and_yaml, _yaml_rows_that_build_apart, _yaml_only_agreeing, _overlay_only_divergent],
    ids=["agreeing-mixed", "build-apart", "yaml-only-agreeing", "overlay-only-divergent"],
)
def test_a_matrix_with_no_yaml_row_divergence_keeps_its_verdict_byte_for_byte(tmp_path, monkeypatch, build_matrix):
    """Property: where no YAML row diverges from another row of its target, the preflight verdict is unchanged.

    Input class: four matrices, none carrying a setup-shared divergence that involves a YAML row (agreeing mixed rows,
    rows that build apart, YAML rows that agree, and an overlay-only divergence, which was refused before and after).
    Class: REGRESSION guard (green before and after). The verdict is every error's field, message, value and hint,
    in order, so a changed count, wording or order shows. Both-states anchor: the measured-case test refuses once a
    YAML row diverges. Why it is required: the entry says not to change behaviour for rows that do not diverge.
    Killed by: refusing every YAML row that shares a target with another row. A second correct implementation passes.
    """
    master = _master(tmp_path)
    df = build_matrix(tmp_path, master)
    with monkeypatch.context() as before:
        _revert_yaml_row_comparison(before)
        verdict_before = _verdict(_preflight(tmp_path, before, df))

    verdict_after = _verdict(_preflight(tmp_path, monkeypatch, df))

    assert verdict_after == verdict_before


def test_the_overlay_only_divergent_matrix_is_refused_so_the_verdict_guard_is_not_vacuous(tmp_path, monkeypatch):
    """Control for the test above: the one matrix in it that diverges is refused, so its equality compares errors.

    Class: NEW CAPABILITY control. Unit: the errors of one preflight result. Killed by: a verdict function that always
    returns an empty list, which would make every equality above true. A second correct implementation passes.
    """
    master = _master(tmp_path)

    result = _preflight(tmp_path, monkeypatch, _overlay_only_divergent(tmp_path, master))

    assert len(_divergence_errors(result)) == 1


def test_a_yaml_versus_yaml_divergence_is_still_refused_with_its_old_refusal_text_unchanged(tmp_path, monkeypatch):
    """Property: two YAML rows that differ in ``DEM_fullres`` are refused, and every refusal the change replaced stays.

    Input class: two YAML rows of one compile target and no master or overlay row (the case the YAML-versus-YAML arm
    already covered). Class: REGRESSION. Reachability: the canonical-YAML arm runs over the loaded files. Both-states
    anchor: the assertion is the pre-change verdict being a SUBSET of the post-change verdict, so it holds whether the
    new comparison also fires or not. Why it is required: the entry says this refusal still refuses as before.
    Killed by: removing or rewording the canonical-YAML arm. A second correct implementation that suppresses the
    duplicate refusal passes.
    """
    master = _master(tmp_path)
    df = _matrix(
        {
            "m0": _row_yaml(tmp_path, master, {}, "a"),
            "m1": _row_yaml(tmp_path, master, _overlay_cells(tmp_path, master, "DEM_fullres"), "b"),
        }
    )
    with monkeypatch.context() as before:
        _revert_yaml_row_comparison(before)
        verdict_before = _verdict(_preflight(tmp_path, before, df))

    result = _preflight(tmp_path, monkeypatch, df)

    assert not result.is_valid
    assert verdict_before, (
        "the pre-change run must refuse the YAML-versus-YAML divergence or this test compares nothing"
    )
    assert all(issue in _verdict(result) for issue in verdict_before)
    assert any("'DEM_fullres'" in message for _field, message, _value, _hint in verdict_before)


def test_a_yaml_row_is_refused_with_the_overlay_rows_wording_field_value_and_hint(tmp_path, monkeypatch):
    """Property: the refusal of a YAML row is byte-identical to the refusal of the overlay row with the same change.

    Input class: master plus a row that changes ``DEM_fullres`` by overlay column, and master plus a row that changes
    it by ``system_config_yaml``, the two rows holding the same member ids. Class: NEW CAPABILITY (the plausible wrong
    implementation words the YAML row's refusal differently or files it under another field). Reachability: both
    matrices reach the one comparison function. Both-states anchor: the overlay verdict is the pre-change behaviour.
    Why it is required: the entry asks for the same wording as the overlay path, and a reader fixes both by one hint.
    Killed by: a separate message template for YAML rows. A second correct implementation that routes both through one
    function passes.
    """
    master = _master(tmp_path)
    cells = _overlay_cells(tmp_path, master, "DEM_fullres")
    overlay_result = _preflight(tmp_path, monkeypatch, _matrix({"m0": {}, "m1": cells}))
    yaml_result = _preflight(
        tmp_path, monkeypatch, _matrix({"m0": {}, "m1": _row_yaml(tmp_path, master, cells, "dem")})
    )

    overlay_issues = [i for i in overlay_result.errors if "shares a compile target with" in i.message]
    yaml_issues = [i for i in yaml_result.errors if "shares a compile target with" in i.message]

    assert len(overlay_issues) == 1
    assert len(yaml_issues) == 1
    assert _verdict_of(yaml_issues) == _verdict_of(overlay_issues)


def _verdict_of(issues) -> list[tuple[str, str, str | None, str | None]]:
    """The ``(field, message, current_value, fix_hint)`` of each issue in a list."""
    return [(i.field, i.message, i.current_value, i.fix_hint) for i in issues]


@pytest.mark.parametrize("unreadable", ["missing", "not-a-config"])
def test_an_unreadable_yaml_row_neither_raises_nor_hides_a_divergence_between_readable_rows(
    tmp_path, monkeypatch, unreadable
):
    """Property: a row whose YAML is missing or fails the loader gives no comparison entry and breaks no other row.

    Input class: a master row, a readable YAML row that diverges in ``DEM_fullres``, and a third row whose YAML is
    missing or is not a system config. Class: REGRESSION. Reachability: the helper catches the loader's exception and
    the missing file by ``is_file()``. Both-states anchor: the control below drops the unreadable row and expects the
    same divergence refusal, so the refusal turns on the readable pair. Why it is required: the helper
    ``_load_yaml_row_system_configs`` wraps the loader in a bare ``except``; a loader error escaping it would turn a
    preflight report into a traceback.
    Killed by: removing the guard around ``load_system_config``. A second correct implementation that reports the file
    itself passes if it still reports the divergence.
    """
    master = _master(tmp_path)
    readable = _row_yaml(tmp_path, master, _overlay_cells(tmp_path, master, "DEM_fullres"), "dem")
    bad = tmp_path / "row_yamls" / "bad.yaml"
    if unreadable == "not-a-config":
        bad.parent.mkdir(exist_ok=True)
        bad.write_text("not_a_system_config_field: 1\n")

    result = _preflight(tmp_path, monkeypatch, _matrix({"m0": {}, "m1": readable, "m2": str(bad)}))

    assert len(_divergence_errors(result)) == 1
    assert any(str(bad) in str(e) for e in result.errors), "the unreadable file must still be reported by name"


def test_the_divergence_refusal_does_not_depend_on_an_unreadable_row_being_present(tmp_path, monkeypatch):
    """Control for the test above: without the unreadable row the same divergence is refused."""
    master = _master(tmp_path)
    readable = _row_yaml(tmp_path, master, _overlay_cells(tmp_path, master, "DEM_fullres"), "dem")

    result = _preflight(tmp_path, monkeypatch, _matrix({"m0": {}, "m1": readable}))

    assert len(_divergence_errors(result)) == 1


def _source_files_naming(names: tuple[str, ...]) -> list[str]:
    """Repo-relative paths of the text files under ``src/`` and ``scripts/`` that contain any of ``names``."""
    pattern = re.compile("|".join(re.escape(name) for name in names))
    hits = []
    for root in ("src", "scripts"):
        for path in sorted((_REPO_ROOT / root).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".txt", ".yaml", ".md"}:
                if pattern.search(path.read_text(errors="replace")):
                    hits.append(str(path.relative_to(_REPO_ROOT)))
    return hits


def test_the_census_finds_a_name_that_is_present_so_a_zero_for_the_deleted_helpers_means_absence():
    """Control for the census below: the same matcher over the same population finds a helper that exists."""
    assert _source_files_naming(("_build_unique_system_targets",)) != []


def test_no_module_under_src_or_scripts_names_a_deleted_sensitivity_compile_helper():
    """Property: no file of ``src/`` or ``scripts/`` names ``compile_and_preprocess_all_targets`` or
    ``compile_TRITON_SWMM_for_sensitivity_analysis``, in code, a comment, a docstring or the published-surface baseline.

    Input class: every ``.py``, ``.txt``, ``.yaml`` and ``.md`` file under those two roots (unit: files; matcher: the
    two names, escaped, ORed). Class: REGRESSION. Both-states anchor: the control above finds a live helper name with
    the same matcher. Why it is required: a comment that explains a decision by what a deleted function does asserts
    a fact about code that is gone, and the next reader acts on it. Killed by: leaving the comment in
    ``setup_workflow.py`` that names the helper as a remedy the message avoids. A second correct implementation that
    deletes the comment or rewrites it without the name passes.
    """
    assert _source_files_naming(_DELETED_HELPERS) == []


def test_the_sensitivity_analysis_class_no_longer_carries_either_deleted_helper():
    """Property: neither deleted name is an attribute of ``TRITONSWMM_sensitivity_analysis``.

    Input class: the two names of the entry. Class: REGRESSION. Both-states anchor: ``unique_system_targets`` assigned
    in ``__init__`` and a neighbouring method (``_build_unique_system_targets``) are still attributes, so the check
    reads the class and not an empty namespace. Killed by: restoring either helper.
    A second correct implementation that moves the helper to another class passes.
    """
    assert hasattr(TRITONSWMM_sensitivity_analysis, "_build_unique_system_targets")
    assert [name for name in _DELETED_HELPERS if hasattr(TRITONSWMM_sensitivity_analysis, name)] == []
