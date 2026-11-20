"""A ``system_config_yaml`` row is compared with the other rows of its compile target on the setup-shared fields.

The measured defect (change 1118-tk6, earlier reviewer finding F3): a master row plus a ``system_config_yaml`` row
that differed only in ``DEM_fullres`` passed preflight with one compile target. The target's setup reads
``DEM_fullres`` once, from the first row, so the other row's DEM was discarded without a word. The overlay arm of
``_validate_per_member_system_configs`` already refused that divergence between rows with no YAML; a row that
names a ``system_config_yaml`` was excluded from the comparison, and the YAML-versus-YAML arm compares only two
YAMLs, so a YAML row against a master or overlay row was compared by nobody.

What this file decides, for every field in ``_SYSTEM_SETUP_SHARED_FIELDS`` and not for the one that was measured: a
YAML row that shares a compile target with a master row or an overlay row and differs from it in a setup-shared field
is refused with the overlay arm's wording, whichever of the two rows comes first; a YAML row that differs only in a
member-reaching field, or in a compile-key field (so it builds apart), or in nothing, is not refused.

VENUE: laptop-safe. The preflight function is run over a matrix frame and real YAML files under ``tmp_path``; no solver
is compiled or run and no ``TRITONSWMM_system`` is constructed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from hhemt.config.system import system_config
from hhemt.sensitivity_analysis import _SYSTEM_COMPILE_KEY_FIELDS, _SYSTEM_SETUP_SHARED_FIELDS
from tests.test_sensitivity_system_overlay import _frame, _master, _overlay_cells, _preflight

_SHARED_FIELDS = sorted(_SYSTEM_SETUP_SHARED_FIELDS)


def _row_yaml(tmp_path: Path, master, cells: dict, stem: str) -> str:
    """Write the master's config with ``cells`` applied as a schema-valid per-row system YAML; return its path."""
    cfg = system_config.model_validate({**master.model_dump(), **cells})
    path = tmp_path / "row_yamls" / f"{stem}.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
    return str(path)


def _matrix(rows: dict[str, dict | str]) -> pd.DataFrame:
    """A matrix frame: a ``dict`` row is a master or overlay row, a ``str`` row names that ``system_config_yaml``."""
    overlay_rows = {m: spec for m, spec in rows.items() if isinstance(spec, dict)}
    df = _frame({m: overlay_rows.get(m, {}) for m in rows})
    df["system_config_yaml"] = [rows[m] if isinstance(rows[m], str) else np.nan for m in rows]
    return df


def _divergence_errors(result) -> list[str]:
    """The refusals this check writes: the overlay arm's wording, ``shares a compile target with member_id=``."""
    return [str(e) for e in result.errors if "shares a compile target with" in str(e)]


@pytest.mark.parametrize("yaml_first", [False, True], ids=["yaml-row-later", "yaml-row-first"])
def test_a_master_row_and_a_yaml_row_that_differ_only_in_the_dem_are_refused_naming_the_field(
    tmp_path, monkeypatch, yaml_first
):
    """Property: the measured case. A master row and a YAML row that share a compile target and differ only in
    ``DEM_fullres`` are refused, and the refusal names ``system.DEM_fullres`` and the other row.

    Input class: the two rows of the measured defect, in both orders (the first row of a group is the target's config,
    so the position is a separate input). Class: REGRESSION (red before: ``is_valid`` was True with no message).
    Reachability: ``_validate_per_member_system_configs`` is the real function and the YAML is loaded through the real
    loader. Both-states anchor: the next test is the same two rows agreeing, which stay valid, so the refusal turns on
    the DEM difference and not on the presence of a YAML row. Why it is required: the second row's DEM is discarded by
    the target's one setup, so the experiment's result is wrong with no message. Killed by: restoring the exclusion of
    YAML rows from the comparison. A second correct implementation that compares each YAML row's loaded config to the
    group's first row inside the YAML-loading loop passes, and so does one that refuses with different wording only if
    it keeps the field name and the other row's id.
    """
    master = _master(tmp_path)
    yaml_path = _row_yaml(tmp_path, master, _overlay_cells(tmp_path, master, "DEM_fullres"), "dem")
    rows = {"m1": yaml_path, "m0": {}} if yaml_first else {"m0": {}, "m1": yaml_path}

    result = _preflight(tmp_path, monkeypatch, _matrix(rows))

    errors = _divergence_errors(result)
    assert not result.is_valid
    assert len(errors) == 1, [str(e) for e in result.errors]
    assert "system.DEM_fullres" in errors[0]
    first, other = ("m1", "m0") if yaml_first else ("m0", "m1")
    assert f"member_id={other} shares a compile target with member_id={first}" in errors[0]


@pytest.mark.parametrize("name", _SHARED_FIELDS)
def test_no_setup_shared_field_can_differ_between_a_master_row_and_a_yaml_row_of_one_compile_target(
    tmp_path, monkeypatch, name
):
    """Property: for EVERY field of ``_SYSTEM_SETUP_SHARED_FIELDS``, a YAML row that differs from the master row in
    that field alone is refused by a divergence error naming the field.

    Input class: each member of the set (derived from the constant, so a field added to it is exercised without an
    edit here). Class: REGRESSION (red before on every field: no comparison reached a YAML row). Reachability: the real
    preflight over a real YAML. Both-states anchor: ``test_the_set_is_not_empty_and_a_field_outside_it_is_not_refused``
    shows a member-reaching field stays valid, so the refusal is not a refusal of every divergence. Why it is
    required: the claim is "every row of a group is compared on the setup-shared fields", a claim over the set and not
    over its one measured member. Killed by: restricting the comparison to rows with no YAML, or to one field.
    A second correct implementation that derives the set from the setup readers passes unchanged.
    """
    master = _master(tmp_path)
    yaml_path = _row_yaml(tmp_path, master, _overlay_cells(tmp_path, master, name), name)

    result = _preflight(tmp_path, monkeypatch, _matrix({"m0": {}, "m1": yaml_path}))

    errors = _divergence_errors(result)
    assert errors, f"system.{name} differs between the master row and a YAML row and nothing refused it"
    assert any(f"system.{name} " in message for message in errors), errors


def test_a_yaml_row_that_differs_from_an_overlay_row_in_a_setup_shared_field_is_refused(tmp_path, monkeypatch):
    """Property: the overlay row is the other kind of non-YAML row; a YAML row is compared with it too.

    Input class: an overlay row (``system.SWMM_model`` set, a member-reaching field) and a YAML row that also moves
    ``DEM_fullres``. Class: REGRESSION. Both-states anchor: the overlay row's own SWMM overlay is not what is refused
    (the first test of the class below). Killed by: building the comparison map from master rows only. A second
    correct implementation that compares through the builder's own grouping passes.
    """
    master = _master(tmp_path)
    swmm = _overlay_cells(tmp_path, master, "SWMM_model")
    yaml_path = _row_yaml(tmp_path, master, _overlay_cells(tmp_path, master, "DEM_fullres"), "dem")

    result = _preflight(tmp_path, monkeypatch, _matrix({"m0": swmm, "m1": yaml_path}))

    errors = _divergence_errors(result)
    assert len(errors) == 1, [str(e) for e in result.errors]
    assert "system.DEM_fullres" in errors[0] and "system.SWMM_model" not in errors[0]


def test_the_set_is_not_empty_and_a_field_outside_it_is_not_refused(tmp_path, monkeypatch):
    """Control for the class above: the divergence census reads nonzero, and a member-reaching field is not refused.

    Class: NEW CAPABILITY control. Unit: field names of ``_SYSTEM_SETUP_SHARED_FIELDS``. A YAML row that differs only
    in ``SWMM_model`` (reaches its member through the member's own system) shares the master's target and must stay
    valid; the same two rows over ``DEM_fullres`` must not. Killed by: a comparison over every field, which would
    refuse the first and pass the second. A second correct implementation that compares only the set passes.
    """
    master = _master(tmp_path)
    swmm_yaml = _row_yaml(tmp_path, master, _overlay_cells(tmp_path, master, "SWMM_model"), "swmm")

    result = _preflight(tmp_path, monkeypatch, _matrix({"m0": {}, "m1": swmm_yaml}))

    assert len(_SHARED_FIELDS) > 10
    assert "SWMM_model" not in _SYSTEM_SETUP_SHARED_FIELDS
    assert result.is_valid, [str(e) for e in result.errors]
    assert _divergence_errors(result) == []


def test_a_yaml_row_that_agrees_with_the_master_row_is_valid(tmp_path, monkeypatch):
    """Rows that do not diverge keep their verdict: a YAML row identical to the master config is not refused.

    Class: REGRESSION guard (green before and after, so it holds the unchanged behaviour the entry asks for).
    Both-states anchor: the measured case above refuses the same pair once a field differs. Killed by: refusing every
    YAML row that shares a target with a master row. A second correct implementation passes.
    """
    master = _master(tmp_path)
    same = _row_yaml(tmp_path, master, {}, "same")

    result = _preflight(tmp_path, monkeypatch, _matrix({"m0": {}, "m1": same}))

    assert result.is_valid, [str(e) for e in result.errors]


@pytest.mark.parametrize("name", sorted(_SYSTEM_COMPILE_KEY_FIELDS))
def test_a_yaml_row_naming_a_different_build_is_not_compared_on_the_setup_shared_fields(tmp_path, monkeypatch, name):
    """Property: rows whose compile key differs are two targets, so a setup-shared difference between them is honoured.

    Input class: every compile-key field, with the YAML row also moving ``DEM_fullres``. Class: REGRESSION guard
    (valid before and after). Both-states anchor: the same pair without the key difference is refused by the
    measured-case test. Why it is required: the check refuses a discard, and rows that build apart discard nothing.
    Killed by: comparing every YAML row to the master row whatever their keys. A second correct implementation that
    groups through ``_system_compile_key`` passes.
    """
    master = _master(tmp_path)
    cells = {**_overlay_cells(tmp_path, master, "DEM_fullres"), **_overlay_cells(tmp_path, master, name)}
    yaml_path = _row_yaml(tmp_path, master, cells, f"split_{name}")

    result = _preflight(tmp_path, monkeypatch, _matrix({"m0": {}, "m1": yaml_path}))

    assert _divergence_errors(result) == [], [str(e) for e in result.errors]
