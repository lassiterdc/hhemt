from pathlib import Path

import pytest
from pydantic import ValidationError

from hhemt.config.analysis import analysis_config
from hhemt.config.loaders import load_system_config_from_dict
from hhemt.config.synthetic_experiment import synthetic_experiment_config
from hhemt.config.system import system_config


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test")
    return path


def _minimal_system_config_dict(tmp_path: Path) -> dict:
    system_dir = tmp_path / "system"
    system_dir.mkdir(
        parents=True, exist_ok=True
    )  # output root must exist for model_dump()->model_validate() round-trips
    return {
        "system_directory": str(system_dir),
        "watershed_gis_polygon": str(_touch(tmp_path / "inputs" / "watershed.shp")),
        "DEM_fullres": str(_touch(tmp_path / "inputs" / "dem.tif")),
        "SWMM_hydraulics": str(_touch(tmp_path / "inputs" / "hydraulics.inp")),
        "TRITONSWMM_software_directory": str(tmp_path / "triton"),
        "TRITONSWMM_git_URL": "https://example.com/triton.git",
        "SWMM_git_URL": "https://example.com/swmm.git",
        "triton_swmm_configuration_template": str(_touch(tmp_path / "inputs" / "TRITONSWMM.cfg")),
        "toggle_use_swmm_for_hydrology": False,
        "toggle_use_constant_mannings": True,
        "toggle_triton_model": False,
        "toggle_tritonswmm_model": True,
        "toggle_swmm_model": False,
        "target_dem_resolution": 5.0,
        "constant_mannings": 0.05,
        "crs": {"horizontal_epsg": 4326},
    }


def _minimal_analysis_config_dict(tmp_path: Path) -> dict:
    return {
        "analysis_id": "analysis_01",
        "weather_event_indices": ["event_id"],
        "weather_timeseries": str(_touch(tmp_path / "inputs" / "weather.nc")),
        "weather_time_series_timestep_dimension_name": "timestep",
        "weather_time_series_spatial_mean_rainfall_datavar": "RG_synth",
        "rainfall_units": "mm/hr",
        "run_mode": "serial",
        "multi_sim_run_method": "local",
        "toggle_sensitivity_analysis": False,
        "toggle_storm_tide_boundary": False,
        "weather_events_to_simulate": str(_touch(tmp_path / "inputs" / "events.csv")),
        "manhole_diameter": 1.0,
        "manhole_loss_coefficient": 0.1,
        "hydraulic_timestep_s": 1.0,
        "TRITON_reporting_timestep_s": 60,
        "open_boundaries": 1,
        "report": {},
        "clear_raw": "none",
        "force_rerun": "none",
    }


def test_analysis_config_dataset_license_default_and_frozen_vocab(tmp_path: Path):
    # R5: dataset_license defaults to CC0-1.0 when omitted; both frozen values load; a third is rejected.
    cfg = _minimal_analysis_config_dict(tmp_path)
    assert "dataset_license" not in cfg  # optional field
    assert analysis_config.model_validate(cfg).dataset_license == "CC0-1.0"
    cfg["dataset_license"] = "CC-BY-NC-4.0"
    assert analysis_config.model_validate(cfg).dataset_license == "CC-BY-NC-4.0"
    cfg["dataset_license"] = "MIT"  # not in the frozen 2-entry SPDX vocab
    with pytest.raises(ValidationError):
        analysis_config.model_validate(cfg)


def test_system_config_forbids_unknown_keys(tmp_path: Path):
    cfg = _minimal_system_config_dict(tmp_path)
    cfg["unexpected_extra"] = "should fail"

    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_system_config_from_dict(cfg)


def test_system_config_explicit_toggle_dependency(tmp_path: Path):
    cfg = _minimal_system_config_dict(tmp_path)
    cfg["toggle_use_constant_mannings"] = True
    cfg["constant_mannings"] = None

    with pytest.raises(ValidationError, match="constant_mannings"):
        system_config.model_validate(cfg)


def test_analysis_config_explicit_toggle_dependency(tmp_path: Path):
    cfg = _minimal_analysis_config_dict(tmp_path)
    cfg["toggle_storm_tide_boundary"] = True

    with pytest.raises(ValidationError, match="storm_tide_boundary_line_gis"):
        analysis_config.model_validate(cfg)


# ---------------------------------------------------------------------------
# Phase 1 — report_config schema validation
# ---------------------------------------------------------------------------


def _write_report_yaml(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


def test_report_config_loads_default(tmp_path: Path):
    from hhemt.config.loaders import yaml_to_model
    from hhemt.config.report import report_config

    yaml_path = _write_report_yaml(
        tmp_path / "report.yaml",
        "figure_defaults:\n  dpi: 120\n",
    )
    cfg = yaml_to_model(yaml_path, report_config)
    assert cfg.figure_defaults.dpi == 120
    assert cfg.sensitivity is None  # F-I-7: default is None


def test_report_config_rejects_unknown_field(tmp_path: Path):
    """Flag 7 — `extra='forbid'` regression test."""
    from hhemt.config.loaders import yaml_to_model
    from hhemt.config.report import report_config

    yaml_path = _write_report_yaml(
        tmp_path / "report.yaml",
        "system-map:\n  target_epsg: 4326\n",  # hyphen, not underscore
    )
    with pytest.raises(ValidationError, match="extra_forbidden"):
        yaml_to_model(yaml_path, report_config)


def test_report_config_sensitivity_missing_independent_vars(tmp_path: Path):
    from hhemt.config.loaders import yaml_to_model
    from hhemt.config.report import report_config

    yaml_path = _write_report_yaml(
        tmp_path / "report.yaml",
        "sensitivity:\n  mode: benchmarking\n",  # missing independent_vars
    )
    with pytest.raises(ValidationError, match="independent_vars"):
        yaml_to_model(yaml_path, report_config)


def test_validate_sensitivity_independent_vars_missing_columns(tmp_path: Path):
    import pandas as pd

    from hhemt.config.report import (
        SensitivityReportConfig,
        report_config,
        validate_sensitivity_independent_vars,
    )
    from hhemt.exceptions import ConfigurationError

    csv_path = tmp_path / "member.csv"
    pd.DataFrame({"n_omp_threads": [1, 2], "run_mode": ["serial", "parallel"]}).to_csv(csv_path, index=False)
    cfg = report_config(sensitivity=SensitivityReportConfig(independent_vars=["n_omp_threads", "missing_col"]))
    with pytest.raises(ConfigurationError) as exc:
        validate_sensitivity_independent_vars(cfg, csv_path)
    assert "missing_col" in str(exc.value)


def test_validate_sensitivity_independent_vars_charset(tmp_path: Path):
    """Flag 17 — Snakemake-safe charset validation."""
    import pandas as pd

    from hhemt.config.report import (
        SensitivityReportConfig,
        report_config,
        validate_sensitivity_independent_vars,
    )
    from hhemt.exceptions import ConfigurationError

    csv_path = tmp_path / "member.csv"
    pd.DataFrame({"bad name": [1, 2]}).to_csv(csv_path, index=False)
    cfg = report_config(sensitivity=SensitivityReportConfig(independent_vars=["bad name"]))
    with pytest.raises(ConfigurationError, match="charset"):
        validate_sensitivity_independent_vars(cfg, csv_path)


def test_validate_sensitivity_fails_when_block_missing_but_csv_present(tmp_path: Path):
    """F-I-6 — sensitivity CSV present with no sensitivity block raises."""
    from hhemt.config.report import (
        report_config,
        validate_sensitivity_independent_vars,
    )
    from hhemt.exceptions import ConfigurationError

    csv_path = tmp_path / "member.csv"
    csv_path.write_text("col\n1\n")
    cfg = report_config()  # no sensitivity block
    with pytest.raises(ConfigurationError, match="must be set"):
        validate_sensitivity_independent_vars(cfg, csv_path)


def test_validate_sensitivity_fails_when_block_present_but_no_csv(tmp_path: Path):
    from hhemt.config.report import (
        SensitivityReportConfig,
        report_config,
        validate_sensitivity_independent_vars,
    )
    from hhemt.exceptions import ConfigurationError

    cfg = report_config(sensitivity=SensitivityReportConfig(independent_vars=["col"]))
    with pytest.raises(ConfigurationError, match="no sensitivity CSV path"):
        validate_sensitivity_independent_vars(cfg, None)


def test_test_reference_report_scoping_passes_and_guard_intact():
    """analysis.test()'s _test/ reference scopes the inherited benchmarking report
    block (sensitivity=None, reporting_set='default'), so a non-sensitivity single
    run passes validate_active_reporting_set -- WHILE the validator's real
    sensitivity-config<->sensitivity-analysis guard still fires for an UN-scoped
    benchmarking report on a non-sensitivity run.

    This pins the _build_test_members overlay-scoping recipe (analysis.py:
    2725-2733) at the validator layer: the fix nulls report.sensitivity and resets
    report.reporting_set to the 'default' sentinel for the reference sub.
    """
    from hhemt.config.report import (
        SensitivityReportConfig,
        report_config,
        validate_active_reporting_set,
    )
    from hhemt.exceptions import ConfigurationError

    # Pre-fix state: a member cfg inherits the sensitivity master's report
    # block verbatim onto a non-sensitivity reference (explicit benchmarking-validator
    # reporting_set + a sensitivity: benchmarking block). The guard MUST still fire
    # here -- Option A does NOT mask it.
    unscoped = report_config(
        reporting_set="benchmarking",
        sensitivity=SensitivityReportConfig(independent_vars=["n_devices"]),
    )
    # MOVED REFERENT (S16): this match= string was written against
    # validate_sensitivity_independent_vars. The S16 shape check in
    # validate_active_reporting_set now PRE-EMPTS that call, so the literal is
    # produced by the shape branch instead. The bytes are unchanged and the
    # function under assertion is not -- recorded so the guard is not read as
    # still testing what it originally tested.
    with pytest.raises(ConfigurationError, match="not a sensitivity analysis"):
        validate_active_reporting_set(unscoped, is_sensitivity=False, sensitivity_csv_path=None)

    # Post-fix state: the overlay-scoping recipe nulls sensitivity and resets
    # reporting_set to the 'default' sentinel. A non-sensitivity reference now
    # resolves to the standard set and passes validation.
    scoped = report_config.model_validate({**unscoped.model_dump(), "sensitivity": None, "reporting_set": "default"})
    resolved = validate_active_reporting_set(scoped, is_sensitivity=False, sensitivity_csv_path=None)
    assert resolved == "default"


# ---------------------------------------------------------------------------
# Phase 1 (P1a) — named-reporting-sets: registry + reporting_set selector + D4
# ---------------------------------------------------------------------------


def test_reporting_sets_registry_imports_cleanly():
    """Registry smoke (no import cycle): REPORTING_SETS holds exactly the
    shipped sets and benchmarking carries its run-entry validator key."""
    from hhemt.report_renderers._reporting_sets import (
        REPORTING_SETS,
        get_reporting_set,
    )

    # Exact-set equality (NOT a subset check) so an accidental registry member
    # still fails. `combined` (PIP-1 cross-experiment), `compute-sensitivity`
    # (PIP-2 / R11 in-report EDA), `dem-resolution` (D13, the four-figure
    # DEM-resolution EDA family) and `b4b` (Phase 4, the raw-output byte-for-byte
    # comparison family) are shipped, exercised sets; the assertion simply trailed
    # their landing.
    assert set(REPORTING_SETS) == {
        "default",
        "benchmarking",
        "sensitivity",
        "combined",
        "compute-sensitivity",
        "dem-resolution",
        "b4b",
    }
    assert get_reporting_set("benchmarking").validator_key == "benchmarking"
    assert get_reporting_set("default").validator_key == "none"


def test_reporting_set_field_defaults_to_default():
    from hhemt.config.report import report_config

    assert report_config().reporting_set == "default"


def test_legacy_mode_key_rewritten_with_deprecation_warning():
    """R3 — a pre-conversion report_config.yaml carrying
    sensitivity.mode: benchmarking loads with a DeprecationWarning and the
    legacy `mode` key dropped (no extra_forbidden), independent_vars retained."""
    from hhemt.config.report import report_config

    # Anchored on the behaviour both wordings share, NOT on the model-vs-YAML naming:
    # 5ada991b deliberately renamed `report_config.` -> `report.` in user-facing
    # messages, and pinning that naming here is what made this assertion stale. The
    # naming itself is guarded separately below, where a rename fails loudly instead
    # of failing as a missing DeprecationWarning.
    with pytest.warns(DeprecationWarning, match=r"sensitivity\.mode is retired") as rec:
        cfg = report_config.model_validate({"sensitivity": {"mode": "benchmarking", "independent_vars": ["x"]}})
    # The message must name the YAML key a user edits, not the model's qualified name.
    # `any(...)` rather than `rec[0]`: pytest.warns(match=) searches every recorded
    # warning, but `rec` is ordered by EMISSION across all categories, so indexing [0]
    # would make this assertion depend on an ordering nothing in the code guarantees.
    assert any("report.sensitivity.mode is retired" in str(w.message) for w in rec)
    assert cfg.sensitivity is not None
    assert cfg.sensitivity.independent_vars == ["x"]
    assert not hasattr(cfg.sensitivity, "mode")


def test_resolve_active_reporting_set_name_sentinel_resolution():
    """The 'default' sentinel resolves to 'benchmarking' for sensitivity
    analyses and to the standard 'default' set otherwise (CSV-free)."""
    from hhemt.config.report import (
        report_config,
        resolve_active_reporting_set_name,
    )

    cfg = report_config()  # reporting_set == "default"
    assert resolve_active_reporting_set_name(cfg, is_sensitivity=False) == "default"
    assert resolve_active_reporting_set_name(cfg, is_sensitivity=True) == "benchmarking"


def test_resolve_active_reporting_set_name_explicit_value_taken_verbatim():
    from hhemt.config.report import (
        report_config,
        resolve_active_reporting_set_name,
    )

    cfg = report_config(reporting_set="benchmarking")
    # Explicit value is honored regardless of is_sensitivity.
    assert resolve_active_reporting_set_name(cfg, is_sensitivity=False) == "benchmarking"


def test_resolve_active_reporting_set_name_unknown_raises():
    """R2 — an unknown reporting_set raises ConfigurationError naming the
    registered sets (CSV-free resolver; this is what the render-path fail-soft
    catches before degrading to 'default')."""
    from hhemt.config.report import (
        report_config,
        resolve_active_reporting_set_name,
    )
    from hhemt.exceptions import ConfigurationError

    cfg = report_config(reporting_set="does_not_exist")
    with pytest.raises(ConfigurationError) as exc:
        resolve_active_reporting_set_name(cfg, is_sensitivity=False)
    msg = str(exc.value)
    assert "does_not_exist" in msg
    assert "benchmarking" in msg and "default" in msg  # registered sets named


# --- resolve_reporting_set_name: the shared name-taking helper (Plan Phase 5 / D14) ---
#
# resolve_active_reporting_set_name above is now a thin config-object wrapper over
# this helper. The helper exists so raw-dict callers -- specifically the bundle
# Snakefile harvest, which holds a yaml.safe_load'd cfg_analysis and no
# report_config -- share ONE implementation of the sentinel-plus-validation rule
# instead of reimplementing the sentinel half and silently dropping validation.


def test_resolve_reporting_set_name_sentinel_matches_config_object_path():
    """The helper and its config-object wrapper agree on the sentinel."""
    from hhemt.config.report import (
        report_config,
        resolve_active_reporting_set_name,
        resolve_reporting_set_name,
    )

    cfg = report_config()  # reporting_set == "default"
    for is_sensitivity in (False, True):
        assert resolve_reporting_set_name("default", is_sensitivity=is_sensitivity) == (
            resolve_active_reporting_set_name(cfg, is_sensitivity=is_sensitivity)
        )


def test_resolve_reporting_set_name_treats_none_and_empty_as_the_sentinel():
    """An absent report.reporting_set key resolves like the explicit sentinel.

    The bundle harvest passes `_report.get("reporting_set")`, which is None when
    the key is absent -- the common case for a bundle whose source run took the
    default set.
    """
    from hhemt.config.report import resolve_reporting_set_name

    for absent in (None, ""):
        assert resolve_reporting_set_name(absent, is_sensitivity=True) == "benchmarking"
        assert resolve_reporting_set_name(absent, is_sensitivity=False) == "default"


def test_resolve_reporting_set_name_takes_non_default_verbatim():
    """A non-sentinel name is honored regardless of is_sensitivity."""
    from hhemt.config.report import resolve_reporting_set_name

    for is_sensitivity in (False, True):
        assert resolve_reporting_set_name("dem-resolution", is_sensitivity=is_sensitivity) == "dem-resolution"


def test_resolve_reporting_set_name_unknown_raises_naming_the_requested_value():
    """The VALIDATION half -- the property the bundle path had lost.

    The message must name the value the CALLER supplied, not just the resolved
    one, so a typo is greppable in the error.
    """
    from hhemt.config.report import resolve_reporting_set_name
    from hhemt.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError) as exc:
        resolve_reporting_set_name("dem-resolutoin", is_sensitivity=True)
    msg = str(exc.value)
    assert "reporting_set" in msg
    assert "dem-resolutoin" in msg
    assert "dem-resolution" in msg  # registered sets named


def test_validate_active_reporting_set_unknown_raises():
    """R2 — validate_active_reporting_set surfaces the same unknown-set error
    at run-entry."""
    from hhemt.config.report import (
        report_config,
        validate_active_reporting_set,
    )
    from hhemt.exceptions import ConfigurationError

    cfg = report_config(reporting_set="nope")
    with pytest.raises(ConfigurationError, match="nope"):
        validate_active_reporting_set(cfg, is_sensitivity=False, sensitivity_csv_path=None)


def test_validate_active_reporting_set_benchmarking_delegates_csv(tmp_path: Path):
    """The 'benchmarking' set's validator_key routes run-entry validation to
    validate_sensitivity_independent_vars — a missing CSV column raises."""
    import pandas as pd

    from hhemt.config.report import (
        SensitivityReportConfig,
        report_config,
        validate_active_reporting_set,
    )
    from hhemt.exceptions import ConfigurationError

    csv_path = tmp_path / "member.csv"
    pd.DataFrame({"n_omp_threads": [1, 2]}).to_csv(csv_path, index=False)
    cfg = report_config(sensitivity=SensitivityReportConfig(independent_vars=["n_omp_threads", "missing_col"]))
    # reporting_set "default" + is_sensitivity True -> "benchmarking" -> CSV check.
    with pytest.raises(ConfigurationError, match="missing_col"):
        validate_active_reporting_set(
            cfg,
            is_sensitivity=True,
            sensitivity_csv_path=csv_path,
            varied_axes=frozenset({"n_omp_threads"}),
        )


def test_validate_active_reporting_set_returns_resolved_name(tmp_path: Path):
    """Happy path: resolves to 'benchmarking' for a sensitivity analysis whose
    independent_vars all match the CSV, and returns that name."""
    import pandas as pd

    from hhemt.config.report import (
        SensitivityReportConfig,
        report_config,
        validate_active_reporting_set,
    )

    csv_path = tmp_path / "member.csv"
    pd.DataFrame({"n_omp_threads": [1, 2]}).to_csv(csv_path, index=False)
    cfg = report_config(sensitivity=SensitivityReportConfig(independent_vars=["n_omp_threads"]))
    name = validate_active_reporting_set(
        cfg,
        is_sensitivity=True,
        sensitivity_csv_path=csv_path,
        varied_axes=frozenset({"n_omp_threads"}),
    )
    assert name == "benchmarking"


def test_required_axes_derivation_matches_template_wildcards():
    """Every set's `required_axes` is EMPTY iff its selection carries no axis-wildcarded
    template. `required_axes` is derivable from the registry's own templates but stored
    by hand, so nothing else checks the two agree.

    SCOPE LIMIT, measured and recorded rather than implied: this is a biconditional on
    EMPTINESS only. It catches an `()` that silently becomes false when a set gains an
    axis-plotting figure. It catches NOTHING about a wrong NON-empty value -- swap
    `dem-resolution`'s `(("target_dem_resolution",),)` for the compute-config tuple and
    this test still passes, because both are non-empty. `dem-resolution` is in fact the
    set that proves the limit: its only axis wildcard is `independent_var`, inherited
    with `sensitivity_benchmarking`, while the axis it requires is a system-config field
    that no template is wildcarded on.

    The failing direction is verified, not assumed: three violating registries built with
    `dataclasses.replace` (a frozen dataclass blocks mutation, not derivation) each drive
    this assertion to raise -- including a `sensitivity` set given the benchmarking axis
    figure while keeping its `()`, which is the exact regression this test exists to catch.
    """
    from hhemt.report_renderers._reporting_sets import REPORTING_SETS

    # member_id / event_id index the (member, event) product, not a sensitivity axis.
    non_axis = {"member_id", "event_id"}
    for name, rset in REPORTING_SETS.items():
        wildcards = {w for sel in rset.renderer_selection for tmpl in sel.rule_spec_template for w in tmpl.wildcards}
        has_axis_figure = bool(wildcards - non_axis)
        assert bool(rset.required_axes) == has_axis_figure, (
            f"reporting set {name!r}: required_axes={rset.required_axes!r} but its templates "
            f"carry axis wildcards {sorted(wildcards - non_axis)!r}. A set with an "
            f"axis-plotted figure must declare an axis requirement, and a set without one "
            f"must declare the empty tuple."
        )


# ---------------------------------------------------------------------------
# Phase 1 (P1a) — _react_surgery category_order threading (FQ2)
# ---------------------------------------------------------------------------


def test_react_surgery_order_js_none_default_is_byte_identical():
    """FQ2 byte-identity: the None-default (historical) ORDER literal is exactly
    the pre-refactor hardcoded comparator dict, so non-passing callers see no
    change in the rendered HTML."""
    from hhemt.report_renderers._react_surgery import (
        _DEFAULT_CATEGORY_ORDER,
        _order_js,
    )

    expected = (
        '{"Workflow Status": 1, "Errors and Warnings": 2, "Key Results": 3, '
        '"System Information": 4, "Simulation Health (placeholder)": 5, '
        '"Per Simulation Results": 6}'
    )
    assert _order_js(_DEFAULT_CATEGORY_ORDER) == expected


def test_react_surgery_threads_config_driven_category_order():
    """A config-driven category_order produces a 1-indexed ORDER literal in the
    surgered comparator (not the alphabetical fallback)."""
    from hhemt.report_renderers._react_surgery import (
        apply_post_process_surgery,
    )

    html = "sort((a, b) => a.localeCompare(b))"
    custom_order = ["Key Results", "Workflow Status", "Errors and Warnings"]
    out = apply_post_process_surgery(html, category_order=custom_order)
    assert "a.localeCompare(b)" in out  # kept as the tie-breaker
    assert '"Key Results": 1' in out
    assert '"Workflow Status": 2' in out
    assert '"Errors and Warnings": 3' in out
    # The bare alphabetical comparator must be gone (config order applied).
    assert "(a, b) => a.localeCompare(b)" not in out


def test_react_surgery_none_default_matches_hardcoded_comparator():
    """Render regression: with category_order=None the surgered comparator is
    byte-identical to the historical hardcoded comparator (the standard sidebar
    order). This is the render-path observable for the default set."""
    from hhemt.report_renderers._react_surgery import (
        apply_post_process_surgery,
    )

    html = "sort((a, b) => a.localeCompare(b))"
    out = apply_post_process_surgery(html)  # no category_order -> historical default
    assert (
        '(a, b) => {const ORDER = {"Workflow Status": 1, "Errors and Warnings": 2, '
        '"Key Results": 3, "System Information": 4, '
        '"Simulation Health (placeholder)": 5, "Per Simulation Results": 6}; '
        "return (ORDER[a] ?? 99) - (ORDER[b] ?? 99) || a.localeCompare(b);}"
    ) in out


def test_report_artifacts_not_in_globus_exclude_patterns():
    """Flag 14 — R12 automated Globus-exclude audit."""
    from hhemt.config.globus import DEFAULT_EXCLUDE_PATTERNS

    for bad in ("plots", "report", "analysis_report.html"):
        assert not any(bad in p for p in DEFAULT_EXCLUDE_PATTERNS), (
            f"{bad!r} would match an entry of DEFAULT_EXCLUDE_PATTERNS "
            f"{DEFAULT_EXCLUDE_PATTERNS}; R12 requires report artifacts "
            "to be included in the default Globus transfer."
        )


def test_pydantic_config_field_names_are_snakemake_wildcard_safe():
    """Phase 1 R9 — every system_config and analysis_config field name MUST match
    `^[A-Za-z0-9_.]+$` so the prefixed-column overlay mechanism (which routes
    `system.{field}` / `analysis.{field}` cells into Snakemake wildcards) cannot
    silently break on a future field addition with a hyphen or other unsafe char.

    Developer-facing assertion against the codebase's Pydantic model schemas.
    Failure indicates a toolkit author has introduced a bad field name.
    """
    import re

    charset = re.compile(r"^[A-Za-z0-9_.]+$")
    offenders: list[str] = []
    for model_name, model in [
        ("system_config", system_config),
        ("analysis_config", analysis_config),
    ]:
        for field_name in model.model_fields:
            if not charset.match(field_name):
                offenders.append(f"{model_name}.{field_name}")
    assert not offenders, (
        f"Pydantic field names outside Snakemake-wildcard-safe charset "
        f"^[A-Za-z0-9_.]+$ (toolkit author defect, not a user-config issue): "
        f"{offenders}."
    )


# ---------------------------------------------------------------------------
# Phase 1 — cleanup-rerun-delete-redesign: clear_raw + force_rerun fields
# ---------------------------------------------------------------------------


def test_clear_raw_defaults_to_none(tmp_path: Path):
    """Loading an analysis_config without `clear_raw` defaults to 'none' (strict-safe)."""
    cfg = _minimal_analysis_config_dict(tmp_path)
    del cfg["clear_raw"]
    loaded = analysis_config.model_validate(cfg)
    assert loaded.clear_raw == "none"


def test_force_rerun_defaults_to_none(tmp_path: Path):
    """Loading an analysis_config without `force_rerun` defaults to 'none' (strict-safe)."""
    cfg = _minimal_analysis_config_dict(tmp_path)
    del cfg["force_rerun"]
    loaded = analysis_config.model_validate(cfg)
    assert loaded.force_rerun == "none"


@pytest.mark.parametrize(
    "value",
    ["all", "none", ["tritonswmm"], ["triton", "swmm"], ["tritonswmm", "triton", "swmm"]],
)
def test_clear_raw_accepts_valid_shapes(value, tmp_path: Path):
    cfg = _minimal_analysis_config_dict(tmp_path)
    cfg["clear_raw"] = value
    result = analysis_config.model_validate(cfg)
    assert result.clear_raw == value


@pytest.mark.parametrize(
    "value",
    [
        ["all"],  # sentinel inside list
        ["none"],  # sentinel inside list
        [],  # empty list
        ["tritonswmm", "tritonswmm"],  # duplicates
        "unknown",  # not in Literal arm
    ],
)
def test_clear_raw_rejects_invalid(value, tmp_path: Path):
    cfg = _minimal_analysis_config_dict(tmp_path)
    cfg["clear_raw"] = value
    with pytest.raises(ValidationError):
        analysis_config.model_validate(cfg)


def test_force_rerun_accepts_all_none_sentinels(tmp_path: Path):
    cfg = _minimal_analysis_config_dict(tmp_path)
    # force_rerun is a ForceRerunSpec: the sentinel round-trips on the SUBJECT axis, and
    # the stage floor defaults to "simulate" (the historical meaning of a bare sentinel).
    cfg["force_rerun"] = "all"
    spec = analysis_config.model_validate(cfg).force_rerun
    assert spec.subject == "all"
    assert spec.stage == "simulate"
    cfg["force_rerun"] = "none"
    spec = analysis_config.model_validate(cfg).force_rerun
    assert spec.subject == "none"
    assert spec.stage == "simulate"


def test_force_rerun_event_iloc_accepts_list(tmp_path: Path):
    cfg = _minimal_analysis_config_dict(tmp_path)
    cfg["toggle_sensitivity_analysis"] = False
    cfg["force_rerun"] = {"event_iloc": [3, 7]}
    result = analysis_config.model_validate(cfg)
    # Same claim against the field's new representation: the subject round-trips unchanged
    # and a legacy dict form defaults to the simulate floor.
    assert result.force_rerun.subject == {"event_iloc": [3, 7]}
    assert result.force_rerun.stage == "simulate"


def test_force_rerun_member_id_requires_sensitivity_toggle(tmp_path: Path):
    cfg = _minimal_analysis_config_dict(tmp_path)
    cfg["toggle_sensitivity_analysis"] = False
    cfg["force_rerun"] = {"sa_id": ["0", "5"]}
    with pytest.raises(ValidationError, match="toggle_sensitivity_analysis=True"):
        analysis_config.model_validate(cfg)


def test_force_rerun_event_iloc_requires_no_sensitivity(tmp_path: Path):
    cfg = _minimal_analysis_config_dict(tmp_path)
    cfg["toggle_sensitivity_analysis"] = True
    cfg["sensitivity_analysis"] = str(_touch(tmp_path / "inputs" / "sensitivity.csv"))
    cfg["force_rerun"] = {"event_iloc": [3, 7]}
    with pytest.raises(ValidationError, match="toggle_sensitivity_analysis=False"):
        analysis_config.model_validate(cfg)


@pytest.mark.parametrize(
    "value,match",
    [
        ({"sa_id": [], "event_iloc": [1]}, "exactly one key"),
        ({"bad_key": [1]}, "'sa_id' or 'event_iloc'"),
        ({"sa_id": []}, "non-empty list"),
        ({"sa_id": ["0", "0"]}, "duplicates"),
        ({"sa_id": ["bad id with spaces"]}, r"\^\[A-Za-z0-9_\.\]\+\$"),
    ],
)
def test_force_rerun_rejects_invalid_dict_shapes(value, match, tmp_path: Path):
    cfg = _minimal_analysis_config_dict(tmp_path)
    # Use sensitivity-on so member_id paths don't trip on the cross-field rule
    # before the per-field validator gets a chance to fire.
    cfg["toggle_sensitivity_analysis"] = True
    cfg["sensitivity_analysis"] = str(_touch(tmp_path / "inputs" / "sensitivity.csv"))
    cfg["force_rerun"] = value
    with pytest.raises(ValidationError, match=match):
        analysis_config.model_validate(cfg)


@pytest.mark.parametrize(
    "value,should_raise",
    [
        (60, False),
        (480, False),
        (10080, False),
        (59, True),
        (10081, True),
    ],
)
def test_hpc_max_wait_for_inflight_min_field_bounds(value, should_raise, tmp_path: Path):
    cfg = _minimal_analysis_config_dict(tmp_path)
    cfg["hpc_max_wait_for_inflight_min"] = value
    if should_raise:
        with pytest.raises(ValidationError, match="hpc_max_wait_for_inflight_min"):
            analysis_config.model_validate(cfg)
    else:
        result = analysis_config.model_validate(cfg)
        assert result.hpc_max_wait_for_inflight_min == value


# ---------------------------------------------------------------------------
# Phase 2 — process_append_batch_memory_budget_mb resolver (job-RAM budget)
# ---------------------------------------------------------------------------


def test_process_append_batch_budget_resolves_from_job_ram(tmp_path: Path, monkeypatch):
    """None resolves to round(0.35 * hpc_mem_allocation_for_sim_output_processing_mb).

    The cgroup reader is monkeypatched to None so the assertion is deterministic
    regardless of the host's actual cgroup ceiling (which would only clamp lower).
    """
    import hhemt.config.analysis as cfg_mod

    monkeypatch.setattr(cfg_mod, "_read_cgroup_memory_limit_mib", lambda: None)
    cfg = _minimal_analysis_config_dict(tmp_path)
    cfg["hpc_mem_allocation_for_sim_output_processing_mb"] = 12000
    # process_append_batch_memory_budget_mb left unset (None) -> resolver fills it.
    model = analysis_config.model_validate(cfg)
    assert model.process_append_batch_memory_budget_mb == round(0.35 * 12000)  # 4200


def test_process_append_batch_budget_ceiling_raises(tmp_path: Path):
    """A hand-set value exceeding 0.5 * job_RAM raises (R4 guard 1)."""
    cfg = _minimal_analysis_config_dict(tmp_path)
    cfg["hpc_mem_allocation_for_sim_output_processing_mb"] = 12000
    cfg["process_append_batch_memory_budget_mb"] = 7000  # > 0.5 * 12000 = 6000
    with pytest.raises(ValidationError, match="exceeds 0.5"):
        analysis_config.model_validate(cfg)


# ---------------------------------------------------------------------------
# Phase 2 — static_plot_configs layer-4 field (V-8 / R-7)
# ---------------------------------------------------------------------------


def test_static_plot_configs_defaults_to_empty(tmp_path: Path):
    """An analysis_config without `static_plot_configs` loads with the
    strict-safe empty default (yamls written before the field load cleanly)."""
    cfg = _minimal_analysis_config_dict(tmp_path)
    loaded = analysis_config.model_validate(cfg)
    assert loaded.static_plot_configs == []


def test_static_plot_configs_existent_list_validates(tmp_path: Path):
    """A list of existent paths validates and is normalized to Path objects."""
    cfg = _minimal_analysis_config_dict(tmp_path)
    p1 = _touch(tmp_path / "plots" / "plot_a.yaml")
    p2 = _touch(tmp_path / "plots" / "plot_b.yaml")
    cfg["static_plot_configs"] = [str(p1), str(p2)]
    loaded = analysis_config.model_validate(cfg)
    assert loaded.static_plot_configs == [Path(p1), Path(p2)]
    assert all(isinstance(p, Path) for p in loaded.static_plot_configs)


def test_static_plot_configs_nonexistent_path_raises(tmp_path: Path):
    """A list element that does not exist raises at config-load (R-7 / V-8) —
    the dedicated list-aware validator, since the base `*` validator only
    existence-checks scalar Path fields."""
    cfg = _minimal_analysis_config_dict(tmp_path)
    existent = _touch(tmp_path / "plots" / "plot_a.yaml")
    missing = tmp_path / "plots" / "does_not_exist.yaml"
    cfg["static_plot_configs"] = [str(existent), str(missing)]
    with pytest.raises(ValidationError, match="static_plot_configs path does not exist"):
        analysis_config.model_validate(cfg)


# ---------------------------------------------------------------------------
# compile-bearing-synth-ci-tier Phase 1 — toolkit-owned-output path exemption
# ---------------------------------------------------------------------------


def test_toolkit_owned_output_dirs_exempt_from_existence_check(tmp_path: Path):
    """R1/R2: a toolkit-owned-output software-dir field validates even when its
    value is a non-existent Path (the json_schema_extra sentinel exempts it
    BEFORE the isinstance(v, Path) existence check — the overlay-revalidation
    shape); a genuine INPUT path field still fails fast on a non-existent Path."""
    cfg = _minimal_system_config_dict(tmp_path)
    # Pass the software dir as a non-existent Path (not str) so the validator's
    # isinstance(v, Path) branch is exercised. Pre-fix this raised; post-fix the
    # toolkit_owned_output sentinel exempts it.
    cfg["TRITONSWMM_software_directory"] = tmp_path / "_software" / "triton"  # absent
    cfg["SWMM_software_directory"] = (
        tmp_path / "_software" / "swmm"
    )  # absent, Optional field SET to a non-existent Path
    validated = system_config.model_validate(cfg)
    assert not validated.TRITONSWMM_software_directory.exists()
    assert (
        not validated.SWMM_software_directory.exists()
    )  # R2: Optional sentinel field is exempted when SET to an absent Path (not merely when None)

    # R2: a genuine INPUT Path field pointing at an absent file still raises.
    bad = _minimal_system_config_dict(tmp_path)
    bad["DEM_fullres"] = tmp_path / "inputs" / "does_not_exist.tif"  # absent Path
    with pytest.raises(ValidationError, match="does not exist"):
        system_config.model_validate(bad)


def test_toolkit_owned_output_exempt_under_overlay_revalidation(tmp_path: Path):
    """R3: reproduce the sensitivity overlay-revalidation round-trip that
    sensitivity_analysis.py:1971 performs — base.model_dump() (mode='python')
    yields the software dirs as PosixPath, then model_validate({**dumped,
    **overlay}) re-fires the wildcard validator against those Paths. Pre-fix this
    raised on the absent software dirs; post-fix the toolkit_owned_output sentinel
    exempts them. This is the fresh-worktree / per-row construction surface (R3),
    covered here WITHOUT the compile toolchain."""
    base = system_config.model_validate(_minimal_system_config_dict(tmp_path))
    dumped = base.model_dump()  # mode='python' -> PosixPath preserved (A2)
    dumped["TRITONSWMM_software_directory"] = tmp_path / "_software" / "triton"  # absent
    dumped["SWMM_software_directory"] = tmp_path / "_software" / "swmm"  # absent
    overlay = {"target_dem_resolution": 10.0}  # a representative overlaid cell
    revalidated = system_config.model_validate({**dumped, **overlay})
    assert revalidated.target_dem_resolution == 10.0
    assert not revalidated.TRITONSWMM_software_directory.exists()
    assert not revalidated.SWMM_software_directory.exists()


# VALIDATION - SYNTHETIC EXPERIMENT MODEL ARMS
def _synth_cfg(tmp_path: Path, **overrides) -> synthetic_experiment_config:
    """Build a synthetic_experiment_config for model_arms rejection tests.

    ``hpc_system_config_yaml`` only has to EXIST here: ``_validate_model_arms`` is a
    field validator, and pydantic runs every field validator before any
    ``mode="after"`` model validator — so a model_arms rejection short-circuits
    ``_validate_caps``, which is the only thing that would parse the YAML body.
    """
    return synthetic_experiment_config(
        hpc_system_config_yaml=_touch(tmp_path / "hpc.yaml"),
        ensemble_partition="gpu-a6000",
        setup_partition="standard",
        **overrides,
    )


def test_model_arms_rejects_empty(tmp_path: Path):
    """An empty arm set would produce zero sibling masters — no experiment at all."""
    with pytest.raises(ValidationError, match="model_arms cannot be empty"):
        _synth_cfg(tmp_path, model_arms=())


def test_model_arms_rejects_duplicates(tmp_path: Path):
    """A duplicated arm would generate two identical masters racing one output tree."""
    with pytest.raises(ValidationError, match="duplicates"):
        _synth_cfg(tmp_path, model_arms=("tritonswmm", "tritonswmm"))


def test_model_arms_rejects_unknown_arm(tmp_path: Path):
    """A typo must fail at config-load naming the field, not as a bare KeyError
    from inside _build_case (which gives the operator no field name)."""
    with pytest.raises(ValidationError, match="unknown arm"):
        _synth_cfg(tmp_path, model_arms=("tritonswmm", "tritn"))


def test_model_arms_exempt_from_pydantic_protected_namespace():
    """`model_arms` collides with pydantic's default `model_` protected namespace.

    The model relaxes the namespace for itself; dropping that relaxation
    reintroduces a UserWarning on every import of this config module. Asserted on
    the resolved config rather than by capturing a warning, because re-emitting it
    would require reloading the module and swapping the class object other tests
    already hold a reference to.
    """
    assert synthetic_experiment_config.model_config.get("protected_namespaces") == ()
    # The relaxation must not clobber the strictness guard inherited from cfgBaseModel.
    assert synthetic_experiment_config.model_config.get("extra") == "forbid"
