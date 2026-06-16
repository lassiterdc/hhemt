"""Regression tests for CaseStudyBuilder cfg_analysis write path.

Specifically guards against the bug where instantiating a UVA benchmarking
case study writes `cfg_analysis.yaml::report.sensitivity: null`, which
causes `analysis.run()` to raise `ConfigurationError`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

import TRITON_SWMM_toolkit.case_study_catalog as cat


@pytest.fixture(scope="module")
def example_data_available() -> bool:
    """Skip these tests if the Norfolk Irene example data is not cached locally.

    When TRITON_SWMM_REQUIRE_EXAMPLE_DATA=1 (set on CI runners that cache the
    example data), a load failure is re-raised as a hard error instead of a
    silent skip, so the regression guard cannot pass vacuously on CI.
    """
    try:
        cat.all_examples.norfolk_irene(download_if_exists=False)
    except Exception as exc:
        if os.environ.get("TRITON_SWMM_REQUIRE_EXAMPLE_DATA") == "1":
            raise AssertionError(
                f"Norfolk Irene example data required (TRITON_SWMM_REQUIRE_EXAMPLE_DATA=1) but load failed: {exc!r}"
            ) from exc
        pytest.skip(f"Norfolk Irene example data not available locally: {exc!r}")
    return True


def _require_software_dirs() -> None:
    """Skip when the compiled ``swmm``/``triton`` software-dir build artifacts are
    absent under the example's test_case_directory.

    The coupled and triton-only sensitivity XLSX carry ``system.*`` overlay rows;
    ``export_sensitivity_definition_csv`` revalidates each such row against the
    base ``cfg_system``, whose ``SWMM_software_directory`` / ``TRITONSWMM_software_directory``
    point at ``<test_case_directory>/swmm`` and ``/triton``. Those dirs are
    build artifacts present on a provisioned tree (CI / main) but not in a fresh
    worktree. The swmm-only variant has no overlay columns and so does not call
    this gate. Under TRITON_SWMM_REQUIRE_EXAMPLE_DATA=1 the absence is a hard
    error rather than a silent skip.
    """
    example_dir = cat.all_examples.norfolk_irene().test_case_directory
    missing = [name for name in ("swmm", "triton") if not (example_dir / name).exists()]
    if missing:
        msg = (
            f"software-dir build artifacts {missing} absent under {example_dir}; "
            "coupled/triton-only sensitivity rows cannot revalidate system.* overlays."
        )
        if os.environ.get("TRITON_SWMM_REQUIRE_EXAMPLE_DATA") == "1":
            raise AssertionError(f"{msg} (TRITON_SWMM_REQUIRE_EXAMPLE_DATA=1)")
        pytest.skip(msg)


# Per-variant expected independent_vars. The coupled and triton-only variants
# share full_benchmarking_experiment_uva.xlsx and the shared report config
# (3 axes); the swmm-only variant uses the CPU-only swmm XLSX + its own report
# config restricted to [n_devices]. requires_software_dirs flags the variants
# whose sensitivity XLSX carry system.* overlay rows (which revalidate against
# the compiled swmm/triton software dirs); the swmm-only XLSX has none.
_SHARED_UVA_VARS = ["n_devices", "system.target_dem_resolution", "system.gpu_hardware"]
_SWMM_VARS = ["n_devices"]


@pytest.mark.slow
@pytest.mark.parametrize(
    "factory_name, expected_independent_vars, requires_software_dirs",
    [
        ("benchmarking_norfolk_irene", _SHARED_UVA_VARS, True),
        ("benchmarking_norfolk_irene_triton_only", _SHARED_UVA_VARS, True),
        ("benchmarking_norfolk_irene_swmm_only", _SWMM_VARS, False),
    ],
)
def test_uva_benchmarking_factory_populates_report_sensitivity(
    factory_name: str,
    expected_independent_vars: list[str],
    requires_software_dirs: bool,
    example_data_available: bool,
) -> None:
    """Each UVA benchmarking factory must produce a case whose cfg_analysis.report.sensitivity is populated."""
    if "system.gpu_hardware" in expected_independent_vars:
        pytest.xfail(
            "Phase-4 retired gpu_hardware off system_config; the `system.gpu_hardware` "
            "sensitivity overlay column is rejected by the column allowlist. Re-enabling "
            "needs the experiment-definition migration (the UVA benchmarking CSV's axis "
            "moves from `system.gpu_hardware` to `analysis.hpc_ensemble_partition`, with "
            "gpu_hardware DERIVED per-partition) + anonymized UVA example hpc_system_config "
            "profiles declaring those partitions (Phase-5 example-profiles work) — beyond "
            "the 4d field-retirement. Remove this xfail when that lands."
        )
    if requires_software_dirs:
        _require_software_dirs()
    factory = getattr(cat.UVACaseStudies, factory_name)
    case = factory(start_from_scratch=True, download_if_exists=False)
    analysis = case.analysis

    # In-memory Pydantic model assertion
    sens = analysis.cfg_analysis.report.sensitivity
    assert sens is not None, (
        f"{factory_name}: cfg_analysis.report.sensitivity is None — the bug being guarded against has regressed."
    )
    assert sens.independent_vars == expected_independent_vars, (
        f"{factory_name}: independent_vars={sens.independent_vars} (expected {expected_independent_vars})"
    )
    # NOTE: SensitivityReportConfig.mode retired (ADR-5 ReportingSet registry).
    # Reporting-set selection now lives on report_config.reporting_set, which
    # resolves to the "benchmarking" set at analysis.run() entry for sensitivity
    # analyses — there is no per-config `mode` field to assert.

    # On-disk cfg_analysis.yaml assertion
    cfg_path = Path(analysis.analysis_config_yaml)
    assert cfg_path.exists(), f"{factory_name}: cfg_analysis.yaml not written at {cfg_path}"
    cfg_data = yaml.safe_load(cfg_path.read_text())
    report_block = cfg_data.get("report")
    assert isinstance(report_block, dict), f"{factory_name}: cfg_analysis.yaml::report is not a dict: {report_block!r}"
    on_disk_sens = report_block.get("sensitivity")
    assert on_disk_sens is not None, (
        f"{factory_name}: cfg_analysis.yaml::report.sensitivity is null on disk — "
        "the dominant write-site bug has regressed."
    )
    assert on_disk_sens.get("independent_vars") == expected_independent_vars
