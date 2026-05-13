"""Structural tests for bundle/snakefile_generator.py (Plan Phase 2 VMS-10).

Per Plan Phase 2 D5: these tests parse the emitted Snakefile as a string
and run assertion-based checks. End-to-end execution validation against
a bundle is the Phase 6 smoketest's scope.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Literal

import pytest

from TRITON_SWMM_toolkit.bundle.snakefile_generator import (
    generate_regeneration_snakefile,
    write_regeneration_snakefile,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "bundles"
MULTI_SIM_FIXTURE = FIXTURES_DIR / "multi_sim"
SENSITIVITY_FIXTURE = FIXTURES_DIR / "sensitivity_master"

REGEN_RULE_SET = {
    "all",
    "render_report",
    "plot_system_overview",
    "plot_per_sim_peak_flood_depth",
    "plot_per_sim_conduit_flow",
    "plot_per_analysis_summary_table",
    "plot_scenario_status_appendix",
    "plot_errors_and_warnings",
    "plot_sensitivity_benchmarking",
    "plot_per_sim_per_sa_peak_flood_depth",
    "plot_per_sim_per_sa_conduit_flow",
}

NON_REGEN_RULES = {
    "setup",
    "prepare_scenario",
    "run_triton",
    "run_tritonswmm",
    "run_swmm",
    "process_triton",
    "process_tritonswmm",
    "process_swmm",
    "consolidate",
    "consolidate_master",
}


@pytest.fixture
def multi_sim_bundle(tmp_path: Path) -> Path:
    dest = tmp_path / "multi_sim"
    shutil.copytree(MULTI_SIM_FIXTURE, dest)
    return dest


@pytest.fixture
def sensitivity_bundle(tmp_path: Path) -> Path:
    if not SENSITIVITY_FIXTURE.exists():
        pytest.skip(f"sensitivity_master fixture missing at {SENSITIVITY_FIXTURE}")
    dest = tmp_path / "sensitivity_master"
    shutil.copytree(SENSITIVITY_FIXTURE, dest)
    return dest


def _extract_rule_names(text: str) -> set[str]:
    return set(re.findall(r"^rule\s+(\w+):", text, re.MULTILINE))


@pytest.mark.parametrize(
    "bundle_fixture",
    ["multi_sim_bundle", "sensitivity_bundle"],
)
def test_regeneration_scoped_rule_set(
    bundle_fixture: str, request: pytest.FixtureRequest
) -> None:
    """Emitted rules are a subset of REGEN_RULE_SET; no NON_REGEN_RULES."""
    bundle = request.getfixturevalue(bundle_fixture)
    text = generate_regeneration_snakefile(bundle, static_backend="matplotlib")
    rule_names = _extract_rule_names(text)
    assert rule_names.issubset(REGEN_RULE_SET), (
        f"Unexpected rules emitted: {rule_names - REGEN_RULE_SET}"
    )
    assert not (rule_names & NON_REGEN_RULES), (
        f"Forbidden simulation/processing rules emitted: "
        f"{rule_names & NON_REGEN_RULES}"
    )


@pytest.mark.parametrize(
    "bundle_fixture",
    ["multi_sim_bundle", "sensitivity_bundle"],
)
def test_no_absolute_paths_in_rule_shells(
    bundle_fixture: str, request: pytest.FixtureRequest
) -> None:
    """No /sfs/ /scratch/ /work/ /home/ substrings anywhere in the emitted text."""
    bundle = request.getfixturevalue(bundle_fixture)
    text = generate_regeneration_snakefile(bundle, static_backend="matplotlib")
    for marker in ("/sfs/", "/scratch/", "/work/", "/home/"):
        assert marker not in text, f"Absolute-path leak: {marker!r}"


@pytest.mark.parametrize(
    "static_backend,expected_ext_for_system_overview",
    [
        ("matplotlib", ".png"),
        ("plotly", ".svg"),
    ],
)
def test_static_backend_controls_output_ext(
    multi_sim_bundle: Path,
    static_backend: Literal["matplotlib", "plotly"],
    expected_ext_for_system_overview: str,
) -> None:
    """static_backend toggles per-renderer output_ext at the rule's output: site."""
    text = generate_regeneration_snakefile(
        multi_sim_bundle, static_backend=static_backend
    )
    pattern = (
        r"rule\s+plot_system_overview:.*?output:.*?"
        rf"\"plots/system_overview\{expected_ext_for_system_overview}\""
    )
    assert re.search(pattern, text, re.DOTALL), (
        f"Expected system_overview output {expected_ext_for_system_overview!r} "
        f"for static_backend={static_backend!r}"
    )


def test_writes_to_bundle_root_snakefile(multi_sim_bundle: Path) -> None:
    """write_regeneration_snakefile returns path = {bundle_root}/Snakefile."""
    out = write_regeneration_snakefile(multi_sim_bundle, static_backend="matplotlib")
    assert out == multi_sim_bundle / "Snakefile"
    assert out.exists()
    assert out.read_text().strip()


def test_output_ext_propagates_to_all_three_sites(multi_sim_bundle: Path) -> None:
    """Three-place output_ext coupling under static_backend='plotly'."""
    text = generate_regeneration_snakefile(multi_sim_bundle, static_backend="plotly")
    # Site 1: rule output:
    assert re.search(
        r"rule\s+plot_system_overview:.*?output:.*?\"plots/system_overview\.svg\"",
        text,
        re.DOTALL,
    ), "system_overview output not .svg under plotly"
    # Site 2: rule report(...) first arg (output is wrapped in report)
    assert re.search(
        r"report\(\s*\n\s*\"plots/system_overview\.svg\"",
        text,
    ), "report() first arg not .svg under plotly"
    # Site 3a: rule all input list
    rule_all_match = re.search(
        r"rule\s+all:\s*\n\s*input:(.*?)(?=\nrule\s)",
        text,
        re.DOTALL,
    )
    assert rule_all_match and (
        "plots/system_overview.svg" in rule_all_match.group(1)
    ), "rule all does not reference .svg system_overview"
    # Site 3b: render_report input list
    render_match = re.search(
        r"rule\s+render_report:\s*\n\s*input:(.*?)(?=\n\s*output:)",
        text,
        re.DOTALL,
    )
    assert render_match and (
        "plots/system_overview.svg" in render_match.group(1)
    ), "render_report does not reference .svg system_overview"


def test_preamble_preserved_for_jinja2_conditionals(multi_sim_bundle: Path) -> None:
    """The emitted body contains the import preamble and _toolkit_version fallback."""
    text = generate_regeneration_snakefile(multi_sim_bundle, static_backend="matplotlib")
    assert "from importlib.metadata import version as _pkg_version" in text
    assert '_toolkit_version = _pkg_version("TRITON_SWMM_toolkit")' in text
    assert 'config["toolkit_version"] = _toolkit_version' in text


@pytest.mark.parametrize(
    "bundle_fixture,expected_keys",
    [
        (
            "multi_sim_bundle",
            ['"analysis_id"', '"toolkit_version"', '"n_sims"', '"is_sensitivity"', '"report"'],
        ),
        (
            "sensitivity_bundle",
            [
                '"analysis_id"', '"toolkit_version"', '"n_sims"', '"is_sensitivity"',
                '"n_sub_analyses"', '"independent_vars"', '"report"',
            ],
        ),
    ],
)
def test_jinja2_config_keys_covered(
    bundle_fixture: str,
    expected_keys: list[str],
    request: pytest.FixtureRequest,
) -> None:
    """Every mandatory + present-conditional config[...] key is assigned."""
    bundle = request.getfixturevalue(bundle_fixture)
    text = generate_regeneration_snakefile(bundle, static_backend="matplotlib")
    for key in expected_keys:
        assert f"config[{key}]" in text, (
            f"Missing config[{key}] assignment in {bundle_fixture}"
        )


@pytest.mark.parametrize(
    "bundle_fixture",
    ["multi_sim_bundle", "sensitivity_bundle"],
)
def test_render_report_inputs_match_rule_all_plot_outputs(
    bundle_fixture: str, request: pytest.FixtureRequest
) -> None:
    """render_report's input list must be the plot-output subset of rule all's input list."""
    bundle = request.getfixturevalue(bundle_fixture)
    text = generate_regeneration_snakefile(bundle, static_backend="matplotlib")
    rule_all_inputs = _extract_input_block(text, "all")
    render_report_inputs = _extract_input_block(text, "render_report")
    plot_outputs_in_rule_all = {
        line.strip().strip(",").strip('"')
        for line in rule_all_inputs.splitlines()
        if line.strip().startswith('"plots/')
    }
    render_inputs = {
        line.strip().strip(",").strip('"')
        for line in render_report_inputs.splitlines()
        if line.strip().startswith('"plots/')
    }
    assert render_inputs == plot_outputs_in_rule_all, (
        f"render_report inputs diverge from rule all plot outputs:\n"
        f"  only in rule all: {plot_outputs_in_rule_all - render_inputs}\n"
        f"  only in render_report: {render_inputs - plot_outputs_in_rule_all}"
    )


@pytest.mark.parametrize(
    "bundle_fixture",
    ["multi_sim_bundle", "sensitivity_bundle"],
)
def test_top_level_report_directive_present(
    bundle_fixture: str, request: pytest.FixtureRequest
) -> None:
    """The top-level report: directive is required for snakemake --report
    to render the Jinja2 workflow_description template.
    """
    bundle = request.getfixturevalue(bundle_fixture)
    text = generate_regeneration_snakefile(bundle, static_backend="matplotlib")
    assert 'report: "report/workflow_description.rst"' in text


def _extract_input_block(text: str, rule_name: str) -> str:
    pattern = (
        rf"rule\s+{re.escape(rule_name)}:\s*\n\s*input:"
        r"(.*?)(?=\n\s*(?:output:|onsuccess:|onerror:|rule\s))"
    )
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise AssertionError(f"Rule {rule_name!r} not found in emitted text")
    return match.group(1)
