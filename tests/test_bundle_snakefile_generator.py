"""Structural tests for bundle/snakefile_generator.py (Plan Phase 2 VMS-10).

Per Plan Phase 2 D5: these tests parse the emitted Snakefile as a string
and run assertion-based checks. End-to-end execution validation against
a bundle is the Phase 6 smoketest's scope.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Literal

import pytest

from hhemt.bundle.snakefile_generator import (
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
    "plot_per_sim_event_page",
    "plot_per_analysis_summary_table",
    "plot_scenario_status_appendix",
    "plot_errors_and_warnings",
    "plot_disk_utilization",  # P1b: registry-driven bundle now emits it (drift-fix)
    "plot_metadata",  # ADR-14 / C10: auto-carried into the bundle via _TMPL_METADATA
    # [Q160](7): the run-timeline + SLURM-efficiency page extracted out of metadata.
    # Auto-carried into the bundle via _TMPL_WORKFLOW_PERFORMANCE, zero generator edits.
    "plot_workflow_performance",
    "plot_sensitivity_benchmarking",
    "plot_per_sim_per_sa_peak_flood_depth",
    "plot_per_sim_per_sa_conduit_flow",
    # EDA rules: emitted only for a bundle whose cfg_analysis selects a
    # reporting set carrying the eda_compute_sensitivity renderer. Absent from
    # the default/benchmarking fixtures, so the subset assertion above is
    # unaffected; present here so a dem-resolution or compute-sensitivity
    # bundle does not trip it.
    "plot_eda_compute_sensitivity",
    "plot_eda_dem_resolution_cost_error",
    "plot_eda_dem_resolution_error_ecdf",
    "plot_eda_dem_resolution_diff_maps",
    "plot_eda_dem_resolution_coupling_table",
    # b4b: emitted only for a bundle whose cfg_analysis selects reporting_set="b4b".
    "plot_b4b_clean_identity",
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
        msg = f"sensitivity_master bundle fixture missing at {SENSITIVITY_FIXTURE}"
        if os.environ.get("TRITON_SWMM_REQUIRE_BUNDLE_FIXTURE") == "1":
            raise AssertionError(
                f"{msg} (TRITON_SWMM_REQUIRE_BUNDLE_FIXTURE=1) — the checked-in bundle fixture must be present in CI."
            )
        pytest.skip(msg)
    dest = tmp_path / "sensitivity_master"
    shutil.copytree(SENSITIVITY_FIXTURE, dest)
    return dest


def _extract_rule_names(text: str) -> set[str]:
    return set(re.findall(r"^rule\s+(\w+):", text, re.MULTILINE))


@pytest.mark.parametrize(
    "bundle_fixture",
    ["multi_sim_bundle", "sensitivity_bundle"],
)
def test_regeneration_scoped_rule_set(bundle_fixture: str, request: pytest.FixtureRequest) -> None:
    """Emitted rules are a subset of REGEN_RULE_SET; no NON_REGEN_RULES."""
    bundle = request.getfixturevalue(bundle_fixture)
    text = generate_regeneration_snakefile(bundle, static_backend="matplotlib")
    rule_names = _extract_rule_names(text)
    assert rule_names.issubset(REGEN_RULE_SET), f"Unexpected rules emitted: {rule_names - REGEN_RULE_SET}"
    assert not (rule_names & NON_REGEN_RULES), (
        f"Forbidden simulation/processing rules emitted: {rule_names & NON_REGEN_RULES}"
    )


@pytest.mark.parametrize(
    "bundle_fixture",
    ["multi_sim_bundle", "sensitivity_bundle"],
)
def test_disk_utilization_rule_present(bundle_fixture: str, request: pytest.FixtureRequest) -> None:
    """P1b BUNDLE DRIFT NOTE: the registry-driven bundle now emits
    plot_disk_utilization, closing the pre-P1b source-vs-bundle drift where the
    source-side multisim/master generators emitted disk_utilization but the bundle
    did not (regeneration-only metadata; no analysis rerun)."""
    bundle = request.getfixturevalue(bundle_fixture)
    text = generate_regeneration_snakefile(bundle, static_backend="matplotlib")
    assert "plot_disk_utilization" in _extract_rule_names(text), (
        "disk_utilization rule missing from the regeneration Snakefile — the "
        "registry-driven harvest should emit it for every shipped set."
    )


@pytest.mark.parametrize(
    "bundle_fixture",
    ["multi_sim_bundle", "sensitivity_bundle"],
)
def test_no_absolute_paths_in_rule_shells(bundle_fixture: str, request: pytest.FixtureRequest) -> None:
    """No /sfs/ /scratch/ /work/ /home/ substrings anywhere in the emitted text."""
    bundle = request.getfixturevalue(bundle_fixture)
    text = generate_regeneration_snakefile(bundle, static_backend="matplotlib")
    for marker in ("/sfs/", "/scratch/", "/work/", "/home/"):
        assert marker not in text, f"Absolute-path leak: {marker!r}"


@pytest.mark.parametrize(
    "static_backend,expected_ext_for_system_overview",
    [
        ("matplotlib", ".png"),
        ("plotly", ".html"),
    ],
)
def test_static_backend_controls_output_ext(
    multi_sim_bundle: Path,
    static_backend: Literal["matplotlib", "plotly"],
    expected_ext_for_system_overview: str,
) -> None:
    """static_backend toggles per-renderer output_ext at the rule's output: site."""
    text = generate_regeneration_snakefile(multi_sim_bundle, static_backend=static_backend)
    pattern = (
        r"rule\s+plot_system_overview:.*?output:.*?" rf"\"plots/system_overview\{expected_ext_for_system_overview}\""
    )
    assert re.search(pattern, text, re.DOTALL), (
        f"Expected system_overview output {expected_ext_for_system_overview!r} for static_backend={static_backend!r}"
    )


def test_writes_to_bundle_root_snakefile(multi_sim_bundle: Path) -> None:
    """write_regeneration_snakefile returns path = {bundle_root}/Snakefile."""
    out = write_regeneration_snakefile(multi_sim_bundle, static_backend="matplotlib")
    assert out == multi_sim_bundle / "Snakefile"
    assert out.exists()
    assert out.read_text().strip()


@pytest.mark.parametrize(
    "static_backend,ext",
    [
        ("matplotlib", ".png"),
        ("plotly", ".html"),
    ],
)
def test_output_ext_propagates_to_all_three_sites(
    multi_sim_bundle: Path,
    static_backend: Literal["matplotlib", "plotly"],
    ext: str,
) -> None:
    """Three-place output_ext coupling: rule output, report() first arg, rule_all + render_report inputs."""
    text = generate_regeneration_snakefile(multi_sim_bundle, static_backend=static_backend)
    # Site 1: rule output:
    assert re.search(
        rf"rule\s+plot_system_overview:.*?output:.*?\"plots/system_overview{re.escape(ext)}\"",
        text,
        re.DOTALL,
    ), f"system_overview output not {ext} under {static_backend}"
    # Site 2: rule report(...) first arg (output is wrapped in report)
    assert re.search(
        rf"report\(\s*\n\s*\"plots/system_overview{re.escape(ext)}\"",
        text,
    ), f"report() first arg not {ext} under {static_backend}"
    # Site 3a: rule all input list
    rule_all_match = re.search(
        r"rule\s+all:\s*\n\s*input:(.*?)(?=\nrule\s)",
        text,
        re.DOTALL,
    )
    assert rule_all_match and (f"plots/system_overview{ext}" in rule_all_match.group(1)), (
        f"rule all does not reference {ext} system_overview"
    )
    # Site 3b: render_report input list
    render_match = re.search(
        r"rule\s+render_report:\s*\n\s*input:(.*?)(?=\n\s*output:)",
        text,
        re.DOTALL,
    )
    assert render_match and (f"plots/system_overview{ext}" in render_match.group(1)), (
        f"render_report does not reference {ext} system_overview"
    )


def test_preamble_preserved_for_jinja2_conditionals(multi_sim_bundle: Path) -> None:
    """The emitted body contains the import preamble and _toolkit_version fallback."""
    text = generate_regeneration_snakefile(multi_sim_bundle, static_backend="matplotlib")
    assert "from importlib.metadata import version as _pkg_version" in text
    assert '_toolkit_version = _pkg_version("hhemt")' in text
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
                '"analysis_id"',
                '"toolkit_version"',
                '"n_sims"',
                '"is_sensitivity"',
                '"n_sub_analyses"',
                '"independent_vars"',
                '"report"',
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
        assert f"config[{key}]" in text, f"Missing config[{key}] assignment in {bundle_fixture}"


@pytest.mark.parametrize(
    "bundle_fixture",
    ["multi_sim_bundle", "sensitivity_bundle"],
)
def test_render_report_inputs_match_rule_all_plot_outputs(bundle_fixture: str, request: pytest.FixtureRequest) -> None:
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
def test_top_level_report_directive_present(bundle_fixture: str, request: pytest.FixtureRequest) -> None:
    """The top-level report: directive is required for snakemake --report
    to render the Jinja2 workflow_description template.
    """
    bundle = request.getfixturevalue(bundle_fixture)
    text = generate_regeneration_snakefile(bundle, static_backend="matplotlib")
    assert 'report: "report/workflow_description.rst"' in text


def _extract_input_block(text: str, rule_name: str) -> str:
    pattern = rf"rule\s+{re.escape(rule_name)}:\s*\n\s*input:" r"(.*?)(?=\n\s*(?:output:|onsuccess:|onerror:|rule\s))"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise AssertionError(f"Rule {rule_name!r} not found in emitted text")
    return match.group(1)


# ---------------------------------------------------------------------------
# Reporting-set routing (Plan Phase 5 / D14)
#
# These lock behavior that ALREADY SHIPPED upstream (commit 9face09) with no
# bundle-side coverage: the harvest resolves the active reporting set from
# cfg_analysis rather than hardcoding benchmarking-or-default, so a bundle whose
# source run selected a non-default set regenerates the SAME renderer set the
# source side emitted. They are a regression lock, not a proof of new behavior --
# their teeth come from the phase's verify-by-deletion step (revert the harvest's
# config-read to the hardcode and these must fail).
# ---------------------------------------------------------------------------


def _set_reporting_set(bundle_root: Path, name: str) -> None:
    """Point a copied bundle fixture's cfg_analysis at reporting set `name`."""
    import yaml

    cfg_path = bundle_root / "cfg_analysis.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    cfg.setdefault("report", {})
    cfg["report"]["reporting_set"] = name
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _eda_rule_names(text: str) -> set[str]:
    return {r for r in _extract_rule_names(text) if r.startswith("plot_eda_")}


def test_dem_resolution_bundle_emits_all_four_eda_rules(sensitivity_bundle: Path) -> None:
    """A dem-resolution bundle regenerates all four DEM EDA rules.

    Membership PARITY against the registry, not a bare count: a count would pass
    if the harvest emitted four rules with the wrong names.
    """
    from hhemt.report_renderers._reporting_sets import get_reporting_set

    _set_reporting_set(sensitivity_bundle, "dem-resolution")
    text = generate_regeneration_snakefile(sensitivity_bundle, static_backend="matplotlib")

    expected = {
        f"plot_{tmpl.rule_name}" if not tmpl.rule_name.startswith("plot_") else tmpl.rule_name
        for sel in get_reporting_set("dem-resolution").renderer_selection
        if sel.builder_key == "eda_compute_sensitivity"
        for tmpl in sel.rule_spec_template
    }
    assert _eda_rule_names(text) == expected, (
        f"Emitted EDA rules {_eda_rule_names(text)} != registry templates {expected}"
    )
    assert len(expected) == 4, f"dem-resolution should carry four EDA templates, got {len(expected)}"


def test_compute_sensitivity_bundle_emits_its_eda_rule(sensitivity_bundle: Path) -> None:
    """The SHIPPED compute-sensitivity set regenerates its one EDA rule.

    This is the set whose EDA figure was silently omitted from every regenerated
    bundle before the harvest read the reporting set from config.
    """
    _set_reporting_set(sensitivity_bundle, "compute-sensitivity")
    text = generate_regeneration_snakefile(sensitivity_bundle, static_backend="matplotlib")
    assert _eda_rule_names(text) == {"plot_eda_compute_sensitivity"}


def test_b4b_bundle_emits_config_diff_maps_eda_rule(sensitivity_bundle: Path) -> None:
    """A b4b bundle regenerates the config_diff_maps EDA rule (Thread-C wiring).

    Enumeration/emission parity guard for the report-wiring edit: switching a
    bundle to the b4b set must emit ``plot_eda_compute_sensitivity`` in the
    regeneration Snakefile (it was absent under the ``benchmarking`` set the driver
    used before). This exercises the emission path so a listed-but-unemitted output
    would surface here rather than as a ``MissingInputException`` at snakemake parse.
    """
    _set_reporting_set(sensitivity_bundle, "b4b")
    text = generate_regeneration_snakefile(sensitivity_bundle, static_backend="matplotlib")
    assert "plot_eda_compute_sensitivity" in _eda_rule_names(text), (
        f"b4b bundle did not emit the config_diff_maps EDA rule; emitted: "
        f"{_eda_rule_names(text)}"
    )


def test_benchmarking_bundle_emits_no_eda_rules(sensitivity_bundle: Path) -> None:
    """The default (sentinel) sensitivity bundle carries no EDA rule.

    The negative control: without it, a harvest that emitted EDA rules
    unconditionally would satisfy both positive tests above.
    """
    text = generate_regeneration_snakefile(sensitivity_bundle, static_backend="matplotlib")
    assert _eda_rule_names(text) == set()


def test_unknown_reporting_set_raises_named_configuration_error(sensitivity_bundle: Path) -> None:
    """A typo'd reporting_set names the field, not a bare KeyError.

    Regression lock on the validation half of the resolution rule: the harvest
    delegates to config.report.resolve_reporting_set_name, so an unknown set is
    rejected with a ConfigurationError listing the registered sets instead of
    reaching the bare get_reporting_set accessor and raising KeyError.
    """
    from hhemt.exceptions import ConfigurationError

    _set_reporting_set(sensitivity_bundle, "dem-resolutoin")  # deliberate typo
    with pytest.raises(ConfigurationError) as excinfo:
        generate_regeneration_snakefile(sensitivity_bundle, static_backend="matplotlib")
    message = str(excinfo.value)
    assert "reporting_set" in message
    assert "dem-resolutoin" in message
    assert "dem-resolution" in message, "the error should list the registered sets"


@pytest.fixture
def pure_triton_sensitivity_bundle(sensitivity_bundle: Path) -> Path:
    """A SINGLE-MODEL (pure-TRITON) sensitivity bundle whose conduit family is empty.

    Derived from the checked-in coupled fixture rather than added as a second checked-in
    tree: the only differences that matter are the model toggles and the absence of the
    conduit artifacts, and expressing them here keeps the single-model-ness visible in the
    test instead of buried in a fixture directory.

    Deleting the conduit *.manifest.json sidecars is load-bearing. The coupled fixture
    carries sidecars but no conduit .html, so _expand_wildcards_to_existing_files' SIDECAR
    fallback succeeds on it and the double-miss branch is never reached — which is why the
    existing suite is blind to this defect.
    """
    import yaml as _yaml

    sys_path = sensitivity_bundle / "cfg_system.yaml"
    cfg_system = _yaml.safe_load(sys_path.read_text())
    cfg_system["toggle_triton_model"] = True
    cfg_system["toggle_tritonswmm_model"] = False
    cfg_system["toggle_swmm_model"] = False
    sys_path.write_text(_yaml.safe_dump(cfg_system))

    for stale in sensitivity_bundle.rglob("*conduit*"):
        stale.unlink()
    return sensitivity_bundle


def _input_block(text: str, rule: str) -> str:
    """The `input:` block of `rule {rule}`, up to the next directive or rule."""
    body = text.split(f"rule {rule}:", 1)[1]
    body = body.split("input:", 1)[1]
    out: list[str] = []
    for line in body.splitlines():
        if line.strip() and not line.startswith((" ", "\t")):
            break
        if re.match(r"\s+(output|shell|log|params|resources|run|report):", line):
            break
        out.append(line)
    return "\n".join(out)


def test_single_model_empty_family_emits_no_wildcards(pure_triton_sensitivity_bundle: Path) -> None:
    """A family that expands to nothing is OMITTED, never emitted wildcarded.

    rule all and render_report cannot carry wildcards; a raw templated path in either is
    a WildcardError at `snakemake --touch`. Asserted on the BRACE COUNT rather than by
    invoking Snakemake, so this stays a unit test and still discriminates: pre-fix both
    blocks contain {sa_id}/{event_id}, post-fix neither does.
    """
    text = generate_regeneration_snakefile(pure_triton_sensitivity_bundle, static_backend="plotly")

    for rule in ("all", "render_report"):
        block = _input_block(text, rule)
        assert "{" not in block, f"rule {rule} input block carries a wildcard: {block!r}"

    # Distinguishes the harvest gate from the expansion fix alone: with only the latter,
    # rule all is clean but the orphan conduit rule is still emitted.
    assert "rule plot_per_sim_per_sa_conduit_flow:" not in text


def test_coupled_bundle_emission_unaffected_by_the_gate(sensitivity_bundle: Path) -> None:
    """The satisfying arm: the gate does not fire on a SWMM-bearing bundle.

    Guards against over-firing — a fix for the empty case narrowing the populated one.

    Asserts RULE PRESENCE, not expansion contents. Expansion is the wrong probe on this
    fixture: its per-sim manifest sidecars are named `conduit_flow.manifest.json` (the bare
    renderer name) while the glob builds `conduit_flow__sa.*__evt.*.manifest.json`, so BOTH
    per-sim families expand to nothing here for a NAMING reason unrelated to model gating.
    An earlier version of this test asserted `"conduit_flow" in` the rule-all input block
    and passed only because the WILDCARDED TEMPLATE contains that substring — i.e. it was
    asserting on an artifact of the very bug under repair. Rule presence is what the
    predicate gate actually controls, and it survives a later fixture rename.
    """
    text = generate_regeneration_snakefile(sensitivity_bundle, static_backend="plotly")

    assert "rule plot_per_sim_per_sa_conduit_flow:" in text
    assert "{" not in _input_block(text, "all")
