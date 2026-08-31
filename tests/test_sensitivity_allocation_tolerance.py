"""Tolerance of the sensitivity + regular allocation/status chain to members
(or model run-rules) missing from the on-disk Snakefile.

Regression coverage for the partial-completion reprocess crash where
`sensitivity.members` is a strict superset of the Snakefile's
`simulation_member_*` rules (observed 2026-06-02, uva_sensitivity_suite 37/49).
See library/docs/planning/projects/hhemt/bugs/sensitivity
consolidation tolerates members missing from snakefile.md.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from hhemt.snakemake_snakefile_parsing import (
    SnakefileParsingError,
    parse_sensitivity_analysis_workflow_model_allocations,
)

# --- helpers ---------------------------------------------------------------

_SIM_RULE_BLOCK_RE = re.compile(r"(?ms)^rule simulation_member_(?P<member>.+?)_evt_(?P<evt>.+?):.*?(?=^rule \S|\Z)")


def _write_minimal_sensitivity_snakefile(path: Path, member_ids: list[str]) -> None:
    """Write a Snakefile whose only parseable content is one
    `rule simulation_member_{id}_evt_0` block per id, each carrying the
    `tasks` / `cpus_per_task` resources the parser requires."""
    blocks = []
    for member in member_ids:
        blocks.append(
            f"rule simulation_member_{member}_evt_0:\n"
            f"    output: '_status/c_run_member-{member}_evt-0_complete.flag'\n"
            f"    resources:\n"
            f"        tasks=1,\n"
            f"        cpus_per_task=1,\n"
            f"    shell: 'true'\n"
        )
    path.write_text("\n".join(blocks))


def _delete_sim_rule_blocks(snakefile_path: Path, member_ids_to_drop: set[str]) -> None:
    """Delete every `rule simulation_member_{id}_evt_*` block whose member id is in
    member_ids_to_drop from the on-disk Snakefile (faithful to the text-regex the
    parser uses)."""
    text = snakefile_path.read_text()

    def _maybe_drop(m: re.Match) -> str:
        return "" if m.group("member") in member_ids_to_drop else m.group(0)

    snakefile_path.write_text(_SIM_RULE_BLOCK_RE.sub(_maybe_drop, text))


# --- parser-level tests (no fixture) ---------------------------------------


def test_parser_strict_true_raises_on_missing(tmp_path: Path) -> None:
    sf = tmp_path / "Snakefile"
    _write_minimal_sensitivity_snakefile(sf, ["0", "1"])
    with pytest.raises(SnakefileParsingError):
        parse_sensitivity_analysis_workflow_model_allocations(
            snakefile_path=sf,
            expected_analysis_ids=["0", "1", "2"],
            strict=True,
        )


def test_parser_strict_false_tolerates_missing(tmp_path: Path) -> None:
    sf = tmp_path / "Snakefile"
    _write_minimal_sensitivity_snakefile(sf, ["0", "1"])
    allocs = parse_sensitivity_analysis_workflow_model_allocations(
        snakefile_path=sf,
        expected_analysis_ids=["0", "1", "2"],
        strict=False,
    )
    assert set(allocs.keys()) == {"0", "1"}  # missing "2" tolerated, not raised


def test_parser_strict_defaults_true(tmp_path: Path) -> None:
    sf = tmp_path / "Snakefile"
    _write_minimal_sensitivity_snakefile(sf, ["0"])
    with pytest.raises(SnakefileParsingError):
        parse_sensitivity_analysis_workflow_model_allocations(snakefile_path=sf, expected_analysis_ids=["0", "1"])


# --- property-level tests (synth sensitivity builder fixture) --------------


@pytest.fixture(autouse=True)
def _restore_full_snakefile_after_property_test(synth_sensitivity_builder):
    """Session-shared fixture safety: the property tests below mutate the
    on-disk master Snakefile in the session-scoped analysis_dir. Regenerate the
    full master Snakefile after each test so no later-collected test inherits a
    sim-rule-deleted Snakefile (cross-module order-independence)."""
    yield
    content = synth_sensitivity_builder.sensitivity._workflow_builder.generate_master_snakefile_content()
    sf = synth_sensitivity_builder.analysis_paths.analysis_dir / "Snakefile"
    sf.write_text(content)


def _generate_and_write_master_snakefile(analysis) -> Path:
    """Generate the master sensitivity Snakefile content and write it to the
    analysis_dir / 'Snakefile' that df_snakemake_allocations reads."""
    content = analysis.sensitivity._workflow_builder.generate_master_snakefile_content()
    sf = analysis.analysis_paths.analysis_dir / "Snakefile"
    sf.write_text(content)
    return sf


def test_df_snakemake_allocations_tolerates_missing_member(synth_sensitivity_builder) -> None:
    analysis = synth_sensitivity_builder
    sf = _generate_and_write_master_snakefile(analysis)
    all_member_ids = sorted(analysis.sensitivity.members.keys())
    drop = {all_member_ids[-1]}  # remove the last member's sim-rule block
    _delete_sim_rule_blocks(sf, drop)

    df = analysis._df_snakemake_allocations  # must NOT raise
    assert isinstance(df, pd.DataFrame)
    # Rows for dropped members are skipped (their scenario_directory absent)
    assert len(df) > 0


def test_df_status_tolerates_and_annotates_missing_member(synth_sensitivity_builder) -> None:
    analysis = synth_sensitivity_builder
    sf = _generate_and_write_master_snakefile(analysis)
    all_member_ids = sorted(analysis.sensitivity.members.keys())
    drop = {all_member_ids[-1]}
    _delete_sim_rule_blocks(sf, drop)

    df = analysis.df_status  # must NOT raise (R1/R8)
    assert isinstance(df, pd.DataFrame)
    assert "snakemake_allocation_parse_error" in df.columns
    # At least one row carries the parse-error annotation for the dropped member (R5)
    annotated = df["snakemake_allocation_parse_error"].dropna()
    assert (annotated.astype(str).str.contains("not run")).any()


def test_all_present_no_regression(synth_sensitivity_builder) -> None:
    """R6: when the Snakefile matches members, full allocations, no parse_error."""
    analysis = synth_sensitivity_builder
    _generate_and_write_master_snakefile(analysis)  # no deletion

    df = analysis.df_status
    assert isinstance(df, pd.DataFrame)
    # No row carries a parse-error string in the all-present case
    parse_err = df.get("snakemake_allocation_parse_error")
    if parse_err is not None:
        assert parse_err.dropna().empty
